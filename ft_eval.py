import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torcheval.metrics.functional import mean_squared_error
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18
import time
from copy import deepcopy
import argparse
import os
import pickle as pkl
from models import NONA_FT
import time
from tqdm import tqdm
import sys

def tensor(arr):
    if type(arr) != torch.Tensor:
        arr = torch.Tensor(arr)
    return arr.to(dtype=torch.float32, device=device)

class RSNADataset(Dataset):
    def __init__(self, indices, transform=None):
        self.indices = indices
        self.transform = transform

        self.features = pd.read_csv('data/rsna/all_features.csv')
        self.features = self.features[self.features['id'].isin(self.indices)].reset_index()
    
    def __len__(self):
        return len(self.indices)
    
    def __getitem__(self, idx):
        img_path = self.features.loc[idx, 'path']
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        label = self.features.loc[idx, 'boneage norm']
        
        return image, label

def load_data_params(dataset):
    if dataset == 'rsna':
        task = 'regression'
        data_df = pd.read_csv('data/rsna/all_features.csv')
        transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        fe = resnet18
        data_percentage = 0.125
    return task, data_df, transform, fe, data_percentage

def get_fold_indices(data_df, seed, data_percentage=0.25):
    dd = {} # data dict
    
    ids = data_df['id'].values
    sample_size = int(data_percentage * ids.size)
    ids = np.random.choice(ids, size=sample_size, replace=False)
    
    train_val, dd['test'] = train_test_split(ids, test_size=0.25, random_state=seed)
    dd['train'], dd['val'] = train_test_split(train_val, test_size=0.15, random_state=seed)

    return dd

class Score(nn.Module):
    def __init__(self, metric):
        super(Score, self).__init__()
        self.metric = metric

    def forward(self, y_hat, y):
        if self.metric == 'accuracy':
            if y_hat.shape[1] > 1: #multiclass
                y_hat = torch.argmax(y_hat, dim=1)
            else:
                y_hat = torch.round(y_hat)
            
            return (y_hat == y).float().mean().item()
        
        elif self.metric == 'mse':
            return - mean_squared_error(y_hat, y).item() # Negative mse to simplify early stopping code

def collate(batch):
    x, y = zip(*batch)
    x = torch.stack(x).to(device).to(torch.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return x, y

def mlps_train_eval(train, val, test, feature_extractor):
    scores = {}
    
    learning_rate = 1e-5
    
    train_dataset = RSNADataset(train, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
    all_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate) # for use as neighbors with val and test

    val_dataset = RSNADataset(val, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, collate_fn=collate)

    test_dataset = RSNADataset(test, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate)

    for predictor_head in ['nona euclidean', 'nona dot', 'dense']:
        
        print("Training", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        if dataset == 'rsna': # reinitialize weights
            feature_extractor_weights = feature_extractor(weights='DEFAULT')

        hls = [200, 50]
        model = NONA_FT(feature_extractor=feature_extractor_weights, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        task=task, 
                        dtype=torch.float32)
        
        criterion = crit_dict[task][0]()

        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start = time.time()
        patience = 10
        start_after_epoch = 5
        count = 0
        best_val_score = float('-inf')
        epoch = 1
        while count < patience:
            model.train()
            train_loss = 0.0
            print('Epoch:', epoch)
            for batch_X, batch_y in tqdm(train_loader, desc="Train", file=sys.stdout):
                outputs = model(batch_X, batch_X, batch_y)
                loss = criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            report = f"Train Loss: {train_loss: .5f}"

            # Early stopping
            if epoch > start_after_epoch:
                model.eval()
                val_scores = []
                with torch.no_grad():
                    for (X_train, y_train), (X_val, y_val) in tqdm(zip(all_train_loader, val_loader), desc="Val", file=sys.stdout):
                        y_hat_val = model(X_val, X_train, y_train)  
                        val_scores.append(score(y_hat_val, y_val))

                val_score = sum(val_scores) / len(val_scores)

                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = deepcopy(model.state_dict())
                    count = 0
                else:
                    count += 1
                report = report + f': Val Score: {abs(val_score): .5f}'
            
            print(report)
            epoch += 1
        
        print("Evaluating", predictor_head) 
        y_hats = []
        y_tests = []
        model.load_state_dict(best_model_state)
        with torch.no_grad():
            for (X_train, y_train), (X_test, y_test) in tqdm(zip(all_train_loader, test_loader), desc="Test", file=sys.stdout):
                y_hat_batch = model(X_test, X_train, y_train)  
                y_hats.append(y_hat_batch)
                y_tests.append(y_test)

        y_hat = torch.cat(y_hats, dim=0)
        y_test = torch.cat(y_tests, dim=0)
        end = time.time()
        

        scores[f'{predictor_head} mlp'] =  [score(y_hat, y_test), end-start]
        if save_models:
            model_path = f'results/{dataset}/models/{script_start_time}/{predictor_head}_{seed}.pth'
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset and other configs.")
    parser.add_argument('--dataset', type=str, default='Which dataset to load in.', help='Path to data directory.')
    parser.add_argument('--seeds', type=int, default=10, help='How many splits of the data to train and test on.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether or not to save final models')
    args = parser.parse_args()
    dataset = args.dataset
    save_models = args.savemodels
    seeds = args.seeds

    script_start_time = time.strftime("%m%d%H%M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task, data_df, transform, fe, data_percentage = load_data_params(dataset)
    
    crit_dict = {'binary': [nn.BCELoss, 'accuracy'],
                 'multiclass': [nn.CrossEntropyLoss, 'accuracy'],
                 'ordinal': [nn.MSELoss, 'accuracy'],
                 'regression': [nn.MSELoss, 'mse']}
    score = Score(crit_dict[task][1])

    results_path = f'results/{dataset}/scores_{script_start_time}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    scores_list = ["200,50 for visualization"]

    for seed in range(seeds):
        print(f'Training and evaluating models for split {seed+1}.')
        
        idx_dict = get_fold_indices(data_df, seed=seed, data_percentage=data_percentage)

        scores = mlps_train_eval(**idx_dict, feature_extractor=fe)

        for k,v in scores.items():
            if score.metric == 'accuracy':
                test_score = f'{round(100*v[0],3)}%'
            else:
                test_score = f'{-round(v[0],3)}'
            print(f'{k}: {test_score} {score.metric} in {round(v[1],3)}s.')

        scores_list.append(scores)
        
        with open(results_path, "wb") as file:
            pkl.dump(obj=scores_list, file=file)