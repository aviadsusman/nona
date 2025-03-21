import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torcheval.metrics.functional import mean_squared_error
from torcheval.metrics.aggregation.auc import AUC
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
from similarity_masks import SoftKNNMask, HardKNNMask, SoftSimMask, HardSimMask, SoftPointwiseKNN
import time
from tqdm import tqdm
import sys
from utils import Score, load_data_params, get_folds, tune_knn

class RSNADataset(Dataset):
    def __init__(self, indices, transform=None, label_scaler=None):
        super(RSNADataset, self).__init__()
        self.indices = indices
        self.transform = transform
        self.label_scaler = label_scaler

        self.features = pd.read_csv('data/rsna/all_features.csv')
        self.features = self.features[self.features['id'].isin(self.indices)].reset_index()
    
    def __len__(self):
        return len(self.indices)

    def _scale_tensor(self, tensor):
        if not self.transform:
            pixel_sum = 0
            pixel_sq_sum = 0
            num_pixels = 0
            for idx in tqdm(self.indices, desc="tensor scalers", file=sys.stdout):
                img_tensor = torch.load(f'data/rsna/tensors/{idx}.pt').float()
                pixels = img_tensor[0].view(-1)  # Only use the first channel

                pixel_sum += pixels.sum()
                pixel_sq_sum += (pixels ** 2).sum()
                num_pixels += pixels.numel()

            train_mean = [(pixel_sum / num_pixels).item()] * 3 # 3 channels
            train_var = pixel_sq_sum / num_pixels - train_mean[0] ** 2
            train_std = [torch.sqrt(train_var).item()] * 3

            self.transform = transforms.Normalize(mean=train_mean, std=train_std) 

        return self.transform(tensor)

    def _scale_label(self, label):
        if not self.label_scaler:
            labels = self.features['boneage']
            min_label, max_label = labels.min(), labels.max()
            self.scaler = [min_label, max_label]
        else:
            min_label, max_label = self.scaler[0], self.scaler[1]
        
        return (label - min_label) / (max_label - min_label)
    
    def __getitem__(self, idx):
        img_tensor = torch.load(f'data/rsna/tensors/{self.indices[idx]}.pt')
        label = self.features.loc[idx, 'boneage']
        
        return self._scale_tensor(img_tensor) , self._scale_label(label)
def collate(batch):
    x, y = zip(*batch)
    x = torch.stack(x).to(device).to(torch.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return x, y

def mlps_train_eval(train, val, test, feature_extractor):
    scores = {}
    
    if dataset == 'rsna':
        train_dataset = RSNADataset(train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, collate_fn=collate)
        all_train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, collate_fn=collate) # for use as neighbors with val and test

        transform = train_dataset.transform
        scaler = train_dataset.label_scaler

        val_dataset = RSNADataset(val, transform=transform, label_scaler=scaler)
        val_loader = DataLoader(val_dataset, batch_size=len(val_dataset), shuffle=True, collate_fn=collate)

        test_dataset = RSNADataset(test, transform=transform, label_scaler=scaler)
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=True, collate_fn=collate)
    
    elif dataset == 'cifar':
        train_loader = DataLoader(train, batch_size=128, shuffle=True, collate_fn=collate)
        all_train_loader = DataLoader(train, batch_size=len(train), shuffle=False, collate_fn=collate)
        val_loader = DataLoader(val, batch_size=len(val), shuffle=True, collate_fn=collate)
        test_loader = DataLoader(test, batch_size=len(test), shuffle=False, collate_fn=collate)

    for predictor_head in ['nona euclidean', 'nona dot']:#, 'dense']:
        
        print("Training", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        feature_extractor_weights = feature_extractor(weights='DEFAULT')
        mc = 10 if dataset=='cifar' else None
        agg = 'mean' if dataset == 'cifar' else None
        hls = [200, 50]
        mask = SoftPointwiseKNN()
        model = NONA_FT(feature_extractor=feature_extractor_weights, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        mask=mask,
                        multiclass=mc,
                        dtype=torch.float32
                        )
        
        criterion = crit_dict[task][0]()
        
        # if hasattr(model.nona.output_layer, 'mask'):
        #     print(f'k = {(model.nona.output_layer.mask.k.data).item()}, s = {(model.nona.output_layer.mask.s.data).item()}')
        optimizer = torch.optim.Adam([
            {'params': mask.parameters(), 'lr': 1e-3}, # Regular tuning of soft mask params.
            {'params': [p for n, p in model.named_parameters() if not n.startswith("mask")], 'lr': 1e-5}  # Fine tuning
        ])

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
                if dataset=='cifar':
                    batch_y = batch_y.to(torch.long) 
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
                        if dataset == 'cifar':
                            y_val = y_val.to(torch.long)  
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
            # if hasattr(model.nona.output_layer, 'mask'):
            #     print(f'k = {(model.nona.output_layer.mask.k.data).item()}, s = {(model.nona.output_layer.mask.s.data).item()}')
            epoch += 1
        
        print("Evaluating", predictor_head) 
        y_hats = []
        y_tests = []
        model.load_state_dict(best_model_state)
        with torch.no_grad():
            for (X_train, y_train), (X_test, y_test) in tqdm(zip(all_train_loader, test_loader), desc="Test", file=sys.stdout):
                y_hat_batch, z_test, z_train = model(X_test, X_train, y_train, get_embeddings=True)
                y_hats.append(y_hat_batch)
                y_tests.append(y_test)

        y_hat = torch.cat(y_hats, dim=0)
        y_test = torch.cat(y_tests, dim=0)
        if dataset == 'cifar':
            y_train = y_train.to(torch.long)
            y_test = y_test.to(torch.long)
        end = time.time()
        
        scores[f'{predictor_head} mlp'] =  [score(y_hat, y_test), end-start]

        print(f"Training and evaluating tuned knn with {predictor_head} final embeddings.") 
        start = time.time()
        y_hat_knn = tune_knn(z_train, z_test, y_train, y_test, task, score)
        end = time.time()
        scores[f'{predictor_head} tuned knn'] = [score(y_hat_knn, y_test), end-start]
        
        if save_models:
            model_path = f'results/{dataset}/models/{script_start_time}/{predictor_head}_{seed}.pth'
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            torch.save(model.state_dict(), model_path)

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset and other configs.")
    parser.add_argument('--dataset', type=str, default='cifar', help='Path to data directory.')
    parser.add_argument('--seeds', type=int, default=10, help='How many splits of the data to train and test on.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether or not to save final models')
    args = parser.parse_args()
    dataset = args.dataset
    save_models = args.savemodels
    seeds = args.seeds

    script_start_time = time.strftime("%m%d%H%M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task, fe = load_data_params(dataset)
    
    crit_dict = {'binary': [nn.BCELoss, 'auc'],
                 'multiclass': [nn.CrossEntropyLoss, 'accuracy'],
                 'ordinal': [nn.MSELoss, 'accuracy'],
                 'regression': [nn.MSELoss, 'mse']}
    score = Score(crit_dict[task][1])

    results_path = f'results/{dataset}/scores_{script_start_time}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    scores_list = ["200, 50 SoftPointWiseKNN. bn + var lr (1e-3). s/(1-s) scaling."]

    for seed in range(seeds):
        print(f'Training and evaluating models for split {seed+1}.')
        
        folds_dict = get_folds(dataset=dataset, seed=seed)

        scores = mlps_train_eval(**folds_dict, feature_extractor=fe)

        for k,v in scores.items():
            if score.metric == 'accuracy':
                test_score = f'{round(100*v[0],3)}%'
            else:
                test_score = f'{-round(v[0],3)}'
            print(f'{k}: {test_score} {score.metric} in {round(v[1],3)}s.')

        scores_list.append(scores)
        
        with open(results_path, "wb") as file:
            pkl.dump(obj=scores_list, file=file)