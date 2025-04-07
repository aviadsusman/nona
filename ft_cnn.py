import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader, Subset
from torch.optim import Adam
import torchvision.transforms as transforms
from PIL import Image
from models import NONA_FT
import similarity_masks as s
from utils import Score, load_data_params, get_folds, tune_knn
import argparse
import time
from copy import deepcopy
import pickle as pkl
from tqdm import tqdm
import sys
import os

class RSNADataset(Dataset):
    def __init__(self, indices, scaler=None):
        super(RSNADataset, self).__init__()
        self.indices = indices
        self.transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        # self.scaler = scaler
        self.scaler = [1, 228]

        self.features = pd.read_csv('data/rsna/all_features.csv')
        self.features = self.features[self.features['id'].isin(self.indices)].reset_index()
    
    def __len__(self):
        return len(self.indices)

    def _scale_label(self, label):
        if self.scaler == None:
            labels = self.features[self.features['id'].isin(self.indices)]['boneage']
            min_label, max_label = labels.min(), labels.max()
            self.scaler = [min_label, max_label]
        else:
            min_label, max_label = self.scaler[0], self.scaler[1]
        
        return (label - min_label) / (max_label - min_label)
    
    def __getitem__(self, idx):
        img_path = self.features.loc[idx, 'path']
        image = Image.open(img_path)

        if self.transform:
            image = self.transform(image)
        label = self.features.loc[idx, 'boneage']
        
        return image, self._scale_label(label)

def collate(batch):
    x, y = zip(*batch)
    x = torch.stack(x).to(device).to(torch.float32)
    y = torch.tensor(y, dtype=torch.float32, device=device)
    return x, y

def mlps_train_eval(train, val, test, feature_extractor):
    scores = {}

    # Prep data loaders
    # It is necessary to shuffle training data so samples can use new
    # neighbors between batches.
    if dataset == 'rsna':
        train_dataset = RSNADataset(train)
        train_loader = DataLoader(train_dataset, batch_size=128, shuffle=true, collate_fn=collate)
        
        val_dataset = RSNADataset(val)
        val_loader = DataLoader(val_dataset, batch_size=128, collate_fn=collate)
        
        test_dataset = RSNADataset(test)
        test_loader = DataLoader(test_dataset, batch_size=128, collate_fn=collate)
    
    elif dataset == 'cifar':
        train_loader = DataLoader(train, batch_size=128, shuffle=True, collate_fn=collate)
        val_loader = DataLoader(val, batch_size=128, collate_fn=collate)
        test_loader = DataLoader(test, batch_size=128, collate_fn=collate)

    for predictor_head in ['nona l2', 'nona l1']:#, 'nona dot', 'dense']:
        
        print("Training", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        # Build model
        feature_extractor_weights = feature_extractor(weights='DEFAULT')
        mc = 10 if dataset=='cifar' else None
        # agg = 'mean' if dataset == 'cifar' else None
        agg = None
        hls = [200, 50]
        mask = s.LnSmoothStep()
        model = NONA_FT(feature_extractor=feature_extractor_weights, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        mask=mask,
                        multiclass=mc,
                        agg=agg,
                        dtype=torch.float32
                        )

        criterion = crit_dict[task][0]()

        # Variable learning rates for finetuning feature extractor and training similarity shift params
        optimizer = Adam(model.parameters(), lr=1e-5)
        # optimizer = Adam([
        #     {'params': mask.parameters(), 'lr': 1e-3}, # Regular tuning of soft mask params.
        #     {'params': [p for n, p in model.named_parameters() if not n.startswith("mask")], 'lr': 1e-5}  # Fine tuning
        # ])

        start = time.time()
        
        # Early stopping params
        patience = 10
        start_after_epoch = 5
        count = 0
        best_val_score = float('-inf')

        # Train
        epoch = 1
        while count < patience: 
            model.train()
            train_loss = 0.0
            print('Epoch:', epoch)
            for X, y in tqdm(train_loader, desc="Train", file=sys.stdout):
                outputs = model(X, X, y)

                if dataset=='cifar':
                    y = y.to(torch.long) 
                loss = criterion(outputs, y)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            report = f"Train Loss: {train_loss: .5f}"

            # Early stopping on val set
            if epoch > start_after_epoch:
                model.eval()
               
                y_val = []
                y_hat_val = []

                with torch.no_grad():
                    if predictor_head == 'dense':
                        for X, y in val_loader:
                            y_val.append(y)
                            y_hat = model(X, X, y)
                            y_hat_val.append(y_hat)
                    else:
                        z_train = []
                        y_train = []
                        for (X, y) in train_loader: # Get embeddings for full train set
                            y_train.append(y)
                        
                            _, z, _ = model(X, X[:2], y[:2], get_embeddings=True)
                            z_train.append(z)
                        
                        z_train = torch.cat(z_train, dim=0)
                        y_train = torch.cat(y_train, dim=0)
                        if dataset == 'cifar':            
                            y_train = one_hot(y_train.long()).to(model.device, model.dtype)  

                        for (X, y) in val_loader:
                            y_val.append(y)
                           
                            _, z, _ = model(X, X[:2], y[:2], get_embeddings=True)
                            y_hat = torch.clip(model.nona.output_layer(z, z_train, y_train), 0, 1)
                            y_hat_val.append(y_hat)
                        
                y_val = torch.cat(y_val, dim=0)
                y_hat_val = torch.cat(y_hat_val, dim=0)
                
                if dataset == 'cifar':
                    y_val = y_val.to(torch.long)
                
                val_score = score(y_hat_val, y_val)
                if val_score > best_val_score:
                    best_val_score = val_score
                    best_model_state = deepcopy(model.state_dict())
                    count = 0
                else:
                    count += 1
                report = report + f': Val Score: {abs(val_score): .5f}'
            
            print(report)
            epoch += 1

        # Eval on test set and extract embeddings for KNN
        print("Evaluating", predictor_head) 
        model.load_state_dict(best_model_state)
        z_test = []
        y_test = []
        y_hat_test = []
        with torch.no_grad():
            if predictor=='dense':
                z_train = []
                y_train = []
                for (X, y) in train_loader:
                    y_train.append(y)
                    _, z, _ = model(X, X[:2], y[:2], get_embeddings=True)
                    z_train.append(z)
                z_train = torch.cat(z_train, dim=0)
                y_train = torch.cat(y_train, dim=0)

            for i, (X, y) in enumerate(test_loader):
                y_test.append(y)
                y_hat, z, _ = model(X, X[:2], y[:2], get_embeddings=True)
                y_hat_test.append(y_hat)
                z_test.append(z)

                if predictor == 'nona':
                    y_hat = torch.clip(model.nona.output_layer(z, z_train, y_train), 0, 1)
                    y_hat_test[i] = y_hat
            
        z_test = torch.cat(z_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        y_hat_test = torch.cat(y_hat_test, dim=0)
        end = time.time()
        scores[f'{predictor_head} mlp'] =  [score(y_hat_test, y_test), end-start]

        # Save models
        if save_models:
            model_path = f'results/{dataset}/models/{script_start_time}/{predictor_head}_{seed}.pth'
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_objs = {'model weights': model.state_dict(),
            'z_train': z_train, 'z_test': z_test, 
            'y_train': y_train, 'y_test': y_test}
            torch.save(model_objs, model_path)

        # Fine tune a KNN model on final embeddings
        if dataset == 'cifar':
            y_train = y_train.to(torch.long)
            y_test = y_test.to(torch.long)
        print(f"Training and evaluating tuned knn with {predictor_head} final embeddings.") 
        start = time.time()
        y_hat_knn = tune_knn(z_train, z_test, y_train, y_test, task, score)
        end = time.time()
        scores[f'{predictor_head} tuned knn'] = [score(y_hat_knn, y_test), end-start]

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
    
    crit_dict = {'binary': [BCELoss, 'auc'],
                 'multiclass': [CrossEntropyLoss, 'accuracy'],
                 'ordinal': [MSELoss, 'accuracy'],
                 'regression': [MSELoss, 'mse']}
    score = Score(crit_dict[task][1])

    results_path = f'results/{dataset}/scores_{script_start_time}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    scores_list = ["LnSmoothStep"]

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