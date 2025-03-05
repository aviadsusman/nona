import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel
from torcheval.metrics.functional import mean_squared_error
import torch.optim as optim
import torchvision
import time
from copy import deepcopy
import argparse
import os
import pickle as pkl
from models import NONA_FT
import time
from tqdm import tqdm
import sys

def load_data_params(dataset, label):
    if dataset == 'adresso':
        if label == 'mmse':
            task = 'regression'
        elif label=='dx':
            task = 'binary'
        
        data_df = pd.read_parquet('data/adresso/x_y.parquet')
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        fe = AutoModel.from_pretrained
        
    return task, data_df, fe, tokenizer

def get_fold_indices(data_df, seed):
    id_dict = {} # data dict
    
    ids = data_df['id'].values

    if dataset == 'adresso':
        if label == 'mmse':
            splitting_labels = data_df['mmse binned'].values
        elif label == 'dx':
            splitting_labels = data_df['dx'].values

        id_dict['train'], id_dict['val'] = train_test_split(ids, test_size=0.15, stratify=splitting_labels, random_state=seed)

    return id_dict

from datasets import Dataset
import pandas as pd

class AdressoDataset:
    def __init__(self, label, tokenizer, scaler=None, ids=None):
        self.ids = ids
        self.label = label
        self.scaler = scaler

        full_df = pd.read_parquet('data/adresso/x_y.parquet')

        # test and rest are labeled differently
        id_char = 'd' if self.ids is None else 'o'
        df = full_df[full_df['id'].str[4] == id_char]

        if self.ids is not None: # not test
            df = df[df['id'].isin(self.ids)]

        self.df = df
        self.dataset = Dataset.from_pandas(self.df)

        self.tokenizer = tokenizer

        if self.label == 'mmse' and self.scaler is None:
            labels = self.df['mmse']
            self.scaler = [labels.min(), labels.max()]

        extra_cols = ["id", "dx", "mmse", "mmse binned", "path", "text", "__index_level_0__"]
        self.dataset = self.dataset.map(self.process_example, remove_columns=extra_cols)

    def len(self):
        return len(self.df)

    def scale_label(self, label):
            min_label, max_label = self.scaler
            return (label - min_label) / (max_label - min_label)

    def process_example(self, example):
        tokenized = self.tokenizer(example["text"], padding="max_length", truncation=True)

        label = example[self.label]
        if self.label == "mmse":
            label = self.scale_label(label)

        tokenized["labels"] = label
        return tokenized

    def get_dataset(self):
        return self.dataset

class Score(nn.Module):
    def __init__(self, metric):
        super(Score, self).__init__()
        self.metric = metric

    def forward(self, y_hat, y):
        if self.metric == 'accuracy':
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1: #multiclass
                y_hat = torch.argmax(y_hat, dim=1)
            else:
                y_hat = torch.round(y_hat)
            
            return (y_hat == y).float().mean().item()
        
        elif self.metric == 'mse':
            return - mean_squared_error(y_hat, y).item() # Negative mse to simplify early stopping code

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def mlps_train_eval(train, val, feature_extractor):
    scores = {}
    
    learning_rate = 1e-5
    
    train_dataset = AdressoDataset(label=label, tokenizer=tokenizer, ids=train)
    train_loader = train_dataloader = DataLoader(train_dataset.get_dataset(), batch_size=16, shuffle=True, collate_fn=collate_fn)
    all_train_loader = DataLoader(train_dataset.get_dataset(), batch_size=train_dataset.len(), shuffle=True, collate_fn=collate_fn) # for use as neighbors with val and test

    val_dataset = AdressoDataset(label=label, tokenizer=tokenizer, ids=val, scaler=train_dataset.scaler)
    val_loader = DataLoader(val_dataset.get_dataset(), batch_size=val_dataset.len(), shuffle=True, collate_fn=collate_fn)

    test_dataset = AdressoDataset(label=label, tokenizer=tokenizer, scaler=train_dataset.scaler)
    test_loader = DataLoader(test_dataset.get_dataset(), batch_size=test_dataset.len(), shuffle=True, collate_fn=collate_fn)

    for predictor_head in ['nona euclidean', 'nona dot', 'dense']:
        
        print("Training", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        if dataset == 'adresso': # reinitialize weights
            feature_extractor_weights = feature_extractor("distilbert-base-uncased")

        hls = [200, 50]
        model = NONA_FT(feature_extractor=feature_extractor_weights, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        task=task, 
                        dtype=torch.float32,
                        k=5)
        
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
            for batch in tqdm(train_loader, desc="Train", file=sys.stdout):
                batch_X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                batch_y = batch['labels'].to(device)
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
                    for train_batch, val_batch in tqdm(zip(all_train_loader, val_loader), desc="Val", file=sys.stdout):
                        X_train = {key: val.to(device) for key, val in train_batch.items() if key!='labels'}
                        y_train = train_batch['labels'].to(device)
                        
                        X_val = {key: val.to(device) for key, val in val_batch.items() if key!='labels'}
                        y_val = val_batch['labels'].to(device)
                        
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
            for train_batch, test_batch in tqdm(zip(all_train_loader, test_loader), desc="Test", file=sys.stdout):
                X_train = {key: val.to(device) for key, val in train_batch.items() if key!='labels'}
                y_train = train_batch['labels'].to(device)
                
                X_test = {key: val.to(device) for key, val in test_batch.items() if key!='labels'}
                y_test = test_batch['labels'].to(device)
                
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
    parser.add_argument('--label', type=str, default='mmse', help='Which label to predict')
    parser.add_argument('--seeds', type=int, default=10, help='How many splits of the data to train and test on.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether or not to save final models')
    args = parser.parse_args()
    dataset = args.dataset
    label = args.label
    save_models = args.savemodels
    seeds = args.seeds

    script_start_time = time.strftime("%m%d%H%M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task, data_df, fe, tokenizer = load_data_params(dataset, label)
    
    crit_dict = {'binary': [nn.BCELoss, 'accuracy'],
                 'multiclass': [nn.CrossEntropyLoss, 'accuracy'],
                 'ordinal': [nn.MSELoss, 'accuracy'],
                 'regression': [nn.MSELoss, 'mse']}
    score = Score(crit_dict[task][1])

    results_path = f'results/{dataset}/scores_{script_start_time}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    scores_list = ["200,50 dx label. mse loss"]

    for seed in range(seeds):
        print(f'Training and evaluating models for split {seed+1}.')
        
        idx_dict = get_fold_indices(data_df, seed=seed)

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