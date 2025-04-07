import numpy as np
import pandas as pd
import torch
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torch.nn.functional import one_hot, sigmoid
from torch.utils.data import DataLoader
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel
from torcheval.metrics.functional import mean_squared_error
from torcheval.metrics.aggregation.auc import AUC
import torch.optim as optim
import time
from copy import deepcopy
import argparse
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pickle as pkl
from models import NONA_FT
from similarity_masks import SoftKNNMask, HardKNNMask, SoftSimMask, HardSimMask, SoftPointwiseKNN
import time
from tqdm import tqdm
import sys
from utils import Score, load_data_params, get_folds, tune_knn, sliced

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

class DrugReviewDataset:
    def __init__(self, ids, tokenizer):
        self.ids = ids

        full_df = pd.read_csv('data/drug_reviews/all_data.csv')

        df = full_df.iloc[ids]

        self.df = df
        self.dataset = Dataset.from_pandas(self.df)

        self.tokenizer = tokenizer

        extra_cols = ['review', 'rating', '__index_level_0__']
        self.dataset = self.dataset.map(self.process_example, remove_columns=extra_cols)

    def len(self):
        return len(self.df)

    def scale_label(self, label):
        # Min label = 1, max = 10
        return label / 10


    def process_example(self, example):
        tokenized = self.tokenizer(example['review'], padding='max_length', truncation=True)

        label = example['rating']
        label = self.scale_label(label)

        tokenized['labels'] = label
        return tokenized

    def get_dataset(self):
        return self.dataset

def collate_fn(batch):
    input_ids = torch.tensor([item["input_ids"] for item in batch])
    attention_mask = torch.tensor([item["attention_mask"] for item in batch])
    labels = torch.tensor([item["labels"] for item in batch], dtype=torch.float)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

def mlps_train_eval(train, val, test, feature_extractor):
    scores = {}
    
    # It is necessary to shuffle training data so samples can use new
    # neighbors between batches.
    if dataset == 'adresso':
        train_dataset = AdressoDataset(label=label, tokenizer=tokenizer, ids=train)
        train_loader = DataLoader(train_dataset.get_dataset(), batch_size=16, shuffle=True, collate_fn=collate_fn)

        val_dataset = AdressoDataset(label=label, tokenizer=tokenizer, ids=val, scaler=train_dataset.scaler)
        val_loader = DataLoader(val_dataset.get_dataset(), batch_size=val_dataset.len(), collate_fn=collate_fn)

        test_dataset = AdressoDataset(label=label, tokenizer=tokenizer, scaler=train_dataset.scaler)
        test_loader = DataLoader(test_dataset.get_dataset(), batch_size=test_dataset.len(), collate_fn=collate_fn)
    
    elif dataset == 'drugs':
        train_dataset = DrugReviewDataset(ids=train, tokenizer=tokenizer)
        train_loader = DataLoader(train_dataset.get_dataset(), batch_size=128, shuffle=True, collate_fn=collate_fn)
        
        val_dataset = DrugReviewDataset(ids=val, tokenizer=tokenizer)
        val_loader = DataLoader(val_dataset.get_dataset(), batch_size=128, collate_fn=collate_fn)

        test_dataset = DrugReviewDataset(ids=test, tokenizer=tokenizer)
        test_loader = DataLoader(test_dataset.get_dataset(), batch_size=128, collate_fn=collate_fn)

    for predictor_head in ['nona l2', 'nona l1', 'nona dot', 'dense']:
        
        print("Training", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        feature_extractor_weights = feature_extractor("distilbert-base-uncased")

        hls = [200, 50]
        mask = SoftPointwiseKNN()
        model = NONA_FT(feature_extractor=feature_extractor_weights, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        mask=mask, 
                        dtype=torch.float32
                        )
        
        criterion = crit_dict[task][0]()

        # optimizer = optim.Adam([
        #     {'params': mask.parameters(), 'lr': 1e-2}, # Regular tuning of soft mask params.
        #     {'params': [p for n, p in model.named_parameters() if not n.startswith("mask")], 'lr': 1e-5}  # Fine tuning
        # ])
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        
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
                X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                y = batch['labels'].to(device)
                outputs = model(X, X, y)

                if predictor=='dense' and label=='dx':
                    outputs = sigmoid(outputs).squeeze()

                loss = criterion(outputs, y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            report = f"Train Loss: {train_loss: .5f}"

            # Early stopping
            if epoch > start_after_epoch:
                model.eval()
                y_val = []
                y_hat_val = []
                with torch.no_grad():
                    if predictor_head == 'dense':
                        for batch in val_loader:
                            X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                            y = batch['labels'].to(device)
                            y_val.append(y)
                            
                            y_hat = model(X, X, y)
                            y_hat_val.append(y_hat)
                    else:
                        z_train = []
                        y_train = []
                        
                        for batch in train_loader: # Get embeddings for full train set
                            X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                            y = batch['labels'].to(device)                            
                            y_train.append(y)

                            _, z, _ = model(X, sliced(X), sliced(y), get_embeddings=True)
                            z_train.append(z)
                        
                        z_train = torch.cat(z_train, dim=0)
                        y_train = torch.cat(y_train, dim=0)

                        for batch in val_loader:
                            X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                            y = batch['labels'].to(device)
                            y_val.append(y)
                           
                            _, z, _ = model(X, sliced(X), sliced(y), get_embeddings=True)
                            y_hat = torch.clip(model.nona.output_layer(z, z_train, y_train), 0, 1)
                            y_hat_val.append(y_hat)
                        
                y_val = torch.cat(y_val, dim=0)
                y_hat_val = torch.cat(y_hat_val, dim=0)
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
        
        print("Evaluating", predictor_head) 
        z_test = []
        y_test = []
        y_hat_test = []
        model.load_state_dict(best_model_state)
        with torch.no_grad():
            if predictor=='dense':
                
                z_train = []
                y_train = []

                for batch in train_loader:
                    X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                    y = batch['labels'].to(device)              
                    y_train.append(y)

                    _, z, _ = model(X, sliced(X), sliced(y), get_embeddings=True)
                    z_train.append(z)

                z_train = torch.cat(z_train, dim=0)
                y_train = torch.cat(y_train, dim=0)

            for i, batch in enumerate(test_loader):
                X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
                y = batch['labels'].to(device)              
                y_test.append(y)

                y_hat, z, _ = model(X, sliced(X), sliced(y), get_embeddings=True)
                y_hat_test.append(y_hat)
                z_test.append(z)

                if predictor == 'nona':
                    y_hat = torch.clip(model.nona.output_layer(z, z_train, y_train), 0, 1)
                    y_hat_test[i] = y_hat
            
        z_test = torch.cat(z_test, dim=0)
        y_test = torch.cat(y_test, dim=0)
        y_hat_test = torch.cat(y_hat_test, dim=0)
        end = time.time()
        scores[f'{predictor_head} mlp'] = [score(y_hat_test, y_test), end-start]
        
        print(f"Training and evaluating tuned knn with {predictor_head} final embeddings.") 
        start = time.time()
        y_hat_knn = tune_knn(z_train, z_test, y_train, y_test, task, score)
        end = time.time()
        scores[f'{predictor_head} tuned knn'] = [score(y_hat_knn, y_test), end-start]
        
        print(f'Test scores: {scores}')

        if save_models:
            model_path = f'results/{dataset}/{label}/models/{script_start_time}/{predictor_head}_{seed}.pth'
            model_dir = os.path.dirname(model_path)
            if not os.path.exists(model_dir):
                os.makedirs(model_dir)
            model_objs = {'model weights': model.state_dict(),
            'z_train': z_train, 'z_test': z_test, 
            'y_train': y_train, 'y_test': y_test}
            torch.save(model_objs, model_path)

    return scores

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset and other configs.")
    parser.add_argument('--dataset', type=str, default='adresso', help='Path to data directory.')
    parser.add_argument('--label', type=str, default='mmse', help='Which label to predict')
    parser.add_argument('--seeds', type=int, default=10, help='How many splits of the data to train and test on.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether or not to save final models')
    args = parser.parse_args()
    dataset = args.dataset
    label = args.label
    save_models = args.savemodels
    seeds = args.seeds

    if dataset == 'drugs':
        label = None

    script_start_time = time.strftime("%m%d%H%M")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task, fe, tokenizer = load_data_params(dataset, label)
    
    crit_dict = {'binary': [BCELoss, 'f1'],
                 'multiclass': [CrossEntropyLoss, 'accuracy'],
                 'ordinal': [MSELoss, 'accuracy'],
                 'regression': [MSELoss, 'mse']}
    score = Score(crit_dict[task][1])

    results_path = f'results/{dataset}/scores_{script_start_time}.pkl'
    if label is not None:
        results_path = f'results/{dataset}/{label}/scores_{script_start_time}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    scores_list = ["SoftPointWiseKNN"]

    for seed in range(seeds):
        print(f'Training and evaluating models for split {seed+1}.')
        
        idx_dict = get_folds(dataset=dataset, seed=seed, label=label)

        scores = mlps_train_eval(**idx_dict, feature_extractor=fe)

        for k,v in scores.items():
                test_score = abs(round(v[0],3))
                print(f'{k}: {test_score} {score.metric} in {round(v[1],3)}s.')

        scores_list.append(scores)
        
        with open(results_path, "wb") as file:
            pkl.dump(obj=scores_list, file=file)