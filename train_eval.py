import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, mean_squared_error, make_scorer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.nn.functional import one_hot
from torcheval.metrics.functional import mean_squared_error
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from xgboost import XGBClassifier
import time
from copy import deepcopy
import argparse
import os
import pickle as pkl
from models import NONA, NONA_NN
import time
import sys


def tensor(arr):
    if type(arr) != torch.Tensor:
        arr = torch.Tensor(arr)
    return arr.to(dtype=torch.float64, device=device)

def load_data_params(dataset):
    if dataset == 'bc':
        task = 'binary'
        x_y = datasets.load_breast_cancer()
        X = tensor(x_y.data)
        y = x_y.target
        encoder = OrdinalEncoder()
        y = tensor(encoder.fit_transform(y.reshape(-1, 1)))

    elif dataset == 'cifar':
        task = 'multiclass'
        X = torch.load('data/cifar/feature_extracted/resnet50/X.pt')
        y = torch.load('data/cifar/feature_extracted/y.pt')
    
    elif dataset == 'sat':
        task = 'ordinal'
        X = torch.load('data/sat/X.pt')
        y = torch.load('data/sat/y.pt')
        
    classes = len(torch.unique(y))
    if task == 'multiclass':
        agg = 'mean'
    else:
        agg = None

    return task, X, y, classes, agg

def get_folds(X, y, seed):
    dd = {} # data dict

    dd['X_tv'], dd['X_test'], dd['y_tv'], dd['y_test'] = train_test_split(X, y, test_size=0.25, stratify=y, random_state=seed)
    dd['X_train'], dd['X_val'], dd['y_train'], dd['y_val'] = train_test_split(dd['X_tv'], dd['y_tv'], test_size=0.15, stratify=dd['y_tv'], random_state=seed)

    dd_tensor = {k: tensor(v) for k,v in dd.items()}

    return dd_tensor


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
        

def mlps_train_eval(X_tv, X_train, X_val, X_test, y_tv, y_train, y_val, y_test):
    scores = {}
    
    learning_rate = 0.001
    if dataset in ['bc','sat']:
        batch_size = 32
    elif dataset == 'cifar':
        batch_size = 128

    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    for predictor_head in ['nona euclidean', 'nona dot', 'dense']: # 'nona cos'
        
        print("Training and evaluating", predictor_head) 
        
        predictor = predictor_head.split(" ")[0]
        similarity = predictor_head.split(" ")[-1]

        # Classify on raw data/representations
        if predictor == 'nona':
            base_model = NONA(similarity=similarity, batch_norm=X_tv.shape[1], agg=agg)
            
            start = time.time()
            if dataset == 'cifar':
                y_tv_ohe = tensor(one_hot(y_tv.long()))
                y_hat_base = base_model(X_test, X_tv, y_tv_ohe)
            else:
                y_hat_base = base_model(X_test, X_tv, y_tv)
            end = time.time()

            scores[predictor_head] = [score(y_hat_base, y_test), end-start]

        feats = X_train.shape[1]
        hls = [feats // 4, feats // 4, feats // 4]
        model = NONA_NN(input_size=feats, 
                        hl_sizes=hls, 
                        predictor=predictor, 
                        similarity=similarity, 
                        task=task, 
                        classes=classes, 
                        agg=agg)
        
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

            for batch_X, batch_y in train_loader:
                outputs = model(batch_X, batch_X, batch_y)
                loss = criterion(outputs, batch_y.long()) if dataset=='cifar' else criterion(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            report = f"Epoch {epoch}, Train Loss: {train_loss:.4f}"

            # Early stopping
            if epoch > start_after_epoch:
                model.eval()
                with torch.no_grad():
                    y_hat_val = model(X_val, X_train, y_train)
                    val_score = score(y_hat_val, y_val)
                    if val_score > best_val_score:
                        best_val_score = val_score
                        best_model_state = deepcopy(model.state_dict())
                        count = 0
                    else:
                        count += 1
                report = report + f': Val Score: {round(abs(val_score), 4)}'
            
            print(report)
            epoch += 1

        model.load_state_dict(best_model_state)
        y_hat = model(X_test, X_train, y_train)
        end = time.time()
        

        scores[f'{predictor_head} mlp'] =  [score(y_hat, y_test), end-start]

    return scores


def tune_xgb(X_train, X_test, y_train, y_test):
    scorer = make_scorer(accuracy_score, greater_is_better=True) 

    xgb = XGBClassifier(eval_metric='mlogloss')

    param_grid = {'n_estimators': [50, 100, 200],
                  'max_depth': [3, 5, 7],
                  'learning_rate': [0.01, 0.1, 0.2],
                  'subsample': [0.8, 1.0],
                  'colsample_bytree': [0.8, 1.0],
                  'gamma': [0, 0.1, 1]}

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scorer,
        cv=4,
        verbose=1,
        n_jobs=-1)

    start = time.time()
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_

    y_hat_xgb = best_xgb.predict(X_test)
    end = time.time()

    return [accuracy_score(y_hat_xgb, y_test), end-start]

def tune_knn(X_train, X_test, y_train, y_test):

    train_scaler = StandardScaler()
    X_train = train_scaler.fit_transform(X_train.cpu().detach())

    test_scaler = StandardScaler()
    X_test = test_scaler.fit_transform(X_test.cpu().detach())
    
    scorer = make_scorer(accuracy_score, greater_is_better=True) 
    
    knn = KNeighborsClassifier()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski'],
        'p': [1, 2]
    }

    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        scoring=scorer,
        cv=4,
        verbose=1,
        n_jobs=-1
    )

    start = time.time()
    grid_search.fit(X_train, y_train.cpu().detach().squeeze())

    best_knn = grid_search.best_estimator_

    y_hat_knn = best_knn.predict(X_test)
    end = time.time()

    return [accuracy_score(y_hat_knn, y_test), end-start]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset and other configs.")
    parser.add_argument('--dataset', type=str, default='Which dataset to load in.', help='Path to data directory.')
    
    args = parser.parse_args()
    dataset = args.dataset

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    task, X, y, classes, agg = load_data_params(dataset)
    crit_dict = {'binary': [nn.BCELoss, 'accuracy'],
                 'multiclass': [nn.CrossEntropyLoss, 'accuracy'],
                 'ordinal': [nn.MSELoss, 'accuracy']}
    
    score = Score(crit_dict[task][1])

    scores_list = []
    for seed in range(100):
        print(f'Training models for split {seed}.')
        data_dict = get_folds(X=X, y=y, seed=seed)

        scores = mlps_train_eval(**data_dict)
        # scores['tuned xgb'] = tune_xgb(data_dict['X_tv'], data_dict['X_test'], data_dict['y_tv'], data_dict['y_test'])
        # scores['tuned knn'] = tune_knn(data_dict['X_tv'].cpu(), data_dict['X_test'].cpu(), data_dict['y_tv'].cpu(), data_dict['y_test'].cpu())

        for k,v in scores.items():
            if score.metric == 'accuracy':
                test_score = f'{round(100*v[0],3)}%'
            else:
                test_score = f'{-round(v[0],3)}'
            print(f'{k}: {test_score} {score.metric} in {round(v[1],3)}s.')

        scores_list.append(scores)


    scores_list.append("9,9,9")

    results_path = f'results/{dataset}/scores_{time.strftime("%m%d%H%M")}.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(results_path, "wb") as file:
        pkl.dump(obj=scores_list, file=file)