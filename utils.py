import torch
from torcheval.metrics.functional import binary_f1_score
from torcheval.metrics.functional import mean_squared_error as mse
from torcheval.metrics.aggregation.auc import AUC
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error, f1_score, auc
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18, resnet34, resnet50
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np

class Score(torch.nn.Module):
    def __init__(self, metric):
        super(Score, self).__init__()
        self.metric = metric

        if self.metric == 'auc':
            self.auc = AUC()
        
        if self.metric in ['accuracy', 'auc', 'f1']:
            self.higher_is_better = True
        else:
            self.higher_is_better = False

    def forward(self, y_hat, y):
        if self.metric == 'accuracy':
            if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
                y_hat = torch.argmax(y_hat, dim=1)
            else:
                y_hat = torch.round(y_hat)
            
            output = (y_hat == y).float().mean().item()
        
        elif self.metric == 'auc':
            output = self.auc.update(y_hat, y).compute().item()
        
        elif self.metric == 'f1':
            output = binary_f1_score(y_hat, y).item()

        elif self.metric == 'mse':
            output = mse(y_hat, y).item() # Negative mse to simplify early stopping code

        return output if self.higher_is_better else - output

def load_data_params(dataset, label=None):
    if dataset == 'adresso':
        if label == 'mmse':
            task = 'regression'
        elif label=='dx':
            task = 'binary'
        
        data_df = pd.read_parquet('data/adresso/x_y.parquet')
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        fe = AutoModel.from_pretrained
        
        return task, fe, tokenizer
    
    if dataset == 'drugs':
        task = 'regression'
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased") # "huawei-noah/TinyBERT_General_4L_312D"
        fe = AutoModel.from_pretrained

        return task, fe, tokenizer
    
    elif dataset == 'rsna':
        task = 'regression'
        fe = resnet18

        return task, fe
    
    elif dataset == 'cifar':
        task = 'multiclass'
        fe = resnet18

        return task, fe

def get_folds(dataset, seed, label=None):
    if dataset == 'rsna':
        data_df = pd.read_csv('data/rsna/all_features.csv')
        fold_dict = {}
        ids = data_df['id'].values
        binned_labels = data_df['boneage binned'].values

        # _, ids, _, binned_labels = train_test_split(ids, binned_labels, test_size=0.125, stratify=binned_labels, random_state=seed)

        tv_ids, fold_dict['test'], tv_labels, _ = train_test_split(ids, binned_labels, test_size=0.2, stratify=binned_labels, random_state=seed)
        
        fold_dict['train'], fold_dict['val'], _, _ = train_test_split(tv_ids, tv_labels, stratify=tv_labels, test_size=0.15, random_state=seed)

        return fold_dict

    elif dataset == 'adresso':
        data_df = pd.read_parquet('data/adresso/x_y.parquet')
        fold_dict = {}
        ids = data_df['id'].values
        if label == 'mmse':
            splitting_labels = data_df['mmse binned'].values
        elif label == 'dx':
            splitting_labels = data_df['dx'].values

        fold_dict['train'], fold_dict['val'] = train_test_split(ids, test_size=0.15, stratify=splitting_labels, random_state=seed)

        fold_dict['test'] = None

        return fold_dict

    elif dataset == 'cifar':
        torch.manual_seed(seed)
        np.random.seed(seed)

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        tv_set = torchvision.datasets.CIFAR10(root='data/cifar',
                                                  download=True,
                                                  train=True,
                                                  transform=transform)
        test_set = torchvision.datasets.CIFAR10(root='data/cifar',
                                                    download=True,
                                                    train=False,
                                                    transform=transform)

        val_percentage = 0.15
        train_size = int((1-val_percentage) * len(tv_set))
        val_size = len(tv_set) - train_size 

        train_set, val_set = random_split(tv_set, [train_size, val_size])

        return {'train': train_set, 'val': val_set, 'test': test_set}
    
    elif dataset == 'drugs':
        data_df = pd.read_csv('data/drug_reviews/all_data.csv')
        fold_dict = {}
        
        ids = data_df.index
        labels = data_df['rating']
        
        # _, ids, _, labels = train_test_split(ids, labels, test_size=0.01, stratify=labels, random_state=seed)

        tv_ids, fold_dict['test'], tv_labels, _ = train_test_split(ids, labels, test_size=0.2, stratify=labels, random_state=seed)

        fold_dict['train'], fold_dict['val'], _, _ = train_test_split(tv_ids, tv_labels, test_size=0.15, stratify=tv_labels, random_state=seed)
        
        return fold_dict


def tune_knn(X_train, X_test, y_train, y_test, task, score):

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train.cpu().detach())

    X_test = scaler.transform(X_test.cpu().detach())

    sklearn_score_dict = {'accuracy': accuracy_score, 'mse': mean_squared_error, 'f1': f1_score, 'auc': auc}
    scorer = make_scorer(sklearn_score_dict[score.metric], greater_is_better=score.higher_is_better) 
    
    if task in ['binary', 'multiclass']:
        knn = KNeighborsClassifier() 
    else:    
        knn = KNeighborsRegressor()

    param_grid = {
        'n_neighbors': [3, 5, 7, 9, 11],
        'weights': ['uniform', 'distance'],
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

    grid_search.fit(X_train, y_train.cpu().detach().squeeze())

    best_knn = grid_search.best_estimator_

    y_hat_knn = best_knn.predict(X_test)

    return torch.tensor(y_hat_knn, dtype=y_test.dtype).to(y_test.device)

def sliced(data):
    if isinstance(data, torch.Tensor):
        return data[:2]
    elif isinstance(data, dict):
        return {k:v[:2] for k,v in data.items()}