import numpy as np
from sklearn import datasets
from sklearn.preprocessing import OrdinalEncoder, StandardScaler, MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, make_scorer
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.optim as optim
from xgboost import XGBClassifier
import time
from copy import deepcopy
import argparse
import os
import pickle as pkl
from nona import NONA, NONA_NN

def get_data(data, seed):
    if data == 'bc':
        x_y = datasets.load_breast_cancer()

    encoder = OrdinalEncoder()
    y = encoder.fit_transform(x_y.target.reshape(-1, 1))

    X_train, X_test, y_train, y_test = train_test_split(x_y.data, x_y.target, test_size=0.35, stratify=x_y.target, random_state=seed)

    train_scaler = StandardScaler()
    X_train = train_scaler.fit_transform(X_train)

    test_scaler = StandardScaler()
    X_test = test_scaler.fit_transform(X_test)

    return X_train, X_test, y_train, y_test

def mlps_train_eval(X_train, X_test, y_train, y_test, seed, device):
    scores = {}

    learning_rate = 0.001
    batch_size = 32

    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.15, stratify=y_train, random_state=seed)
    val_scaler = StandardScaler()
    X_val = val_scaler.fit_transform(X_val)

    X_train, y_train = torch.tensor(X_train, dtype=torch.float64, device=device), torch.tensor(y_train, dtype=torch.float64, device=device)
    X_val, y_val = torch.tensor(X_val, dtype=torch.float64, device=device), torch.tensor(y_val, dtype=torch.float64, device=device)
    X_test, y_test = torch.tensor(X_test, dtype=torch.float64, device=device), torch.tensor(y_test, dtype=torch.float64, device=device)
    
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    for classifier_head in ['nona euclidean', 'nona dot', 'dense']:
        
        classifier = classifier_head.split(" ")[0]
        similarity = classifier_head.split(" ")[-1]

        if classifier == 'nona':
            start = time.time()
            base_model = NONA(similarity=similarity)
            y_hat_base = base_model(X_test, X_train, y_train)
            end = time.time()
            scores[classifier_head] = [accuracy_score(np.round(y_hat_base.cpu().detach()), y_test.cpu().detach()), end-start]

        model = NONA_NN(input_size=X_train.shape[1], hl_sizes=[6,6], classifier=classifier, similarity=similarity)

        # class_counts = torch.bincount(y_train.to(torch.int))
        # class_weights = 1.0 / class_counts.float()
        
        criterion = nn.BCELoss() # weight=class_weights.to(device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        start = time.time()
        patience = 10
        start_from_epoch = 5
        count = 0
        best_acc_val = float('-inf')
        epoch = 0
        while count < patience:
            model.train()
            train_loss = 0.0

            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(model.device), batch_y.to(model.device)

                outputs = model(batch_X, batch_X, batch_y)  
                try:
                    loss = criterion(outputs, batch_y)
                except RuntimeError:
                    print(outputs, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                train_loss += loss.item()

            train_loss /= len(train_loader)
            report = f"Epoch {epoch + 1}, Train Loss: {train_loss:.4f}"

            # Early stopping
            if epoch >= start_from_epoch:
                model.eval()
                with torch.no_grad():
                    y_hat_val = model(X_val, X_train, y_train)
                    acc_val = accuracy_score(np.round(y_hat_val.cpu().detach()), y_val.cpu().detach())
                    if acc_val > best_acc_val:
                        best_acc_val = acc_val
                        best_model_state = deepcopy(model.state_dict())
                        count = 0
                    else:
                        count += 1

                report = report + f': Val Acc: {round(float(acc_val), 4)}'
            
            print(report)
            epoch += 1

        model.load_state_dict(best_model_state)
        y_hat = model(X_test, X_train, y_train)
        end = time.time()
        
        scores[f'{classifier_head} mlp'] =  [accuracy_score(np.round(y_hat.cpu().detach()), y_test.cpu().detach()), end-start]

    return scores


def tune_xgb(X_train, X_test, y_train, y_test):
    scorer = make_scorer(accuracy_score, greater_is_better=True) 

    xgb = XGBClassifier(eval_metric='mlogloss')

    param_grid = {'n_estimators': [50, 100, 200]}
        # 'max_depth': [3, 5, 7]
    #     'learning_rate': [0.01, 0.1, 0.2],
    #     'subsample': [0.8, 1.0],
    #     'colsample_bytree': [0.8, 1.0],
    #     'gamma': [0, 0.1, 1],
    # }

    grid_search = GridSearchCV(
        estimator=xgb,
        param_grid=param_grid,
        scoring=scorer,
        cv=5,
        verbose=1,
        n_jobs=-1)

    start = time.time()
    grid_search.fit(X_train, y_train)

    best_xgb = grid_search.best_estimator_

    y_hat_xgb = best_xgb.predict(X_test)
    end = time.time()

    return [accuracy_score(y_hat_xgb, y_test), end-start]

def tune_knn(X_train, X_test, y_train, y_test):
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
        cv=5,
        verbose=1,
        n_jobs=-1
    )

    start = time.time()
    grid_search.fit(X_train, y_train)

    best_knn = grid_search.best_estimator_

    y_hat_knn = best_knn.predict(X_test)
    end = time.time()

    return [accuracy_score(y_hat_knn, y_test), end-start]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Dataset and other configs.")
    parser.add_argument('--data', type=str, default='Which dataset to load in.', help='Path to data directory.')
    
    args = parser.parse_args()
    data = args.data
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    scores_list = []
    for seed in range(10):
        X_train, X_test, y_train, y_test = get_data(data=data, seed=seed)

        scores = mlps_train_eval(X_train, X_test, y_train, y_test, seed=seed+1, device=device)
        scores['tuned xgb'] = tune_xgb(X_train, X_test, y_train, y_test)
        scores['tuned knn'] = tune_knn(X_train, X_test, y_train, y_test)

        for k,v in scores.items():
            print(f'{k}: {100*v[0]}% accuracy in {round(v[1],3)}s.')

        scores_list.append(scores)


    results_path = f'results/{data}/scores.pkl'
    results_dir = os.path.dirname(results_path)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    with open(results_path, "wb") as file:
        pkl.dump(obj=scores_list, file=file)