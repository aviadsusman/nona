import torch
from torch.nn import MSELoss, BCELoss, CrossEntropyLoss
from torch.nn.functional import one_hot
from data.dataset_classes import data_loaders
from torch.optim import Adam
from models import NONA_FT, NONA
import similarity_masks as s
from utils import *
import argparse
import time
from copy import deepcopy
import pickle as pkl
from tqdm import tqdm
import sys
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Helper functions
def sliced(data):
    if isinstance(data, torch.Tensor):
        return data[:2]
    elif isinstance(data, dict):
        return {k:v[:2] for k,v in data.items()}

def x_y(batch):
    if isinstance(batch, dict):
        X = {key: val.to(device) for key, val in batch.items() if key!='labels'}
        y = batch['labels'].to(device)      
    else:
        X,y = batch

    return X, y

def z_y(loader, desc):
    z_fold = []
    y_fold = []
    for batch in tqdm(loader, desc=desc, file=sys.stdout):
        X, y = x_y(batch)
        y_fold.append(y)
        _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
        z_fold.append(z)
    return torch.cat(z_fold), torch.cat(y_fold)

# Main functions
def build_model(feature_extractor, predictor, mask):
    
    if predictor == 'dense':
        similarity = None
    else:
        similarity = predictor
        predictor = 'nona'

    # Multiclass params
    mc = 10 if dataset=='cifar' else None
    # agg = 'mean' if dataset == 'cifar' else None
    agg = None

    if dataset in ['rsna', 'cifar']:
        fe = feature_extractor(weights='DEFAULT')
    elif dataset in ['adresso', 'drugs']:
        fe = feature_extractor("distilbert-base-uncased")
    hls = [200, 50]
    if mask == 'uniform':
        nona_mask = s.UniformSoftMask()
    elif mask == 'pointwise':
        nona_mask = s.PointwiseSoftMask()
    elif mask == 'pointwise_mlp':
        nona_mask = s.PointwiseSoftMask(dims=[hls[-1], 10])
    else:
        nona_mask = None
    model = NONA_FT(feature_extractor=fe, 
                    hl_sizes=hls, 
                    predictor=predictor, 
                    similarity=similarity, 
                    mask=nona_mask,
                    multiclass=mc,
                    agg=agg,
                    dtype=torch.float32
                    )
    return model

def train_eval(train, val, test, model, optimization):
    criterion = crit_dict[task][0]()
    # Variable learning rates for finetuning feature extractor and training similarity shift params
    if optimization == 'uniform':
        optimizer = Adam(model.parameters(), lr=1e-5)
    else:
        rate = int(optimization[-1])

        optimizer = Adam([
            {'params': model.nona.output_layer.mask.parameters(), 'lr': 10 ** -rate}, # Regular tuning of soft mask params.
            {'params': [p for n, p in model.named_parameters() if not n.startswith("mask")], 'lr': 1e-5}  # Fine tuning
        ])

    nona_mask = model.nona.output_layer.mask
    if isinstance(nona_mask, s.UniformSoftMask):
        mask_params = [deepcopy(nona_mask.params.data).detach()]
        print('Mask params:', mask_params[0])

    # Early stopping params
    patience = 10
    start_after_epoch = 5
    count = 0
    best_val_score = float('-inf')
    
    # Train loop
    start = time.time()
    epoch = 1
    while count < patience: 
        model.train()
        train_loss = 0.0
        print('Epoch:', epoch)
        for batch in tqdm(train, desc="Train", file=sys.stdout):
            X,y = x_y(batch)
            outputs = model(X, X, y) 

            if dataset=='cifar':
                y = y.to(torch.long) 
            loss = criterion(outputs, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        train_loss /= len(train)
        report = f"Train Loss: {train_loss: .5f}"

        # Early stopping on val set
        if epoch > start_after_epoch:
            model.eval()
            y_val = []
            y_hat_val = []
            with torch.no_grad():
                if isinstance(model.nona.output_layer, torch.nn.Linear):
                    for batch in tqdm(val, desc='Val', file=sys.stdout):
                        X,y = x_y(batch)
                        y_val.append(y)
                        y_hat = model(X, X, y)
                        y_hat_val.append(y_hat)
                else:
                    z_train, y_train = z_y(train, 'Train for val')
                    if dataset == 'cifar':            
                        y_train = one_hot(y_train.long()).to(model.device, model.dtype)  

                    for batch in tqdm(val, desc='Val', file=sys.stdout):
                        X,y = x_y(batch)
                        y_val.append(y)
                        _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
                        y_hat = model.nona.output_layer(z, z_train, y_train)
                        y_hat_val.append(y_hat)
                    
            y_val = torch.cat(y_val)
            y_hat_val = torch.cat(y_hat_val)
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
        if isinstance(nona_mask, s.UniformSoftMask):
            mask_params.append(deepcopy(nona_mask.params.data).detach())
            print('Mask params:', mask_params[0], mask_params[-1])
        
        epoch += 1

    # Eval on test set and extract embeddings for KNN
    print("Evaluating on test") 
    model.load_state_dict(best_model_state)
    z_test = []
    y_test = []
    y_hat_test = []
    with torch.no_grad():
        z_train, y_train = z_y(train, 'Final train embeddings')
        z_val, y_val = z_y(val, 'Final val embeddings') 

        for batch in tqdm(test, desc='Test predictions', file=sys.stdout):
            X,y = x_y(batch)
            y_test.append(y)
            _, z, _ = model(X, sliced(X), sliced(y), embeddings=True)
            z_test.append(z)

            if isinstance(model.nona.output_layer, NONA):
                y_hat = model.nona.output_layer(z, z_train, y_train)
            else:
                y_hat = model.nona.output_layer(z)
            y_hat_test.append(y_hat)
        
    z_test = torch.cat(z_test)
    y_test = torch.cat(y_test)
    y_hat_test = torch.cat(y_hat_test)
    
    end = time.time()
    
    model_objs = {'z_train': z_train, 'y_train': y_train,
                  'z_val': z_val, 'y_val': y_val,
                  'z_test': z_test, 'y_test': y_test,
                  'y_hat_test': y_hat_test,
                  'score': score(y_hat_test, y_test),
                  'time': end-start}

    if save_models:
        model_objs['model weights'] = model.state_dict()
    if isinstance(nona_mask, s.UniformSoftMask):
        model_objs['mask params'] = mask_params

    # Fine tune a KNN model on final embeddings
    if dataset == 'cifar':
        y_train = y_train.to(torch.long)
        y_test = y_test.to(torch.long)
    print(f"Training and evaluating tuned knn on final embeddings.") 
    start = time.time()
    y_hat_knn = tune_knn(z_train, z_test, y_train, y_test, task, score)
    end = time.time()
    model_objs['knn score and time'] = [score(y_hat_knn, y_test), end-start]

    return model_objs

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset and configs.')
    parser.add_argument('--dataset', type=str, default='adresso', help='Which dataset.')
    parser.add_argument('--predictor', type=str, default='l2', help='prediction head')
    parser.add_argument('--mask', type=str, default='uniform', help='Which mask.')
    parser.add_argument('--optimization', type=str, default='uniform', help='Which optimization strategy.')
    parser.add_argument('--batch_size', type=str, default='128', help='batch size')
    parser.add_argument('--seed', type=int, default=0, help='Random split seed.')
    parser.add_argument('--start_time', type=str, default=None, help='Batch job start time for saving results.')
    parser.add_argument('--savemodels', action='store_true', default=False, help='Whether to save final models')
    
    args = parser.parse_args()
    dataset = args.dataset
    predictor = args.predictor
    mask = args.mask
    optimization = args.optimization
    batch_size = int(args.batch_size)
    seed = args.seed
    start_time = args.start_time
    if start_time is None:
        start_time = time.strftime("%m%d%H%M")
    save_models = args.savemodels

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    task, fe = data_params(dataset)
    
    crit_dict = {'binary': [BCELoss, 'auc'],
                 'multiclass': [CrossEntropyLoss, 'accuracy'],
                 'ordinal': [MSELoss, 'accuracy'],
                 'regression': [MSELoss, 'mse']}
    score = Score(crit_dict[task][1])
    
    print(f"{predictor} + {mask} mask + {optimization} training")

    folds_dict = folds(dataset=dataset, seed=seed)

    loaders = data_loaders(**folds_dict, 
                           dataset=dataset,
                           batch_size=batch_size)

    model = build_model(feature_extractor=fe, 
                        predictor=predictor,
                        mask=mask)

    model_objs = train_eval(**loaders, 
                            model=model, 
                            optimization=optimization)
    
    results_dir = f'results/{dataset}/{start_time}/{predictor}'
    results_file = f'{mask}_mask_{optimization}_training_bs_{batch_size}_{seed}.pth'
    results_path = os.path.join(results_dir, results_file)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    torch.save(model_objs, results_path)