import torch.nn as nn
from torcheval.metrics.functional import mean_squared_error, binary_f1_score
from torcheval.metrics.aggregation.auc import AUC
from sklearn.model_selection import train_test_split
import pandas as pd
from transformers import AutoTokenizer, AutoModel
import torchvision
import torchvision.transforms as transforms
from torchvision.models import resnet18

class Score(nn.Module):
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
            output = mean_squared_error(y_hat, y).item() # Negative mse to simplify early stopping code
        
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
        
        return task, data_df, fe, tokenizer
    
    elif dataset == 'rsna':
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

        return task, data_df, fe, data_percentage, transform

def get_fold_indices(dataset, data_df, seed, label=None, data_percentage=0.25, keep_unused=False):
    id_dict = {} # data dict
    
    ids = data_df['id'].values

    if dataset == 'rsna':
        binned_labels = data_df['boneage binned'].values

        # tvt = train/val/test
        unused_ids , tvt_ids, _ , tvt_binned_labels = train_test_split(ids, binned_labels, test_size=data_percentage, stratify=binned_labels, random_state=seed)

        if keep_unused: # for testing effect of neighbor sets at test time
            id_dict['unused'] = unused_ids
        
        train_val_ids, id_dict['test'], train_val_binned_labels, _ = train_test_split(tvt_ids, tvt_binned_labels, stratify=tvt_binned_labels, test_size=0.25, random_state=seed)
        id_dict['train'], id_dict['val'] = train_test_split(train_val_ids, stratify=train_val_binned_labels, test_size=0.15, random_state=seed)

    elif dataset == 'adresso':
        if label == 'mmse':
            splitting_labels = data_df['mmse binned'].values
        elif label == 'dx':
            splitting_labels = data_df['dx'].values

        id_dict['train'], id_dict['val'] = train_test_split(ids, test_size=0.15, stratify=splitting_labels, random_state=seed)

    return id_dict