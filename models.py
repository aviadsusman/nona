import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Tanh, Sigmoid, BatchNorm1d, LayerNorm
from torch.nn.functional import softmax, sigmoid, one_hot, normalize
from torchvision.models import resnet50, ResNet50_Weights

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention Classifier. 
    A differentiable non-parametric (or single parameter) classifier inspired by attention and KNN.
    To classify a sample, rank the nearness of all other samples 
    and use softmax to obtain probabilities for each class.
    In the notation of attention, Q = Fe(X), K = Fe(X_train) and V = y_train where Fe is an upstream feature extractor. 
    In the notation of KNN, k = |X_train|, metric = euclidean distance or dot product, weights = softmax.
    '''
    def __init__(self, similarity='euclidean', batch_norm=None, init_temperature=1.0, agg=None):
        super(NONA, self).__init__()
        self.similarity = similarity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_norm = batch_norm # Should be num_features of data matrix
        self.agg = agg

        if self.batch_norm is not None:
            self.bn = BatchNorm1d(self.batch_norm, dtype=torch.float64, device=self.device)

        self.log_T = nn.Parameter(torch.tensor(init_temperature).log())
   
    def forward(self, x, x_n, y):
        if self.batch_norm is not None:
            x = self.bn(x)
            x_n = self.bn(x_n)

        if self.similarity == 'euclidean':
            # sim = torch.cdist(x,x_n,p=2)
            # sim = torch.max(sim) - sim
            sim = - torch.cdist(x,x_n,p=2)
        
        elif self.similarity == 'dot':
            sim = x @ torch.t(x_n)
        
        elif self.similarity == 'cos':
            x_norm = normalize(x, p=2, dim=1)
            x_n_norm = normalize(x_n, p=2, dim=1)
            sim = x_norm @ torch.t(x_n_norm)
        
        T = self.log_T.exp()
        # sim = sim / T

        if y.shape[-1] <= 1 or self.agg is None: 
            if torch.equal(x, x_n): # train
                inf_id = torch.diag(torch.full((len(sim),), float('inf'))).to(self.device)
                sim_scores = softmax(sim - inf_id, dim=1)
            
            else: # inference
                sim_scores = softmax(sim, dim=1)

            return sim_scores @ y
        
        else: # Aggregate similarities of all samples within each class, then softmax
            if self.agg == 'mean' and y.max() == 1: #ohe check
                class_sums = y.sum(dim=0, keepdim=True).clamp(min=1)
                y = y / class_sums
            
            if torch.equal(x, x_n): # train
                inv_id = (1 - torch.eye(n=len(sim))).to(self.device)
                sim_scores = (sim * inv_id) @ y
        
            else: # inference
                sim_scores = sim @ y
            
            return softmax(sim_scores, dim=1)

class NONA_NN(nn.Module):
    def __init__(self, task, input_size, hl_sizes, similarity='euclidean', classifier='nona', classes=2, agg=None):
        super(NONA_NN, self).__init__()
        self.hl_sizes = hl_sizes
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity = similarity
        self.classifier = classifier # for benchmarking
        self.task = task
        self.classes = classes
        self.agg = agg

        layer_dims = [self.input_size] + self.hl_sizes
        self.fcn = nn.ModuleList(Linear(layer_dims[i], layer_dims[i+1], dtype=torch.float64, device=self.device) for i in range(len(layer_dims)-1))
        
        self.activation = Tanh() # Tanh allows for negative feature covariance between samples
        self.norms = nn.ModuleList(BatchNorm1d(layer_dims[i+1], dtype=torch.float64, device=self.device) for i in range(len(layer_dims)-1))
        self.input_norm = BatchNorm1d(self.input_size, dtype=torch.float64, device=self.device)

        if self.classifier=='nona':
            self.output = NONA(similarity=self.similarity, agg=self.agg)
        
        elif self.classifier=='dense':
            if self.task == 'multiclass':
                self.output = Linear(layer_dims[-1], classes, dtype=torch.float64, device=self.device)
            else:
                self.output = Linear(layer_dims[-1], 1, dtype=torch.float64, device=self.device)

    def forward(self, x, x_n, y_n):
        x = self.input_norm(x)
        if self.classifier=='nona':
            x_n = self.input_norm(x_n)

        for layer, norm in zip(self.fcn, self.norms):
            x = norm(self.activation(layer(x)))

            if self.classifier=='nona':    
                # try normalizing with the learned test params and
                # try with unlearned params
                x_n = norm(self.activation(layer(x_n)))
        
        if self.classifier=='nona':
            if self.task == 'bin':
                return torch.clip(self.output(x, x_n, y_n), 0, 1)
            elif self.task == 'ordinal':
                return torch.clip(self.output(x, x_n, y_n), 0, self.classes-1)
            elif self.task == 'multiclass':
                y_n_ohe = one_hot(y_n.long()).to(self.device, torch.float64)
                return torch.clip(self.output(x, x_n, y_n_ohe), 0, 1)
            
        
        elif self.classifier=='dense':
            x = self.output(x)
            if self.task != 'multiclass':
                return (self.classes - 1) * sigmoid(x)
            else:
                return softmax(x, dim=1)