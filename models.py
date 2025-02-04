import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Tanh, Sigmoid, BatchNorm1d, LayerNorm
from torch.nn.functional import softmax, sigmoid, one_hot
from torchvision.models import resnet50, ResNet50_Weights

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention Classifier. 
    A differentiable non-parametric classifier inspired by attention and KNN.
    To classify a sample, rank the nearness of all other samples 
    and use softmax to obtain probabilities for each class.
    In the notation of attention, Q = Fe(X), K = Fe(X_train) and V = y_train where Fe is an upstream feature extractor. 
    In the notation of KNN, k = |X_train|, metric = euclidean distance or dot product, weights = softmax
    '''
    def __init__(self, similarity='euclidean'):
        super(NONA, self).__init__()
        self.similarity = similarity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    def forward(self, x, x_n, y):
        if self.similarity == 'euclidean':
            sim = torch.cdist(x,x_n,p=2)
            sim = torch.max(sim) - sim
        
        elif self.similarity == 'dot':
            sim = x @ torch.t(x_n)
        
        if torch.equal(x, x_n): # train
            inf_id = torch.diag(torch.full((len(sim),), float('inf'))).to(self.device)
            sim_scores = softmax(sim - inf_id, dim=1)
        
        else: # eval
            sim_scores = softmax(sim, dim=1)

        return sim_scores @ y

class NONA_NN(nn.Module):
    def __init__(self, task, input_size, hl_sizes, similarity='euclidean', classifier='nona', classes=2):
        super(NONA_NN, self).__init__()
        self.hl_sizes = hl_sizes
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity = similarity
        self.classifier = classifier # for benchmarking
        self.task = task
        self.classes = classes

        layer_dims = [self.input_size] + self.hl_sizes
        self.fcn = nn.ModuleList(Linear(layer_dims[i], layer_dims[i+1], dtype=torch.float64, device=self.device) for i in range(len(layer_dims)-1))
        
        self.activation = Tanh()
        self.norms = nn.ModuleList(BatchNorm1d(layer_dims[i+1], dtype=torch.float64, device=self.device) for i in range(len(layer_dims)-1))

        if self.classifier=='nona':
            self.output = NONA(similarity=self.similarity)
        
        elif self.classifier=='dense':
            if self.task == 'multiclass':
                self.output = Linear(layer_dims[-1], classes, dtype=torch.float64, device=self.device)
            else:
                self.output = Linear(layer_dims[-1], 1, dtype=torch.float64, device=self.device)

    def forward(self, x, x_n, y_n):
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