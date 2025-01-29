import torch
import torch.nn as nn
from torch.nn import Linear, Dropout
from torch.nn.functional import softmax, sigmoid

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention Classifier. 
    A differentiable non-parametric classifier inspired by attention and KNN.
    To classify a sample, rank the nearness of all other samples 
    (via euclidean distance or dot product similarity) and use softmax to obtain
    probabilities for each class.
    In the notation of attention, Q,K=x and V=y. 
    '''
    def __init__(self, similarity='euclidean'):
        super(NONA, self).__init__()
        self.similarity = similarity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
   
    def forward(self, x, x_neighbors, y):
        if self.similarity == 'euclidean':
            sim = torch.cdist(x,x_neighbors,p=2)
            sim = torch.max(sim) - sim
        
        elif self.similarity == 'dot':
            sim = x @ torch.t(x_neighbors)
        
        if torch.equal(x, x_neighbors): # train
            inf_id = torch.diag(torch.full((len(sim),), float('inf')))
            sim_scores = softmax(sim - inf_id, dim=1)
        
        else: # eval
            sim_scores = softmax(sim, dim=1)

        return sim_scores @ y

class NONA_NN(nn.Module):
    def __init__(self, input_size, hl_sizes, similarity='euclidean', classifier='nona'):
        super(NONA_NN, self).__init__()
        self.hl_sizes = hl_sizes
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity = similarity
        self.classifier = classifier # for benchmarking

        layer_dims = [self.input_size] + self.hl_sizes
        self.fcn = nn.ModuleList(Linear(layer_dims[i], layer_dims[i+1], dtype=torch.float64, device=self.device) for i in range(len(layer_dims)-1))
        
        self.dropout = Dropout(0.2)

        if self.classifier=='nona':
            self.output = NONA(similarity=self.similarity)
        
        elif self.classifier=='dense':
            self.output = Linear(layer_dims[-1], 1, dtype=torch.float64, device=self.device)

    def forward(self, x, x_neighbors, y):
        for layer in self.fcn:
            x = self.dropout(layer(x))

            if self.classifier=='nona':    
                x_neighbors = layer(x_neighbors)
        
        if self.classifier=='nona':
            return self.output(x, x_neighbors, y)
        
        elif self.classifier=='dense':
            x = self.output(x)
            return sigmoid(x).squeeze()