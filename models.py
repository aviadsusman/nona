import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Tanh, Sigmoid, BatchNorm1d, LayerNorm
from torch.nn.functional import softmax, sigmoid, one_hot, normalize

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention Predictor. 
    A differentiable non-parametric (or single parameter) predictor inspired by attention and KNN.
    To classify a sample, rank the nearness of all other samples 
    and use softmax to obtain probabilities for each class.
    In the notation of attention, Q = Fe(X), K = Fe(X_n) and V = y_n where Fe is an upstream feature extractor
    and X\ 
    In the notation of KNN, k = |X_train|, metric = euclidean distance or dot product, weights = softmax.
    '''
    def __init__(self, similarity='euclidean', batch_norm=None, agg=None, dtype=torch.float64):
        super(NONA, self).__init__()
        self.similarity = similarity
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.batch_norm = batch_norm # Should be num_features of data matrix
        self.agg = agg
        self.dtype = dtype

        if self.batch_norm is not None:
            self.bn = BatchNorm1d(self.batch_norm, dtype=self.dtype, device=self.device)
            
    def forward(self, x, x_n, y):
        if self.batch_norm is not None:
            x = self.bn(x)
            x_n = self.bn(x_n)

        if self.similarity == 'euclidean':
            sim = - torch.cdist(x,x_n,p=2)
        
        elif self.similarity == 'dot':
            sim = x @ torch.t(x_n)
        
        elif self.similarity == 'cos':
            x_norm = normalize(x, p=2, dim=1)
            x_n_norm = normalize(x_n, p=2, dim=1)
            sim = x_norm @ torch.t(x_n_norm)

        if y.shape[-1] <= 1 or self.agg is None: # Not  multiclass with aggregated similarities
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
    def __init__(self, task, input_size, hl_sizes=list(), similarity='euclidean', predictor='nona', classes=2, agg=None, dtype=torch.float64, skip_final_bn=False):
        super(NONA_NN, self).__init__()
        self.hl_sizes = hl_sizes
        self.input_size = input_size
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity = similarity
        self.predictor = predictor # for benchmarking
        self.task = task
        self.classes = classes
        self.agg = agg
        self.dtype = dtype
        self.mlp = mlp
        self.skip_final_bn = skip_final_bn # temp attribute to test effect of final bn on embeddings/performance

        layer_dims = [self.input_size] + self.hl_sizes

        if hl_sizes != list():
            self.fcn = nn.ModuleList(Linear(layer_dims[i], layer_dims[i+1], dtype=self.dtype, device=self.device) for i in range(len(layer_dims)-1))
        
            self.activation = Tanh() # Tanh allows for negative feature covariance between samples
            
            self.norms = nn.ModuleList(BatchNorm1d(layer_dims[i+1], dtype=self.dtype, device=self.device) for i in range(len(layer_dims)-1))
            
            if self.skip_final_bn: 
                self.norms[-1] = nn.Identity(layer_dims[-1], dtype=self.dtype, device=self.device)
        
        self.input_norm = BatchNorm1d(self.input_size, dtype=self.dtype, device=self.device)

        if self.predictor=='nona':
            self.output = NONA(similarity=self.similarity, agg=self.agg, dtype=self.dtype)
        
        elif self.predictor=='dense':
            if self.task == 'multiclass':
                self.output_layer = Linear(layer_dims[-1], self.classes, dtype=self.dtype, device=self.device)
            else:
                self.output_layer = Linear(layer_dims[-1], 1, dtype=self.dtype, device=self.device)

    def forward(self, x, x_n, y_n, get_embeddings=False):
        x = self.input_norm(x)
        if self.predictor=='nona':
            x_n = self.input_norm(x_n)
        
        if self.hl_sizes != list():
            for layer, norm in zip(self.fcn, self.norms):
                x = norm(self.activation(layer(x)))

                if self.predictor=='nona':    
                    x_n = norm(self.activation(layer(x_n)))
            
        if self.predictor=='nona':
            if self.task in ['binary', 'regression']:
                output = torch.clip(self.output_layer(x, x_n, y_n), 0, 1)
            
            elif self.task == 'ordinal':
                output = torch.clip(self.output_layer(x, x_n, y_n), 0, self.classes-1)
            
            elif self.task == 'multiclass':
                y_n_ohe = one_hot(y_n.long()).to(self.device, self.dtype)
                output = torch.clip(self.output_layer(x, x_n, y_n_ohe), 0, 1)
        
        elif self.predictor=='dense':
            logits = self.output_layer(x)
            
            if self.task != 'multiclass':
                output = ((self.classes - 1) * sigmoid(logits)).squeeze()
            
            else:
                output = softmax(logits, dim=1)
        
        if get_embeddings:
            output = [output, x]
        
        return output
            
        

class NONA_FT(nn.Module):
    def __init__(self, task, feature_extractor, hl_sizes, similarity='euclidean', predictor='nona', classes=2, agg=None, dtype=torch.float64, mlp=True):
        super(NONA_FT, self).__init__()
        self.task = task
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.feature_extractor = feature_extractor.to(self.device)
        self.hl_sizes = hl_sizes
        self.similarity = similarity
        self.predictor = predictor # for benchmarking
        self.classes = classes
        self.agg = agg
        self.dtype = dtype
        self.mlp = mlp

        for name, module in self.feature_extractor.named_modules():
            pass
        self.input_size = getattr(self.feature_extractor, name).out_features

        for param in self.feature_extractor.parameters():
            param.requires_grad = True

        self.nona = NONA_NN(
            task=self.task, 
            input_size=self.input_size, 
            hl_sizes=self.hl_sizes, 
            similarity=self.similarity, 
            predictor=self.predictor, 
            classes=self.classes, 
            agg=self.agg, 
            dtype=self.dtype, 
            mlp=self.mlp)

    def forward(self, x, x_n, y_n, get_embeddings=False):
        x = self.feature_extractor(x)
        x_n = self.feature_extractor(x_n)

        return self.nona(x, x_n, y_n, get_embeddings)