import torch
import torch.nn as nn
from torch.nn import Linear, Dropout, ReLU, Tanh, Sigmoid, BatchNorm1d, LayerNorm
from torch.nn.functional import softmax, sigmoid, one_hot, normalize
from similarity_masks import SoftKNNMask, HardKNNMask, SoftSimMask, HardSimMask, SoftPointwiseKNN

class NONA(nn.Module):
    '''
    Nearness of Neighbors Attention Predictor. 
    A differentiable non-parametric (or single parameter) predictor inspired by attention and KNN.
    To classify a sample, rank the nearness of all other samples 
    and use softmax to obtain probabilities for each class.
    In the notation of attention, Q = Fe(X), K = Fe(X_n) and V = y_n where Fe is an upstream feature extractor.
    In the notation of KNN, k = |X_train|, metric = euclidean distance or dot product, weights = softmax.
    '''
    def __init__(self, similarity='euclidean', mask=None, agg=None, batch_norm=None, dtype=torch.float64):
        super(NONA, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        self.similarity = similarity
        
        if mask:
            self.mask = mask.to(self.device).to(self.dtype)
        
        self.agg = agg # For class similarity aggregation in multiclass prediciton.

        self.batch_norm = batch_norm # Num input features. Used for standalone NONA.
        if self.batch_norm:
            self.bn = BatchNorm1d(self.batch_norm, dtype=self.dtype, device=self.device)
    
    def softmax_predict(self, sim, y_n, train):
        '''
        Make predictions using sim matrix and neighbor labels.
        Correct for self-similarity during training.
        Account for aggregation strategy in multiclass prediction.
        '''
        if not self.agg: 
            if train:
                inf_id = torch.diag(torch.full((len(sim),), float('inf'))).to(self.device)
                sim -= inf_id
            
            sim_scores = softmax(sim, dim=1)

            return sim_scores @ y_n

        elif self.agg == 'mean':
            class_sums = y_n.sum(dim=0, keepdim=True).clamp(min=1)
            
            if train:
                sim -= sim.diag().diag()
                class_sums = (class_sums - 1).clamp(min=1)
            
            y_n /= class_sums

            sim_scores = sim @ y_n

            return softmax(sim_scores, dim=1)
        
    def forward(self, x, x_n, y_n):
        if self.batch_norm:
            x = self.bn(x)
            x_n = self.bn(x_n)

        # Create similarity matrix between embeddings of X and embeddings of X_n

        if type(self.mask) == SoftPointwiseKNN:
            sim = self.mask(x,x_n, similarity=self.similarity)

        else:
            if self.similarity == 'euclidean':
                sim = - torch.cdist(x, x_n, p=2)
            
            elif self.similarity == 'dot':
                sim = x @ torch.t(x_n)
            
            elif self.similarity == 'cos':
                x_norm = normalize(x, p=2, dim=1)
                x_n_norm = normalize(x_n, p=2, dim=1)
                sim = x_norm @ torch.t(x_n_norm)
            
            if self.mask:
                sim = self.mask(sim)
        
        train = torch.equal(x, x_n)
        return self.softmax_predict(sim, y_n, train)

class NONA_NN(nn.Module):
    '''
    Attach intermediate dense layers to either a NONA or dense prediction layer.
    '''
    def __init__(self, input_size, hl_sizes=list(), predictor='nona', similarity='euclidean', mask=None, multiclass=None, agg=None, dtype=torch.float64):
        super(NONA_NN, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype

        self.input_size = input_size
        self.hl_sizes = hl_sizes
        self.predictor = predictor # for benchmarking
        self.similarity = similarity
        self.mask = mask
        self.multiclass = multiclass # For multiclass prediction. Initialize with num_classes.
        self.agg = agg
        
        layer_dims = [self.input_size] + self.hl_sizes

        if hl_sizes != list(): # Intermediate MLP between feature extractor and predictor.
            self.fcn = nn.ModuleList(Linear(layer_dims[i], layer_dims[i+1], dtype=self.dtype, device=self.device) for i in range(len(layer_dims)-1))
        
            self.activation = Tanh() # Tanh allows for negative feature covariance between samples

            self.norms = nn.ModuleList(BatchNorm1d(i, dtype=self.dtype, device=self.device) for i in layer_dims[:-1])

        if self.predictor=='nona':
            self.output_layer = NONA(similarity=self.similarity, mask=self.mask, agg=self.agg, dtype=self.dtype)
            
            if hasattr(self.mask, 'min_max'): # Determine size of final embedding space for similarity normalization
                self.output_layer.mask.min_max(similarity=self.similarity, activation='tanh', d=layer_dims[-1])
            
            elif hasattr(self.mask, 'input_dim'):
                self.output_layer.mask.build_knn(layer_dims[-1])
                self.output_layer.mask.to(self.device).to(self.dtype)
        
        elif self.predictor=='dense':
            if self.multiclass:
                self.output_layer = Linear(layer_dims[-1], self.multiclass, dtype=self.dtype, device=self.device)
            else:
                self.output_layer = Linear(layer_dims[-1], 1, dtype=self.dtype, device=self.device)

    def forward(self, x, x_n, y_n, get_embeddings=False):
        if self.hl_sizes != list():
            for layer, norm in zip(self.fcn, self.norms):
                x = self.activation(layer(norm(x)))

                if self.predictor=='nona' or get_embeddings==True:
                    x_n = self.activation(layer(norm(x_n)))
            
        if self.predictor=='nona':
            if self.multiclass:            
                y_n = one_hot(y_n.long()).to(self.device, self.dtype)

            output = torch.clip(self.output_layer(x, x_n, y_n), 0, 1)
        
        elif self.predictor=='dense':
            logits = self.output_layer(x)
            
            if self.multiclass:
                output = softmax(logits, dim=1)
            else:
                output = logits.squeeze()

        if get_embeddings:
            output = [output, x, x_n]
        
        return output

class NONA_FT(nn.Module):
    def __init__(self, feature_extractor, hl_sizes=list(), predictor='nona', similarity='euclidean', mask=None, multiclass=None, agg=None, dtype=torch.float64):
        '''
        Attach a feature extractor to a NONA neural network.
        '''
        super(NONA_FT, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.dtype = dtype
        
        self.feature_extractor = feature_extractor.to(self.device)
        for param in self.feature_extractor.parameters():
            param.requires_grad = True
        
        self.input_size = get_output_size(self.feature_extractor)
        self.hl_sizes = hl_sizes
        
        self.predictor = predictor # Dense for benchmarking
        self.similarity = similarity
        self.mask = mask

        self.multiclass = multiclass
        self.agg = agg

        self.nona = NONA_NN(
            input_size=self.input_size, 
            hl_sizes=self.hl_sizes,
            predictor=self.predictor, 
            similarity=self.similarity, 
            mask=self.mask,
            multiclass=self.multiclass, 
            agg=self.agg, 
            dtype=self.dtype)

    def forward(self, x, x_n, y_n, get_embeddings=False):
        if isinstance(x, dict): # bert style text transformer
            x = self.feature_extractor(**x).last_hidden_state[:,0,:]
            x_n = self.feature_extractor(**x_n).last_hidden_state[:,0,:]
        
        else: # cnn
            x = self.feature_extractor(x)
            x_n = self.feature_extractor(x_n)

        return self.nona(x, x_n, y_n, get_embeddings)

def get_output_size(model):
    ''' Finds the number of output features of a feature extractor'''
    for name, module in model.named_modules():
        pass
    
    if hasattr(module, 'out_features'):
        return module.out_features
    elif hasattr(module, 'normalized_shape'):
        return module.normalized_shape[0]