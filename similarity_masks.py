import torch
import torch.nn as nn

def sim_matrix(x, x_n, similarity):
    if similarity == 'l2':
        sim = - torch.cdist(x, x_n, p=2)
    elif similarity == 'l1':
        sim = - torch.cdist(x, x_n, p=1)
    elif similarity == 'dot':
        sim = x @ torch.t(x_n) / x.shape[1]
    elif similarity == 'cos':
        x_norm = nn.functional.normalize(x, p=2, dim=1)
        x_n_norm = nn.functional.normalize(x_n, p=2, dim=1)
        sim = x_norm @ torch.t(x_n_norm)
    
    return sim

class SoftKNNMask(nn.Module):
    '''
    Shift similarities by rankings.
    '''
    def __init__(self, eps=1e-12):
        super(SoftKNNMask, self).__init__()
        self.params = nn.Parameter(torch.normal(0,1,(2,)))
        self.eps = eps

    def forward(self, x, x_n, similarity):
        sim = sim_matrix(x,x_n, similarity)

        sim_norm = (sim - sim.min(dim=1).values[:,None]) / (sim.max(dim=1).values - sim.min(dim=1).values)[:,None]
        sim_norm = sim_norm.clamp(self.eps) # To avoid log 0 error

        # self.k.exp() # equivalent to sig(k) / (1 - sig(k)). Allows mask to range over full space.
        
        # sim_mask = (self.k ** 2) * (sim_norm).log()

        s, k = torch.sigmoid(self.params)
    
        sim_mask = (s / (1-s)) * (k - sim_norm) * (sim_norm / k).log()
        output = torch.where(sim_norm <= k, sim_mask, 0)

        return sim + output

class SoftPointwiseKNN(nn.Module):
    def __init__(self, input_dim=None, similarity=None, eps=1e-12):
        super(SoftPointwiseKNN, self).__init__()
        self.input_dim = input_dim
        self.similarity = similarity
        self.eps = eps

        if self.input_dim:
            self.knn = nn.Linear(input_dim, 2)
        
    def build_params(self, input_dim):
        '''For building knn after initialization'''
        self.input_dim = input_dim
        self.knn = nn.Linear(input_dim, 2)

    def forward(self, x, x_n, similarity):
        params = torch.sigmoid(self.knn(x))
        k, t = [col.unsqueeze(-1).clamp(min=self.eps, max=1-self.eps) for col in params.T]

        sim = sim_matrix(x,x_n, similarity)
        
        # Normalize sim for interpretable k param
        sim_min = sim.min(dim=1, keepdim=True).values
        sim_max = sim.max(dim=1, keepdim=True).values
        sim = (sim - sim_min) / (sim_max - sim_min)
        sim = sim.clamp(min=self.eps)

        shift = t/(1-t) * (k-sim) * (sim / k).log()
        output = torch.where(sim <= k, shift, 0)

        return sim + output

class HardKNNMask(nn.Module):
    '''
    Mask all similarities outside top k
    '''
    def __init__(self, k, agg=None):
        super(HardKNNMask, self).__init__()
        self.k = k + 1 # K nearest neighbors + self similarity. Remove diagonal after. 
        self.agg = agg
    
    def forward(self, x, x_n, similarity):
        sim = sim_matrix(x,x_n)
        top_k_values, top_k_indices = torch.topk(sim, self.k, dim=1)

        if self.agg is None: # Subtract off inf from all sims outside topk before softmax
            mask = float('inf') * torch.full_like(sim, 1)
            mask.scatter_(1, top_k_indices, 0)
            return sim - mask
        
        elif self.agg == 'mean': # Zero out all sims outside topk before agg
            mask = torch.full_like(sim, 0)
            mask.scatter_(1, top_k_indices, 1)
            return sim * mask

class HardSimMask(nn.Module):
    '''
    Mask out similarities under a fixed amount
    '''
    def __init__(self, k, agg=None):
        super(HardSimMask, self).__init__()
        self.agg = agg
        
        self.k = k
    
    def min_max(self, similarity, activation, d):
        self.min, self.max = get_min_max(similarity, activation, d)
    
    def get_min_sim(self, k):
        '''
        Tuneable parameter is a minimum percentage of the space that we translate into a minimum similarity.
        '''
        return (self.max - self.min) * k + self.min
    
    def forward(self, x, x_n, similarity):
        sim = sim_matrix(x,x_n)
        mask = (sim < self.get_min_sim(self.k))
        if self.agg is None:
            mask = mask.float().masked_fill(mask == 1, float('inf'))
            return sim - mask
        elif self.agg == 'mean':
            return sim * (1 - mask)

class PointwiseSoftMask(nn.Module):
    def __init__(self, mask_function=None, dims=None, eps=1e-12):
        super(PointwiseSoftMask, self).__init__()
        self.mask_function = mask_function
        if isinstance(dims, int):
            dims = [dims]
        self.dims = dims
        self.eps = eps
        
        if self.dims:
            self.build_params(self.dims)
        
    def build_params(self, dims):
        if isinstance(dims, int):
            dims = [dims]
        if self.dims is None:
            self.dims = dims

        layers = []
        if len(dims) == 1:
            layers.append(nn.Linear(dims[0], 3))
            layers.append(nn.Sigmoid())
        else:
            for in_dim, out_dim in zip(dims[:-1], dims[1:]):
                layers.append(nn.Linear(in_dim, out_dim))
                layers.append(nn.Sigmoid())
            layers.append(nn.Linear(dims[-1], 3))
            layers.append(nn.Sigmoid())

        self.params = nn.Sequential(*layers)

    def forward(self, x, x_n, similarity):
        params = self.params(x).clamp(min=self.eps, max=1-self.eps)
        a,b,s = [col.unsqueeze(-1) for col in params.T]
        b = a + b * (1-a) # Ensure 0 < a < b < 1

        # Get normalized similarities
        sim = sim_matrix(x,x_n, similarity)
        sim_norm = norm_sim(sim)

        # Adjust a to ensure at least one neighbor for comparison
        if torch.equal(x,x_n): # train
            top_sims = (sim_norm - sim_norm.diag().diag()).max(dim=1)[0]
        else:
            top_sims = sim_norm.max(dim=1)[0]
        a = torch.minimum(a, top_sims.unsqueeze(-1)) - self.eps

        # Compute softmask
        num = torch.abs(sim_norm - a).pow(1 / s).clamp(self.eps)
        denom = num + torch.abs(b - sim_norm).pow(1 / s).clamp(self.eps)
        soft_mask = num / denom

        soft_mask[sim_norm < a] = 0
        soft_mask[sim_norm > b] = 1

        return sim + soft_mask.log()

class UniformSoftMask(nn.Module):
    def __init__(self, mask_function=None, eps=1e-12): # make one a,b,s mask with passed in function
        super(UniformSoftMask, self).__init__()
        self.mask_function = mask_function
        self.eps = eps
        self.params = nn.Parameter(torch.normal(0,1,(3,)))

    def forward(self, x, x_n, similarity):
        a,b,s = torch.sigmoid(self.params).clamp(self.eps, 1 - self.eps)
        b = a + b * (1-a) # Ensure 0 < a < b < 1

        # Get normalized similarities
        sim = sim_matrix(x,x_n, similarity)
        sim_norm = norm_sim(sim)
        
        # Adjust a to ensure at least one neighbor for comparison
        if torch.equal(x,x_n):
            top_sims = (sim_norm - sim_norm.diag().diag()).max(dim=1)[0]
        else:
            top_sims = sim_norm.max(dim=1)[0]
        a = torch.min(a, top_sims.min()) - self.eps

        # Compute softmask
        num = torch.abs(sim_norm - a).pow(1 / s).clamp(self.eps)
        denom = num + torch.abs(b - sim_norm).pow(1 / s).clamp(self.eps)
        soft_mask = num / denom

        soft_mask[sim_norm < a] = 0
        soft_mask[sim_norm > b] = 1

        return sim + soft_mask.log()

def norm_sim(sim):
    sim_min = sim.min(dim=1, keepdim=True)[0]
    sim_max = sim.max(dim=1, keepdim=True)[0]
    sim_norm = (sim - sim_min) / (sim_max - sim_min)
    return sim_norm

def get_min_max(similarity, activation, d):
    '''
    Min and max similarities are a function of:
    d: dimension of final embedding space.
    activation: Boundary of the embedding space.
    similarity: Bounds on similarities within the space.
    '''
    if similarity == 'dot':
        max_similarity = d
        min_similarity = 0

        if activation == 'tanh':
            min_similarity = - d        

    elif similarity == 'l2':
        max_similarity = 0
        min_similarity = - (d ** (1/2))
        
        if activation == 'tanh':
            min_similarity *= 2
    
    elif similarity == 'l1':
        max_similarity = 0
        min_similarity = - d
        
        if activation == 'tanh':
            min_similarity *= 2
        
    return min_similarity, max_similarity

def sigmoidal(sim_norm, a, b, s, eps=1e-12):
    num = torch.abs(sim_norm - a).pow(1 / s).clamp(eps)
    denom = num + torch.abs(b - sim_norm).pow(1 / s).clamp(eps)
    soft_mask = num / denom

    return soft_mask

def exponential(sim_norm, a, b, s, eps=1e-12):
    arg = ((b - sim_norm) / (sim_norm - a)) ** 2
    soft_mask = s ** arg
    # mask = torch.where(sim_norm < a, torch.tensor(0.0, device=mask.device), mask)
    # mask = torch.where(sim_norm > b, torch.tensor(1.0, device=mask.device), mask)

    return soft_mask
