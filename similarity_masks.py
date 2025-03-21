import torch
import torch.nn as nn

class SoftKNNMask(nn.Module):
    '''
    Shift similarities by rankings.
    '''
    def __init__(self):
        super(SoftKNNMask, self).__init__()
        self.k = nn.Parameter(torch.normal(0,0.1, (1,)))
        self.s = nn.Parameter(torch.normal(0,1, (1,)))

    def forward(self, sim):
        # Converges to SoftSimMask when embedding space is sparse
        sim_norm = (sim - sim.min(dim=1).values[:,None]) / (sim.max(dim=1).values - sim.min(dim=1).values)[:,None]
        sim_norm = sim_norm.clamp(min=1e-8) # To avoid log 0 error

        # self.k.exp() # equivalent to sig(k) / (1 - sig(k)). Allows mask to range over full space.
        
        # sim_mask = (self.k ** 2) * (sim_norm).log()

        k = torch.sigmoid(self.k)
        sim_mask = (self.s ** 2) * (k - sim_norm) * (sim_norm / k).log()
        output = torch.where(sim_mask > k, torch.tensor(0.0, device=sim_norm.device), sim_mask)

        return sim + output

class SoftPointwiseKNN(nn.Module):
    def __init__(self, input_dim=None):
        super(SoftPointwiseKNN, self).__init__()
        self.input_dim = input_dim
        
        if self.input_dim:
            self.knn = nn.Linear(input_dim, 2)
        
    def build_knn(self, input_dim):
        '''For building knn after initialization'''
        self.input_dim = input_dim
        self.knn = nn.Linear(input_dim, 2)

    def forward(self, x, x_n, similarity):
        params = torch.sigmoid(self.knn(x))
        k, s = [col.unsqueeze(-1) for col in params.T]
        
        if similarity == 'euclidean':
            sim = - torch.cdist(x, x_n, p=2)
        elif similarity == 'dot':
            sim = x @ torch.t(x_n)
        
        # Normalize sim for interpretable k param
        sim = (sim - sim.min(dim=1).values[:,None]) / (sim.max(dim=1).values - sim.min(dim=1).values)[:,None]
        sim = sim.clamp(min=1e-8) # To avoid log 0 error

        # shift = torch.tan(s * torch.pi / 2) * (k-sim) * (sim / k).log()
        shift = s/(1-s) * (k-sim) * (sim / k).log()
        output = torch.where(sim <= k, shift, torch.zeros_like(sim))

        return sim + output

class HardKNNMask(nn.Module):
    '''
    Mask all similarities outside top k
    '''
    def __init__(self, k, agg=None):
        super(HardKNNMask, self).__init__()
        self.k = k + 1 # K nearest neighbors + self similarity. Remove diagonal after. 
        self.agg = agg
    
    def forward(self, sim):
        top_k_values, top_k_indices = torch.topk(sim, self.k, dim=1)

        if self.agg is None: # Subtract off inf from all sims outside topk before softmax
            mask = float('inf') * torch.full_like(sim, 1)
            mask.scatter_(1, top_k_indices, 0)
            return sim - mask
        
        elif self.agg == 'mean': # Zero out all sims outside topk before agg
            mask = torch.full_like(sim, 0)
            mask.scatter_(1, top_k_indices, 1)
            return sim * mask

class SoftSimMask(nn.Module):
    '''
    Shift similarities by absolute similarity.
    '''
    def __init__(self):
        super(SoftSimMask, self).__init__()
        self.k = nn.Parameter(torch.randn(1))
        
    def min_max(self, similarity, activation, d):
        self.min, self.max = get_min_max(similarity, activation, d)
    
    def forward(self, sim):
        sim_norm = (sim - self.min) / (self.max - self.min)
        
        # k = self.k.exp() # equivalent to sig(k) / (1 - sig(k)). Allows mask to range over full space.
        sim_mask = (self.k ** 2) * (sim_norm).log()

        return sim + sim_mask

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
    
    def forward(self, sim):
        mask = (sim < self.get_min_sim(self.k))
        if self.agg is None:
            mask = mask.float().masked_fill(mask == 1, float('inf'))
            return sim - mask
        elif self.agg == 'mean':
            return sim * (1 - mask)

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
            min_similarity -= d        

    elif similarity == 'euclidean':
        max_similarity = 0
        min_similarity = - (d ** (1/2))
        
        if activation == 'tanh':
            min_similarity *= 2
    
    return min_similarity, max_similarity

class SoftSigMask(nn.Module):
    '''
    General S shaped function from I to I.
    https://www.desmos.com/calculator/ba2kqqpnih
    '''
    def __init__(self):
        super().__init__()
        self.a = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))
        self.n = nn.Parameter(torch.randn(1))
    
    def min_max(self, similarity, activation, d):
        self.min, self.max = get_min_max(similarity, activation, d)

    def forward(self, sim):
        # Ensure 0 < a < b < 1
        a = torch.sigmoid(self.a)
        b = a + torch.sigmoid(self.b) * (1 - a)
        n = torch.sigmoid(self.n)

        sim_norm = (sim - self.min) / (self.max - self.min)

        num = (sim_norm - a).clamp(min=1e-6).pow(1 / n)  # Avoid zero issues
        denom = num + (b - sim_norm).clamp(min=1e-6).pow(1 / n)

        soft_mask = num / denom

        output = torch.where(sim_norm < a, torch.tensor(0.0, device=sim_norm.device), soft_mask)
        output = torch.where(sim_norm > b, torch.tensor(1.0, device=sim_norm.device), output)

        return soft_mask