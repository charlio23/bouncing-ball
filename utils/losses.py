from statistics import variance
import numpy as np
import torch

def nll_gaussian(mu_x, log_var, target):
    eps = torch.FloatTensor([1e-6]).to(log_var.device)
    
    log_var = torch.maximum(log_var, torch.log(eps))
    neg_log_p = (mu_x - target) ** 2 / (2 * log_var.exp())
    _, _, dim = log_var.size()
    var_term = 0.5 * dim * (np.log((2 * np.pi)) + log_var.sum(dim=-1))
    return (neg_log_p.sum() + var_term.sum())/ (target.size(0))

def kld_loss(mu, log_var, mean_reduction=True):
    mu = mu.flatten(1)
    log_var = log_var.flatten(1)
    kld_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
    if mean_reduction:
        return torch.mean(kld_per_sample, dim = 0)
    else:
        return kld_per_sample