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

def kld_loss(mu, log_var, mu_prior, log_var_prior, mean_reduction=True):
    mu, mu_prior = mu.flatten(1), mu_prior.flatten(1)
    log_var, log_var_prior = log_var.flatten(1), log_var_prior.flatten(1)
    kld_per_sample = -0.5 * torch.sum(1 + log_var - log_var_prior, dim=1)
    mean_dif = (mu_prior - mu).pow(2)
    kld_per_sample += 0.5*torch.sum((1/(log_var_prior.exp()))*(log_var.exp() + mean_dif), dim=1)

    if mean_reduction:
        return torch.mean(kld_per_sample, dim = 0)
    else:
        return kld_per_sample