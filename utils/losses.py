import numpy as np
import torch
import torch.nn.functional as F

def nll_gaussian(mu_x, log_var, target):
    eps = torch.FloatTensor([1e-6]).to(log_var.device)
    
    log_var = torch.maximum(log_var, torch.log(eps))
    neg_log_p = (mu_x - target) ** 2 / (2 * log_var.exp())
    _, _, dim = log_var.size()
    var_term = 0.5 * dim * (np.log((2 * np.pi)) + log_var.sum(dim=-1))
    return (neg_log_p.sum() + var_term.sum())/ (target.size(0))

def nll_gaussian_var_fixed(preds, target, variance, add_const=True):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    neg_log_p = (preds - target) ** 2 / (2 * variance)
    if add_const:
        const = 0.5 * np.log(2 * np.pi * variance)
        neg_log_p += const
    return neg_log_p.sum() / (target.size(0))

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

def kld_loss_standard(mu, log_var, mean_reduction=True):
    mu = mu.flatten(1)
    log_var = log_var.flatten(1)
    kld_per_sample = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim = 1)
    if mean_reduction:
        return torch.mean(kld_per_sample, dim = 0)
    else:
        return kld_per_sample

def kl_categorical(preds, prior, eps=1e-16):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - torch.log(prior + eps))
    return kl_div.sum() / preds.size(0)

def kl_categorical_uniform(preds, num_categories, eps=1e-16):
    """Based on https://github.com/ethanfetaya/NRI (MIT License)."""
    kl_div = preds * (torch.log(preds + eps) - np.log(1/num_categories + eps))
    return kl_div.sum() / preds.size(0)

def mse_through_time(input, target, visual=True):
    total_mse = F.mse_loss(input, target, reduction='none')
    total_mse = total_mse.transpose(0,1)
    if visual:
        return torch.mean(total_mse, dim = (1,2,3,4))
    else:
        return torch.mean(total_mse, dim = (1,2))
