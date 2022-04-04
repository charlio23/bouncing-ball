import torch
import torch.nn as nn

from models.modules import SequentialEncoder
from utils.sampling import gumbel_softmax

class VRSLDS(nn.Module):
    def __init__(self, obs_dim, discr_dim, const_dim, hidden_dim, num_rec_layers):
        super(VRSLDS, self).__init__()
        self.obs_dim = obs_dim
        self.discr_dim = discr_dim
        self.const_dim = const_dim
        self.hidden_dim = hidden_dim
        self.num_rec_layers = num_rec_layers

        self.encoder = SequentialEncoder(obs_dim, hidden_dim, discr_dim, const_dim, num_rec_layers)
        self.A = nn.Linear(const_dim, const_dim)
        self.C = nn.Linear(const_dim, obs_dim)
        self.R = nn.Linear(const_dim, discr_dim-1)

    def _inference(self, x):
        return self.encoder(x)

    def _sample_discrete_states(self, z_distrib):
        return gumbel_softmax(z_distrib)

    def _sample_cont_states(self, x_mean, x_log_var):
        eps = torch.normal(mean=torch.zeros_like(x_mean)).to(x_mean.device)
        x_std = torch.minimum((x_log_var*0.5).exp())
        sample = x_mean + x_std*eps
        return gumbel_softmax(sample)

    def _decode(self):

    
    def _compute_SB_prob(self):


    def _compute_elbo(self):


    def forward(self, input):
        # Input (B, T, obs_dim)
        # Inference
        z_distr, x_mean, x_log_var = self._inference(input)
        # Sample from posterior
        z_sample = self._sample_discrete_states(z_distr)
        x_sample = self._sample_cont_states(x_mean, x_log_var)
        # Autoencode observations
        y_pred = 

