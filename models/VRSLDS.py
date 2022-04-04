import torch
import torch.nn as nn
from models.modules import SequentialEncoder

class VRSLDS(nn.Module):
    def __init__(self, obs_dim, discr_dim, const_dim, hidden_dim, num_rec_layers):
        super(VRSLDS, self).__init__()
        self.obs_dim = obs_dim
        self.discr_dim = discr_dim
        self.const_dim = const_dim
        self.hidden_dim = hidden_dim
        self.num_rec_layers = num_rec_layers

        self.encoder = SequentialEncoder(obs_dim, hidden_dim, discr_dim, const_dim, num_rec_layers)
        

    def _inference(self, x):
        return self.encoder(x)


    def _sample(self):

    def _decode(self):


    def _compute_elbo(self):