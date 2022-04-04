import torch
import torch.nn as nn

from models.modules import SequentialEncoder
from utils.sampling import gumbel_softmax, my_softmax
from utils.losses import kl_categorical, kld_loss, nll_gaussian_var_fixed

class VRSLDS(nn.Module):
    def __init__(self, obs_dim, discr_dim, cont_dim, hidden_dim, num_rec_layers, tau=0.5):
        super(VRSLDS, self).__init__()
        self.obs_dim = obs_dim
        self.discr_dim = discr_dim
        self.cont_dim = cont_dim
        self.hidden_dim = hidden_dim
        self.num_rec_layers = num_rec_layers
        self.tau = tau

        # Network parameters
        self.encoder = SequentialEncoder(obs_dim, hidden_dim, discr_dim, cont_dim, num_rec_layers)
        self.C = nn.ModuleList([
            nn.Linear(cont_dim, obs_dim) for i in range(self.discr_dim)
        ])
        self.A = nn.ModuleList([
            nn.Linear(cont_dim, cont_dim) for i in range(self.discr_dim)
        ])
        self.R = nn.ModuleList([
            nn.Linear(cont_dim, discr_dim-1) for i in range(self.discr_dim)
        ])

    def _inference(self, x):
        return self.encoder(x)

    def _sample_discrete_states(self, z_distrib):
        return gumbel_softmax(z_distrib, tau=self.tau)

    def _sample_cont_states(self, x_mean, x_log_var):
        eps = torch.normal(mean=torch.zeros_like(x_mean)).to(x_mean.device)
        x_std = (x_log_var*0.5).exp()
        sample = x_mean + x_std*eps
        return gumbel_softmax(sample)

    def _compute_SB_prob(self, prob_vector):
        B, T, _ = prob_vector.size()
        sigmoid_vector = torch.cat(
            [torch.sigmoid(prob_vector), torch.ones(B, T, 1).to(prob_vector.device)], dim=-1)
        SB_cummulant = torch.cat(
            [torch.ones(B, T, 1).to(prob_vector.device), (1 - torch.sigmoid(prob_vector)).cumprod(dim=-1)], dim=-1)
        prob_SB = sigmoid_vector*SB_cummulant
        return prob_SB

    def _decode(self, z_sample, x_sample):
        y_pred = torch.zeros(x_sample.size(0), x_sample.size(1), self.obs_dim).to(x_sample.device)
        x_next = torch.zeros(x_sample.size(0), x_sample.size(1)-1, self.cont_dim).to(x_sample.device)
        z_next = torch.zeros(x_sample.size(0), x_sample.size(1)-1, self.discr_dim-1).to(x_sample.device)
        for i in range(self.discr_dim):
            # y_t | x_t, z_t
            y_pred += self.C[i](x_sample)*z_sample[:,:,i:i+1]
            # x_t+1 | x_t, z_t+1
            x_next += self.A[i](x_sample[:,:-1,:])*z_sample[:,1:,i:i+1]
            # z_t+1 | x_t, z_t
            z_next += self.R[i](x_sample[:,:-1,:])*z_sample[:,:-1,i:i+1]
        z_next = self._compute_SB_prob(z_next)
        return y_pred, z_next, x_next
    
    def _compute_elbo(self, y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next):
        # Fixed variance
        # Reconstruction Loss p(y_t | x_t, z_t)
        elbo = nll_gaussian_var_fixed(y_pred, input, variance=1e-4)
        # Continous KL term p(z_1) and p(z_t | z_t-1, x_t-1)
        # Skip p(z_1) for now
        elbo += kl_categorical(z_distr[:,1:,:], z_next)
        # Discrete KL term p(x_1) and p(x_t | z_t, x_t-1)
        # Skip p(x_1) for now
        log_var_prior = torch.ones_like(x_next).to(x_next.device)*torch.log(torch.tensor(1e-4))
        elbo += kld_loss(x_mean[:,1:,:], x_log_var[:,1:,:], x_next, log_var_prior)

        return elbo

    def forward(self, input):
        # Input (B, T, obs_dim)
        # Inference
        z_distr, x_mean, x_log_var = self._inference(input)
        # Sample from posterior
        z_sample = self._sample_discrete_states(z_distr)
        # Convert logits to normalized distribution
        z_distr = my_softmax(z_distr, -1)
        x_sample = self._sample_cont_states(x_mean, x_log_var)
        # Forward step
        # Autoencode + 1-step computation
        y_pred, z_next, x_next = self._decode(z_sample, x_sample)
        elbo = self._compute_elbo(y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next)

        return y_pred, x_sample, z_sample, elbo


if __name__=="__main__":
    # Trial run
    net = VRSLDS(obs_dim=2, discr_dim=4, cont_dim=2, hidden_dim=128, num_rec_layers=2)

    sample = torch.rand((1,3,2))
    torch.autograd.set_detect_anomaly(True)
    y_pred, x_sample, z_sample, elbo = net(sample)
    print(y_pred.size())
    print(x_sample.size())
    print(z_sample.size())
    print(elbo)
    elbo.backward()