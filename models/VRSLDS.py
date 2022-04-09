import torch
import torch.nn as nn

from models.modules import SequentialEncoder
from utils.sampling import gumbel_softmax, my_softmax
from utils.losses import kl_categorical, kl_categorical_uniform, kld_loss, kld_loss_standard, nll_gaussian_var_fixed

class VRSLDS(nn.Module):
    def __init__(self, obs_dim, discr_dim, cont_dim, hidden_dim, num_rec_layers, tau=0.5, bidirectional=True, beta=1, SB=False):
        super(VRSLDS, self).__init__()
        self.obs_dim = obs_dim
        self.discr_dim = discr_dim
        self.cont_dim = cont_dim
        self.hidden_dim = hidden_dim
        self.num_rec_layers = num_rec_layers
        self.tau = tau
        self.bidirectional = bidirectional
        self.beta = beta
        self.SB = SB

        # Network parameters
        self.encoder = SequentialEncoder(obs_dim, hidden_dim, discr_dim, cont_dim, num_rec_layers, bidirectional)
        self.C = nn.ModuleList([
            nn.Linear(cont_dim, obs_dim) for i in range(self.discr_dim)
        ])
        self.A = nn.ModuleList([
            nn.Linear(cont_dim, cont_dim) for i in range(self.discr_dim)
        ])
        self.R = nn.ModuleList([
            nn.Linear(cont_dim, (discr_dim-1) if SB else discr_dim) for i in range(self.discr_dim)
        ])
        self._init_weights()

    def _init_weights(self):
        for i in range(self.discr_dim):
            self.R[i].weight.data.fill_(1e-5)
            self.R[i].bias.data.fill_(1/self.discr_dim)

            self.A[i].weight.data.fill_(0)

    def _inference(self, x):
        return self.encoder(x)

    def _sample_discrete_states(self, z_distrib):
        return gumbel_softmax(z_distrib, tau=self.tau)

    def _sample_cont_states(self, x_mean, x_log_var):
        eps = torch.normal(mean=torch.zeros_like(x_mean)).to(x_mean.device)
        x_std = (x_log_var*0.5).exp()
        sample = x_mean + x_std*eps
        return sample

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
        z_next = torch.zeros(x_sample.size(0), x_sample.size(1)-1, 
            (self.discr_dim-1) if self.SB else self.discr_dim).to(x_sample.device)
        for i in range(self.discr_dim):
            # y_t | x_t, z_t
            y_pred += self.C[i](x_sample)*z_sample[:,:,i:i+1]
            # x_t+1 | x_t, z_t+1
            x_next += self.A[i](x_sample[:,:-1,:])*z_sample[:,1:,i:i+1]
            # z_t+1 | x_t, z_t
            z_next += self.R[i](x_sample[:,:-1,:])*z_sample[:,:-1,i:i+1]
        if self.SB:
            z_next = self._compute_SB_prob(z_next)
        else:
            z_next = z_next.softmax(-1)

        return y_pred, z_next, x_next
    
    def _compute_elbo(self, y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next):
        # Fixed variance
        # Reconstruction Loss p(y_t | x_t, z_t)
        nll = nll_gaussian_var_fixed(y_pred, input, variance=1e-4)
        # Continous KL term p(z_1) and p(z_t | z_t-1, x_t-1)
        kld = kl_categorical(z_distr[:,1:,:], z_next) + kl_categorical_uniform(z_distr[:,0,:], self.discr_dim)
        # Discrete KL term p(x_1) and p(x_t | z_t, x_t-1)
        log_var_prior = torch.ones_like(x_next).to(x_next.device)*torch.log(torch.tensor(1e-4))
        kld += kld_loss(x_mean[:,1:,:], x_log_var[:,1:,:], x_next, log_var_prior) + kld_loss_standard(x_mean[:,0,:], x_log_var[:,0,:])
        elbo = nll + kld
        loss = nll + self.beta*kld
        losses = {
            'kld': kld,
            'elbo': elbo,
            'loss': loss
        }
        return losses

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
        losses = self._compute_elbo(y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next)

        return y_pred, x_sample, z_sample, losses


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