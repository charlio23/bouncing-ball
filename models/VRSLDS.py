import torch
import torch.nn as nn

from models.modules import SequentialEncoder, MLP
from utils.sampling import gumbel_softmax, my_softmax
from utils.losses import kl_categorical, kl_categorical_uniform, kld_loss, kld_loss_standard, nll_gaussian, nll_gaussian_var_fixed

class VRSLDS(nn.Module):
    def __init__(self, obs_dim, discr_dim, cont_dim, hidden_dim, num_rec_layers, tau=0.5, 
                 bidirectional=True, beta=1, SB=False, posterior='first-order',
                 full_montecarlo=False):
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
        # Posterior factorised|first-order|recurrent|hierarchical
        self.posterior = posterior
        self.full_montecarlo = full_montecarlo

        # Network parameters
        self.encoder = SequentialEncoder(obs_dim, hidden_dim, discr_dim, cont_dim, num_rec_layers, bidirectional)

        dim_multiplier = 2 if self.bidirectional else 1
        if self.posterior=='factorised':
            self.out_discr = MLP(hidden_dim*dim_multiplier, hidden_dim, discr_dim)
            self.out_cont_mean = MLP(hidden_dim*dim_multiplier, hidden_dim, cont_dim)
            self.out_cont_log_var = MLP(hidden_dim*dim_multiplier, hidden_dim, cont_dim)
        elif self.posterior=='first-order':
            self.out_discr = MLP(hidden_dim*dim_multiplier + discr_dim + cont_dim, hidden_dim, discr_dim)
            self.out_cont = MLP(hidden_dim*dim_multiplier + discr_dim + cont_dim, hidden_dim, cont_dim*2)
        elif self.posterior=='recurrent':
            self.out_discr = nn.LSTMCell(hidden_dim*dim_multiplier + discr_dim + cont_dim, discr_dim)
            self.out_cont = nn.LSTMCell(hidden_dim*dim_multiplier + discr_dim + cont_dim, cont_dim*2)
        else:
            self.out_discr = MLP(hidden_dim*dim_multiplier, hidden_dim, discr_dim)
            self.out_cont_mean = MLP(hidden_dim*dim_multiplier, hidden_dim, cont_dim)
            self.out_cont_log_var = MLP(hidden_dim*dim_multiplier, hidden_dim, cont_dim)

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
        x = self.encoder(x)
        return x

    def _sample_discrete_states(self, z_distrib):
        return gumbel_softmax(z_distrib, tau=self.tau)

    def _sample_cont_states(self, x_mean, x_log_var):
        eps = torch.normal(mean=torch.zeros_like(x_mean)).to(x_mean.device)
        x_std = (x_log_var*0.5).exp()
        sample = x_mean + x_std*eps
        return sample
    
    def _sample_factorized(self, x):
        z_distrib = self.out_discr(x)
        x_mean = self.out_cont_mean(x)
        x_log_var = self.out_cont_log_var(x)
        # Sample from posterior
        z_sample = self._sample_discrete_states(z_distrib)
        x_sample = self._sample_cont_states(x_mean, x_log_var)
        return z_distrib, x_mean, x_log_var, z_sample, x_sample

    def _encode_and_sample(self, x):
        x = self._inference(x)
        if self.posterior=='factorised':
            z_distrib, x_mean, x_log_var, z_sample, x_sample = self._sample_factorized(x)
        elif self.posterior=='first-order':
            B, T, _ = x.size()
            z_sample = torch.zeros(B, T, self.discr_dim).to(x.device)
            z_distrib = torch.zeros_like(z_sample)
            x_sample = torch.zeros(B, T, self.cont_dim).to(x.device)
            x_mean = torch.zeros_like(x_sample)
            x_log_var = torch.zeros_like(x_sample)
            z_samp_i = torch.zeros(B,self.discr_dim).to(x.device)
            x_samp_i = torch.zeros(B,self.cont_dim).to(x.device)
            for t in range(T):
                embedding = x[:,t]
                z_input = torch.cat([embedding, z_samp_i, x_samp_i],dim=-1)
                z_distrib_i = self.out_discr(z_input)
                z_distrib[:,t,:] = z_distrib_i

                z_samp_i = self._sample_discrete_states(z_distrib_i)
                z_sample[:,t,:] = z_samp_i

                x_input = torch.cat([embedding, z_samp_i, x_samp_i], dim=-1)
                x_mean_i, x_log_var_i = self.out_cont(x_input).split(self.cont_dim, dim=-1)
                x_mean[:,t,:] = x_mean_i
                x_log_var[:,t,:] = x_log_var_i

                x_samp_i = self._sample_cont_states(x_mean_i, x_log_var_i)
                x_sample[:,t,:] = x_samp_i
        elif self.posterior=='recurrent':
            B, T, _ = x.size()
            h_prev_discr = torch.zeros((B, self.discr_dim)).to(x.device)
            c_prev_discr = torch.zeros((B, self.discr_dim)).to(x.device)
            h_prev_cont = torch.zeros((B, self.cont_dim*2)).to(x.device)
            c_prev_cont = torch.zeros((B, self.cont_dim*2)).to(x.device)
            z_sample = torch.zeros(B, T, self.discr_dim).to(x.device)
            z_distrib = torch.zeros_like(z_sample)
            x_sample = torch.zeros(B, T, self.cont_dim).to(x.device)
            x_mean = torch.zeros_like(x_sample)
            x_log_var = torch.zeros_like(x_sample)
            z_samp_i = torch.zeros(B,self.discr_dim).to(x.device)
            x_samp_i = torch.zeros(B,self.cont_dim).to(x.device)
            for t in range(T):
                embedding = x[:,t]
                z_input = torch.cat([embedding, z_samp_i, x_samp_i],dim=-1)
                h_prev_discr, c_prev_discr = self.out_discr(z_input, (h_prev_discr, c_prev_discr))
                z_distrib[:,t,:] = h_prev_discr

                z_samp_i = self._sample_discrete_states(h_prev_discr)
                z_sample[:,t,:] = z_samp_i

                x_input = torch.cat([embedding, z_samp_i, x_samp_i], dim=-1)
                h_prev_cont, c_prev_cont = self.out_cont(x_input, (h_prev_cont, c_prev_cont))
                x_mean_i, x_log_var_i = h_prev_cont.split(self.cont_dim, dim=-1)
                x_mean[:,t,:] = x_mean_i
                x_log_var[:,t,:] = x_log_var_i

                x_samp_i = self._sample_cont_states(x_mean_i, x_log_var_i)
                x_sample[:,t,:] = x_samp_i

        elif self.posterior=='hierarchical':
            B, T, _ = x.size()
            z_smooth = self.out_discr(x)
            x_mean_smooth = self.out_cont_mean(x)
            x_log_var_smooth = self.out_cont_log_var(x)
            z_sample = torch.zeros(B, T, self.discr_dim).to(x.device)
            z_distrib = torch.zeros_like(z_sample)
            x_sample = torch.zeros(B, T, self.cont_dim).to(x.device)
            x_mean = torch.zeros_like(x_sample)
            x_log_var = torch.zeros_like(x_sample)
            z_next = torch.zeros(B,self.discr_dim).to(x.device)
            x_next = torch.zeros(B,self.cont_dim).to(x.device)

            for t in range(T):

                z_distrib_i = z_next + z_smooth
                z_distrib[:,t,:] = z_distrib_i

                z_samp_i = self._sample_discrete_states(z_distrib_i)
                z_sample[:,t,:] = z_samp_i

                x_mean_i = x_next + x_mean_smooth
                x_log_var_i = 1e-4 + x_log_var_smooth
                x_mean[:,t,:] = x_mean_i
                x_log_var[:,t,:] = x_log_var_i

                x_samp_i = self._sample_cont_states(x_mean_i, x_log_var_i)
                x_sample[:,t,:] = x_samp_i

                _, z_next, x_next = self._decode(z_samp_i.unsqueeze(1), x_samp_i.unsqueeze(1))
                z_next.unsqueeze_(1)
                x_next.unsqueeze_(1)

        return z_distrib, x_mean, x_log_var, z_sample, x_sample

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
    
    def _compute_elbo(self, y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next, z_sample, x_sample):
        # Fixed variance
        # Reconstruction Loss p(y_t | x_t, z_t)
        nll = nll_gaussian_var_fixed(y_pred, input, variance=1e-4, add_const=False)

        if self.full_montecarlo:
            eps = 1e-16
            # log q(x) + log q(z) - log p(x_t | x_t-1, z_t) - log p(z_t| x_t-1, z_t-1)
            ## discrete kl
            kld = (z_sample[:,1:,:] * (torch.log(z_distr[:,1:,:] + eps) - torch.log(z_next + eps))).sum()/z_sample.size(0)
            ## continous kl
            kld += nll_gaussian_var_fixed(x_next, x_sample[:,1:,:], variance=1e-4, add_const=True) - nll_gaussian(x_mean[:,1:,:], x_log_var[:,1:,:], x_sample[:,1:,:])
        else:
            # Continous KL term p(z_1) and p(z_t | z_t-1, x_t-1)
            kld = kl_categorical(z_distr[:,1:,:], z_next)# + kl_categorical_uniform(z_distr[:,0,:], self.discr_dim)
            # Discrete KL term p(x_1) and p(x_t | z_t, x_t-1)
            log_var_prior = torch.ones_like(x_next).to(x_next.device)*torch.log(torch.tensor(1e-4))
            kld += kld_loss(x_mean[:,1:,:], x_log_var[:,1:,:], x_next, log_var_prior)# + kld_loss_standard(x_mean[:,0,:], x_log_var[:,0,:])
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
        z_distr, x_mean, x_log_var, z_sample, x_sample = self._encode_and_sample(input)
        # Convert logits to normalized distribution
        z_distr = my_softmax(z_distr, -1)
        # Forward step
        # Autoencode + 1-step computation
        y_pred, z_next, x_next = self._decode(z_sample, x_sample)
        losses = self._compute_elbo(y_pred, input, z_distr, z_next, x_mean, x_log_var, x_next, z_sample, x_sample)

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