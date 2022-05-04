import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal

from modules import MLP, CNNEncoder, CNNResidualDecoder
#from utils.losses import nll_gaussian_var_fixed, nll_gaussian

class KalmanVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, obs_dim, latent_dim, num_modes):
        super(KalmanVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        self.encoder = CNNEncoder(self.input_dim, self.obs_dim, 4)
        self.decoder = CNNResidualDecoder(self.obs_dim, 1)

        self.parameter_net = nn.LSTM(self.obs_dim, self.num_modes, 
                                    1, batch_first=True)

        # Initial latent code a_0
        self.start_code = nn.Parameter(torch.randn(self.obs_dim))
        # Initial p(z_1) distribution
        self.mu_1 = nn.Parameter(torch.zeros(self.latent_dim))
        self.Sigma_1 = nn.Parameter(torch.eye(self.latent_dim))
        # Matrix modes
        self.A = nn.Parameter(torch.randn(self.num_modes, self.latent_dim, self.latent_dim))
        self.C = nn.Parameter(torch.randn(self.num_modes, self.obs_dim, self.latent_dim))

        self.Q = nn.Parameter(torch.eye(self.latent_dim))
        self.R = nn.Parameter(torch.eye(self.obs_dim))



        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _encode_obs(self, x, variational=True):
        (z_mu, z_log_var) = self.encoder(x)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = torch.minimum((z_log_var*0.5).exp(), torch.FloatTensor([100.]).to(x.device))
        sample = z_mu
        if variational:
            sample += z_std*eps
        return sample, z_mu, z_log_var

    def _interpolate_matrices(self, obs):
        # obs: (B ,T, N)
        (B, T, _) = obs.size()
        code = self.start_code.reshape(1,1,-1)
        
        joint_obs = torch.cat([code.expand(B,-1,-1),obs],dim=1)
        dyn_emb, _ = self.parameter_net(joint_obs)
        inter_weight = dyn_emb[:,:-1,:].softmax(-1).reshape(B*T,self.num_modes)
        print(inter_weight.size())
        print(self.A.size())
        A_t = torch.matmul(inter_weight, self.A.reshape(self.num_modes,-1)).reshape(B,T,self.latent_dim,self.latent_dim)
        C_t = torch.matmul(inter_weight, self.C.reshape(self.num_modes,-1)).reshape(B,T,self.obs_dim,self.latent_dim)
        
        print(A_t.size(), C_t.size())
        return A_t, C_t

    def _filter_posterior(self, obs, A, C):
        # obs: (T ,B, N)
        # A: (T, D, D)
        # C: (T, N, D)
        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.latent_dim, 1).to(obs.device)
        Sigma_filt = torch.zeros(T, B, self.latent_dim, self.latent_dim).to(obs.device)
        obs = obs.unsqueeze(-1)
        mu_t = self.mu_1.expand(B,-1).unsqueeze(-1)
        Sigma_t = self.Sigma_1.expand(B,-1,-1)

        mu_pred = torch.zeros_like(mu_filt)
        Sigma_pred = torch.zeros_like(Sigma_filt)

        for t in range(T):

            mu_pred[t] = mu_t
            Sigma_pred[t] = Sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], Sigma_t), torch.transpose(C[:,t,:,:], 1,2))
            S_t += self.R

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            I = torch.eye(self.latent_dim).to(obs.device)
            Sigma_z = torch.matmul((I - torch.matmul(Kalman_gain, C[:,t,:,:])), Sigma_t)

            mu_t = torch.matmul(A[:,t,:,:], mu_z)
            Sigma_t = torch.matmul(torch.matmul(A[:,t,:,:], Sigma_z), torch.transpose(A[:,t,:,:], 1,2))
            Sigma_t += self.Q

            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred)
    
    def _smooth_posterior(self, A, filtered, prediction):
        mu_filt, Sigma_filt = filtered
        mu_pred, Sigma_pred = prediction
        (T, *_) = mu_filt.size()
        mu_z_smooth = torch.zeros_like(mu_filt)
        Sigma_z_smooth = torch.zeros_like(Sigma_filt)

        mu_z_smooth[-1] = mu_filt[-1]
        Sigma_z_smooth[-1] = Sigma_filt[-1]

        for t in reversed(range(T-1)):
            J = torch.matmul(torch.matmul(Sigma_filt[t], torch.transpose(A[:,t,:,:], 1,2)), torch.inverse(Sigma_pred[t]))
            mu_diff = mu_z_smooth[t+1] - mu_pred[t+1]
            mu_z_smooth[t] = mu_filt[t] + torch.matmul(J, mu_diff)

            cov_diff = Sigma_z_smooth[t+1] - Sigma_pred[t+1]
            Sigma_z_smooth[t] = Sigma_filt[t] + torch.matmul(torch.matmul(J, cov_diff), torch.transpose(J, 1, 2))
        
        return mu_z_smooth, Sigma_z_smooth

    def _kalman_posterior(self, obs):
        # obs: (T ,B, N)
        A, C = self._interpolate_matrices(obs)
        filtered, pred = self._filter_posterior(obs.transpose(0,1), A, C)
        smoothed = self._smooth_posterior(A, filtered, pred)
        return smoothed, A, C

    def _decode(self, z):
        x = self.decoder(z)
        return x

    def _decode_latent(self, z_sample, A, C):
        (T, B, *_) = z_sample.size()
        z_next = torch.matmul(A, z_sample.unsqueeze(-1)).squeeze(-1)
        a_next = torch.matmul(C, z_sample.unsqueeze(-1)).squeeze(-1)
        return a_next, z_next

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def _compute_elbo(self, x, x_hat, a_mu, a_log_var, a_sample, smoothed, A, C):
        # Fixed variance
        # Reconstruction Loss p(x_t | a_t)
        #nll = nll_gaussian_var_fixed(x_hat, input, variance=1e-4, add_const=False)
        # log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)
        ## KL terms
        smoothed_mean, smoothed_cov = smoothed
        smoothed_z = MultivariateNormal(smoothed_mean.squeeze(-1), scale_tril=torch.linalg.cholesky(smoothed_cov))
        z_sample = smoothed_z.sample()
        a_pred, z_next = self._decode_latent(z_sample, A, C)
        decoder_z = MultivariateNormal(torch.zeros(self.latent_dim), self.Q)
        decoder_z_0 = MultivariateNormal(self.mu_1, self.Sigma_1)
        decoder_a = MultivariateNormal(torch.zeros(self.obs_dim), self.R)
        #log p(z_t| z_t-1)
        kld = decoder_z.log_prob((z_sample[1:] - z_next[:-1])).mean(dim=1).sum()
        kld += decoder_z_0.log_prob(z_sample[0]).mean(dim=0)
        #log p(a_t| z_t)
        kld += decoder_a.log_prob((a_sample - a_pred)).mean(dim=1).sum()
        #log q(a_t| z_t)
        #kld -= nll_gaussian(a_mu, a_log_var, a_sample)
        kld -= smoothed_z.log_prob(z_sample).mean(dim=1).sum()

        elbo = nll + kld
        losses = {
            'kld': kld,
            'elbo': elbo,
        }
        return z_sample, losses

    def forward(self, x, variational=True):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,C,H,W) = x.size()
        x = x.reshape(B*T,C,H,W)
        a_sample, a_mu, a_log_var = self._encode_obs(x, variational)
        a_sample = a_sample.reshape(B,T,-1)
        smoothed, A, C = self._kalman_posterior(a_sample)
        x_hat = self._decode(a_sample.reshape(B*T,-1))

        z_sample, losses = self._compute_elbo(x, x_hat, a_mu, a_log_var, a_sample, smoothed, A, C)
        return x_hat, a_sample, z_sample, losses

if __name__=="__main__":
    # Trial run
    net = KalmanVAE(input_dim=1, hidden_dim=128, obs_dim=2, latent_dim=4, num_modes=3)

    sample = torch.rand((1,1,1,32,32))
    torch.autograd.set_detect_anomaly(True)
    x_hat, a_mu, a_log_var = net(sample)
    print(x_hat.size())
    print(a_mu.size())
    print(a_log_var.size())