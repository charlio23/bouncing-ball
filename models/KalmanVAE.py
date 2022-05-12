import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli

from models.modules import CNNFastDecoder, CNNFastEncoder, MLP
from utils.losses import nll_gaussian_var_fixed

class KalmanVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, obs_dim, latent_dim, num_modes, beta=1, alpha='mlp'):
        super(KalmanVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        # Beta VAE-like loss: Nll + b*KLD
        self.beta = beta
        self.alpha = alpha
        self.encoder = CNNFastEncoder(self.input_dim, self.obs_dim)
        self.decoder = CNNFastDecoder(self.obs_dim, self.input_dim)
        #self.decoder = CNNResidualDecoder(self.obs_dim, self.input_dim)
        if self.alpha=='mlp':
            self.parameter_net = MLP(self.obs_dim, 50, self.num_modes)
        else:
            self.parameter_net = nn.LSTM(self.obs_dim, 32, 
                                         2, batch_first=True)
            self.alpha_out = nn.Linear(32, self.num_modes)

        # Initial latent code a_0
        self.start_code = nn.Parameter(torch.zeros(self.obs_dim))
        self.state_dyn_net = None
        # Initial p(z_1) distribution
        self.mu_1 = (torch.zeros(self.latent_dim)).cuda()
        self.Sigma_1 = (20*torch.eye(self.latent_dim)).cuda()
        # Matrix modes
        self.A = nn.Parameter(torch.eye(self.latent_dim).unsqueeze(0).repeat(self.num_modes,1,1))
        self.C = nn.Parameter(torch.randn(self.num_modes, self.obs_dim, self.latent_dim)*0.5)

        self.Q = 0.8*torch.eye(self.latent_dim).cuda()
        self.R = 0.3*torch.eye(self.obs_dim).cuda()



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
        
        joint_obs = torch.cat([code.expand(B,-1,-1),obs[:,:-1,:]],dim=1)
        if self.alpha=='mlp':
            dyn_emb = self.parameter_net(joint_obs.reshape(B*T, -1))
        else:
            dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs).reshape(B*T,self.num_modes)
        inter_weight = dyn_emb.softmax(-1)
        A_t = torch.matmul(inter_weight, self.A.reshape(self.num_modes,-1)).reshape(B,T,self.latent_dim,self.latent_dim)
        C_t = torch.matmul(inter_weight, self.C.reshape(self.num_modes,-1)).reshape(B,T,self.obs_dim,self.latent_dim)
        
        return A_t, C_t

    def _filter_posterior(self, obs, A, C):
        # obs: (T ,B, N)
        # A: (T, D, D)
        # C: (T, N, D)
        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.latent_dim, 1).to(obs.device).double()
        Sigma_filt = torch.zeros(T, B, self.latent_dim, self.latent_dim).to(obs.device).double()
        obs = obs.unsqueeze(-1).double()
        mu_t = self.mu_1.expand(B,-1).unsqueeze(-1).double()
        Sigma_t = self.Sigma_1.expand(B,-1,-1).double()

        mu_pred = torch.zeros_like(mu_filt).double()
        Sigma_pred = torch.zeros_like(Sigma_filt).double()

        for t in range(T):

            mu_pred[t] = mu_t
            Sigma_pred[t] = Sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], Sigma_t), torch.transpose(C[:,t,:,:], 1,2))
            S_t += self.R.unsqueeze(0).double()

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            I_ = torch.eye(self.latent_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:,t,:,:])
            #Sigma_z = torch.matmul(I_, Sigma_t)
            Sigma_z = torch.matmul(torch.matmul(I_, Sigma_t), torch.transpose(I_, 1,2))
            Sigma_z += torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0).double()), torch.transpose(Kalman_gain, 1,2))

            mu_t = torch.matmul(A[:,t,:,:], mu_z)
            Sigma_t = torch.matmul(torch.matmul(A[:,t,:,:], Sigma_z), torch.transpose(A[:,t,:,:], 1,2))
            Sigma_t += self.Q.unsqueeze(0).double()

            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred)
    
    def _smooth_posterior(self, A, filtered, prediction):
        mu_filt, Sigma_filt = filtered
        mu_pred, Sigma_pred = prediction
        (T, *_) = mu_filt.size()
        mu_z_smooth = torch.zeros_like(mu_filt).double()
        Sigma_z_smooth = torch.zeros_like(Sigma_filt).double()

        mu_z_smooth[-1] = mu_filt[-1].double()
        Sigma_z_smooth[-1] = Sigma_filt[-1].double()

        for t in reversed(range(T-1)):
            J = torch.matmul(torch.matmul(Sigma_filt[t], torch.transpose(A[:,t+1,:,:], 1,2)), torch.inverse(Sigma_pred[t+1]))
            mu_diff = mu_z_smooth[t+1] - mu_pred[t+1]
            mu_z_smooth[t] = mu_filt[t] + torch.matmul(J, mu_diff)

            cov_diff = Sigma_z_smooth[t+1] - Sigma_pred[t+1]
            Sigma_z_smooth[t] = Sigma_filt[t] + torch.matmul(torch.matmul(J, cov_diff), torch.transpose(J, 1, 2))
        
        return mu_z_smooth.float(), Sigma_z_smooth.float()

    def _kalman_posterior(self, obs, filter_only=False):
        # obs: (T ,B, N)
        A, C = self._interpolate_matrices(obs)
        filtered, pred = self._filter_posterior(obs.transpose(0,1), A.double(), C.double())
        if filter_only:
            return (filtered[0].float(), filtered[1].float()), A, C
        smoothed = self._smooth_posterior(A.double(), filtered, pred)
        return smoothed, A, C

    def _decode(self, z):
        x = self.decoder(z)
        return x

    def _decode_latent(self, z_sample, A, C):
        (T, B, *_) = z_sample.size()
        z_next = torch.matmul(A.transpose(0,1), z_sample.unsqueeze(-1)).squeeze(-1)
        a_next = torch.matmul(C.transpose(0,1), z_sample.unsqueeze(-1)).squeeze(-1)
        return a_next, z_next

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def _compute_elbo(self, x, x_hat, a_mu, a_log_var, a_sample, smoothed, A, C):

        # max: ELBO = log p(x_t|a_t) - (log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1))
        # min: -ELBO =  - log p(x_t|a_t) + log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)
        # Fixed variance
        # Reconstruction Loss p(x_t | a_t)
        decoder_x = Bernoulli(x_hat)

        nll = -decoder_x.log_prob(x).mean(dim=0).sum()
        ## KL terms
        smoothed_mean, smoothed_cov = smoothed
        (T, B, *_) = smoothed_cov.size()
        eps = 1e-2*torch.eye(self.latent_dim).to(x.device).reshape(1,1,self.latent_dim,self.latent_dim).repeat(T, B, 1, 1).double()
        try:
            smoothed_z = MultivariateNormal(smoothed_mean.squeeze(-1), scale_tril=torch.linalg.cholesky(smoothed_cov.double() + eps).float())
        except:
            print("Smooth distrib error!")
            smoothed_z = MultivariateNormal(smoothed_mean.squeeze(-1), scale_tril=torch.linalg.cholesky(eps).float())

        z_sample = smoothed_z.sample()
        a_pred, z_next = self._decode_latent(z_sample, A, C)
        decoder_z = MultivariateNormal(torch.zeros(self.latent_dim).to(x.device), scale_tril=torch.linalg.cholesky(self.Q))
        decoder_z_0 = MultivariateNormal(self.mu_1, scale_tril=torch.tril(self.Sigma_1))
        decoder_a = MultivariateNormal(torch.zeros(self.obs_dim).to(x.device), scale_tril=torch.linalg.cholesky(self.R))
        q_a = MultivariateNormal(a_mu, torch.diag_embed(torch.exp(a_log_var)))
        # -log p(z_t| z_t-1)
        kld = -decoder_z.log_prob((z_sample[1:] - z_next[:-1])).mean(dim=1).sum()
        kld -= decoder_z_0.log_prob(z_sample[0]).mean(dim=0)
        # -log p(a_t| z_t)
        kld -= decoder_a.log_prob((a_sample - a_pred)).mean(dim=1).sum()
        # log q(a)
        kld += q_a.log_prob(a_sample.transpose(0,1)).mean(dim=0).sum()
        # log q(z)
        kld += smoothed_z.log_prob(z_sample).mean(dim=1).sum()

        elbo = kld + nll
        loss = self.beta*kld + nll*0.3
        losses = {
            'kld': kld,
            'elbo': elbo,
            'loss': loss,
            'nll': nll
        }
        return z_sample, losses

    def forward(self, x, variational=True):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,C,H,W) = x.size()
        # q(a_t|x_t)
        a_sample, a_mu, a_log_var = self._encode_obs(x.reshape(B*T,C,H,W), variational)
        a_sample = a_sample.reshape(B,T,-1)
        a_mu = a_mu.reshape(B,T,-1)
        a_log_var = a_log_var.reshape(B,T,-1)
        # q(z|a)
        smoothed, A_t, C_t = self._kalman_posterior(a_sample)
        # p(x_t|a_t)
        x_hat = self._decode(a_sample.reshape(B*T,-1)).reshape(B,T,C,H,W)
        # ELBO
        z_sample, losses = self._compute_elbo(x, x_hat, a_mu, a_log_var, a_sample.transpose(0,1), smoothed, A_t, C_t)
        return x_hat, a_sample, z_sample, losses

    def predict_sequence(self, input, seq_len=None):
        (B,T,C,H,W) = input.size()
        if seq_len is None:
            seq_len = T
        a_sample, _, _ = self._encode_obs(input.reshape(B*T,C,H,W))
        a_sample = a_sample.reshape(B,T,-1)
        filt, A_t, C_t = self._kalman_posterior(a_sample, filter_only=True)
        filt_mean, filt_cov = filt
        eps = 1e-6*torch.eye(self.latent_dim).to(input.device).reshape(1,self.latent_dim,self.latent_dim).repeat(B, 1, 1)
        filt_z = MultivariateNormal(filt_mean[-1].squeeze(-1), scale_tril=torch.linalg.cholesky(filt_cov[-1] + eps))
        z_sample = filt_z.sample()
        _shape = [a_sample.size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        obs_seq = torch.zeros(_shape).to(input.device)
        _shape = [z_sample.unsqueeze(1).size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        latent_seq = torch.zeros(_shape).to(input.device)
        latent_prev = z_sample
        obs_prev = a_sample[:,-1]
        for t in range(seq_len):
            # Compute alpha from a_0:t-1
            if self.alpha=='mlp':
                dyn_emb = self.parameter_net(obs_prev)
            else:
                alpha_, cell_state = self.state_dyn_net
                dyn_emb, self.state_dyn_net = self.parameter_net(obs_prev.unsqueeze(1), (alpha_, cell_state))
            inter_weight = dyn_emb.softmax(-1).squeeze(1)
            ## Compute A_t, C_t
            A_t = torch.matmul(inter_weight, self.A.reshape(self.num_modes,-1)).reshape(B,self.latent_dim,self.latent_dim)
            C_t = torch.matmul(inter_weight, self.C.reshape(self.num_modes,-1)).reshape(B,self.obs_dim,self.latent_dim)

            # Calculate new z_t
            ## Update z_t
            latent_prev = torch.matmul(A_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            latent_seq[:,t] = latent_prev
            # Calculate new a_t
            obs_prev = torch.matmul(C_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            obs_seq[:,t] = obs_prev

        image_seq = self._decode(obs_seq.reshape(B*seq_len,-1)).reshape(B,seq_len,C,H,W)

        return image_seq, obs_seq, latent_seq

if __name__=="__main__":
    # Trial run
    net = KalmanVAE(input_dim=1, hidden_dim=128, obs_dim=2, latent_dim=4, num_modes=3)
    from torch.autograd import Variable
    sample = Variable(torch.rand((6,10,1,32,32)), requires_grad=True)
    torch.autograd.set_detect_anomaly(True)
    image_seq, obs_seq, latent_seq = net.predict_sequence(sample, 10)
    x_hat, a_mu, a_log_var, losses = net(sample)
    loss = losses['elbo']
    loss.backward()
    print(x_hat.size())
    print(a_mu.size())
    print(a_log_var.size())