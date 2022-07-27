import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli

from models.modules import CNNFastDecoder, CNNFastEncoder, MLPJacobian

class ExtendedKalmanVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, obs_dim, latent_dim, beta=1):
        super(ExtendedKalmanVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        # Beta VAE-like loss: Nll + b*KLD
        self.beta = beta
        self.encoder = CNNFastEncoder(self.input_dim, self.obs_dim)
        self.decoder = CNNFastDecoder(self.obs_dim, self.input_dim)
        #self.decoder = CNNResidualDecoder(self.obs_dim, self.input_dim)
        self.transition = MLPJacobian(self.latent_dim, self.hidden_dim, self.latent_dim)
        self.emission = MLPJacobian(self.latent_dim, self.hidden_dim, self.obs_dim)
        self.state_dyn_net = None
        # Initial p(z_1) distribution
        self.mu_1 = (torch.zeros(self.latent_dim)).cuda()
        self.Sigma_1 = (20*torch.eye(self.latent_dim)).cuda()

        self.Q = 0.08*torch.eye(self.latent_dim).cuda()
        self.R = 0.03*torch.eye(self.obs_dim).cuda()

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _encode_obs(self, x, variational=True):
        (z_mu, z_log_var) = self.encoder(x)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = (z_log_var*0.5).exp()
        sample = z_mu
        if variational:
            sample += z_std*eps

        return sample, z_mu, z_log_var

    def _filter_posterior(self, obs):
        # obs: (T ,B, N)
        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.latent_dim, 1).to(obs.device)
        Sigma_filt = torch.zeros(T, B, self.latent_dim, self.latent_dim).to(obs.device)
        obs = obs.unsqueeze(-1)
        mu_t = self.mu_1.expand(B,-1)
        Sigma_t = self.Sigma_1.expand(B,-1,-1)

        mu_pred = torch.zeros_like(mu_filt)
        Sigma_pred = torch.zeros_like(Sigma_filt)

        A_t = torch.zeros(B,T,self.latent_dim,self.latent_dim).to(obs.device)

        for t in range(T):
            
            # mu/sigma: t | t-1
            mu_pred[t] = mu_t.unsqueeze(-1)
            Sigma_pred[t] = Sigma_t

            y_pred, H_t = self.emission.jacobian(mu_t)
            r = obs[t] - y_pred.unsqueeze(-1)
            S_t = torch.matmul(torch.matmul(H_t, Sigma_t), torch.transpose(H_t, 1,2))
            S_t += self.R.unsqueeze(0)

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(H_t, 1,2)), torch.inverse(S_t))       
            # filter: t | t
            mu_z = mu_t.unsqueeze(-1) + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(self.latent_dim).to(obs.device) - torch.matmul(Kalman_gain, H_t)
            #Sigma_z = torch.matmul(I_, Sigma_t)
            Sigma_z = torch.matmul(torch.matmul(I_, Sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            #Sigma_z = (Sigma_z + Sigma_z.transpose(1,2))/2
            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z
            if t != T-1:
                # mu/sigma: t+1 | t for next step
                mu_t, G_t = self.transition.jacobian(mu_z.squeeze(-1))
                Sigma_t = torch.matmul(torch.matmul(G_t, Sigma_z), torch.transpose(G_t, 1,2))
                Sigma_t += self.Q.unsqueeze(0)
                A_t[:,t+1,:,:] = G_t

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred), A_t

    def _smooth_posterior(self, A, filtered, prediction):
        mu_filt, Sigma_filt = filtered
        mu_pred, Sigma_pred = prediction
        (T, *_) = mu_filt.size()
        mu_z_smooth = torch.zeros_like(mu_filt)
        Sigma_z_smooth = torch.zeros_like(Sigma_filt)
        mu_z_smooth[-1] = mu_filt[-1]
        Sigma_z_smooth[-1] = Sigma_filt[-1]
        for t in reversed(range(T-1)):
            J = torch.matmul(Sigma_filt[t], torch.matmul(torch.transpose(A[:,t+1,:,:], 1,2), torch.inverse(Sigma_pred[t+1])))
            mu_diff = mu_z_smooth[t+1] - mu_pred[t+1]
            mu_z_smooth[t] = mu_filt[t] + torch.matmul(J, mu_diff)

            cov_diff = Sigma_z_smooth[t+1] - Sigma_pred[t+1]
            Sigma_z_smooth[t] = Sigma_filt[t] + torch.matmul(torch.matmul(J, cov_diff), torch.transpose(J, 1, 2))
        
        return mu_z_smooth, Sigma_z_smooth

    def _kalman_posterior(self, obs, filter_only=False):
        # obs: (B ,T, N)
        filtered, pred, A = self._filter_posterior(obs.transpose(0,1))
        if filter_only:
            return filtered
        smoothed = self._smooth_posterior(A, filtered, pred)
        
        return smoothed

    def _decode(self, z):
        x = self.decoder(z)
        return x

    def _decode_latent(self, z_sample):
        (T, B, *_) = z_sample.size()
        z_next = self.transition(z_sample.reshape(T*B,-1)).reshape(T,B,-1)
        a_next = self.emission(z_sample.reshape(T*B,-1)).reshape(T,B,-1)
        return a_next, z_next

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def _compute_elbo(self, x, x_hat, a_mu, a_log_var, a_sample, smoothed):
        (B, T, ch, H, W) = x.size()
        # max: ELBO = log p(x_t|a_t) - (log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1))
        # min: -ELBO =  - log p(x_t|a_t) + log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)
        # Fixed variance
        # Reconstruction Loss p(x_t | a_t)
        decoder_x = Bernoulli(x_hat)
        p_x = (decoder_x.log_prob(x)).reshape(B,T,-1).sum(-1)
        nll = -(p_x).mean(dim=0).sum()
        ## KL terms
        smoothed_mean, smoothed_cov = smoothed
        smoothed_z = MultivariateNormal(smoothed_mean.squeeze(-1), 
                                        scale_tril=torch.linalg.cholesky(smoothed_cov))
        z_sample = smoothed_z.sample()
        decoder_z = MultivariateNormal(torch.zeros(self.latent_dim).to(x.device), scale_tril=torch.linalg.cholesky(self.Q))
        decoder_z_0 = MultivariateNormal(self.mu_1, scale_tril=torch.linalg.cholesky(self.Sigma_1))
        decoder_a = MultivariateNormal(torch.zeros(self.obs_dim).to(x.device), scale_tril=torch.linalg.cholesky(self.R))
        q_a = MultivariateNormal(a_mu, torch.diag_embed(torch.exp(a_log_var)))
        a_pred, z_next = self._decode_latent(z_sample)
        # -log p(z_t| z_t-1)
        kld = - decoder_z_0.log_prob(z_sample[0]).mean(dim=0)
        kld -= decoder_z.log_prob((z_sample[1:] - z_next[:-1])).mean(dim=1).sum()
        # -log p(a_t| z_t)
        kld -= (decoder_a.log_prob((a_sample - a_pred)).transpose(0,1)).mean(dim=1).sum()
        # log q(z)
        kld += smoothed_z.log_prob(z_sample).mean(dim=1).sum()
        # log q(a)
        kld += (q_a.log_prob(a_sample.transpose(0,1))).mean(dim=0).sum()
        
        elbo = kld + nll
        loss = kld + self.beta*nll
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
        smoothed = self._kalman_posterior(a_sample)
        # p(x_t|a_t)
        x_hat = self._decode(a_sample.reshape(B*T,-1)).reshape(B,T,C,H,W)
        # ELBO
        z_sample, losses = self._compute_elbo(x, x_hat, a_mu, a_log_var, a_sample.transpose(0,1), smoothed)
        return x_hat, a_sample, z_sample, losses

    def predict_sequence(self, input, seq_len=None):
        (B,T,C,H,W) = input.size()
        if seq_len is None:
            seq_len = T
        a_sample, _, _ = self._encode_obs(input.reshape(B*T,C,H,W))
        a_sample = a_sample.reshape(B,T,-1)
        filt = self._kalman_posterior(a_sample, filter_only=True)
        filt_mean, filt_cov = filt
        filt_z = MultivariateNormal(filt_mean[-1].squeeze(-1), scale_tril=torch.linalg.cholesky(filt_cov[-1]))
        z_sample = filt_z.sample()
        _shape = [a_sample.size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        obs_seq = torch.zeros(_shape).to(input.device)
        _shape = [z_sample.unsqueeze(1).size(i) if i!=1 else seq_len for i in range(len(a_sample.size()))]
        latent_seq = torch.zeros(_shape).to(input.device)
        latent_prev = z_sample
        obs_prev = a_sample[:,-1]
        for t in range(seq_len):
            # Calculate new z_t
            ## Update z_t
            latent_prev = self.transition(latent_prev)
            latent_seq[:,t] = latent_prev
            # Calculate new a_t
            obs_prev = self.emission(latent_prev)
            obs_seq[:,t] = obs_prev

        image_seq = self._decode(obs_seq.reshape(B*seq_len,-1)).reshape(B,seq_len,C,H,W)

        return image_seq, obs_seq, latent_seq

    def test_log_likeli(self, x, target=None, mask_frames=None, L=100):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,C,H,W) = x.size()
        loglikeli = torch.zeros(B).to(x.device)
        p_x = torch.zeros_like(x)
        for l in range(L):
            # q(a_t|x_t)
            a_sample, _, _ = self._encode_obs(x.reshape(B*T,C,H,W))
            a_sample = a_sample.reshape(B,T,-1)


            if mask_frames is not None:
                # q(z|a)
                smoothed, _, C_t = self._kalman_posterior(a_sample, mask_frames)
                smoothed_mean, smoothed_cov = smoothed
                smoothed_z = MultivariateNormal(smoothed_mean.squeeze(-1), 
                                        scale_tril=torch.linalg.cholesky(smoothed_cov))       
                z_sample = smoothed_z.sample()
                a_pred = torch.matmul(C_t.transpose(0,1), z_sample.unsqueeze(-1)).squeeze(-1).transpose(0,1)
                a_sample = (1 - mask_frames.unsqueeze(-1))*a_pred + mask_frames.unsqueeze(-1)*a_sample
                
            x_hat = self._decode(a_sample.reshape(B*T,-1)).reshape(B,T,C,H,W)
            # ELBO
            decoder_x = Bernoulli(x_hat)
            p_x += (decoder_x.log_prob(target)).exp()
        loglikeli = ((1/L)*p_x).log().reshape(B,-1).sum(-1)
        
        return loglikeli

if __name__=="__main__":
    # Trial run
    net = ExtendedKalmanVAE(input_dim=1, hidden_dim=32, obs_dim=2, latent_dim=4).cuda().float()
    from torch.autograd import Variable
    sample = Variable((torch.rand((6,10,1,32,32)) > 0.5).float(), requires_grad=True).cuda()
    torch.autograd.set_detect_anomaly(True)
    x_hat, a_mu, a_log_var, losses = net(sample)
    loss = losses['elbo']
    loss.backward()
    print(x_hat.size())
    print(a_mu.size())
    print(a_log_var.size())