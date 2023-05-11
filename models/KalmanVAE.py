import numpy as np
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Bernoulli

from models.modules import CNNFastDecoder, CNNFastEncoder, MLP, CNNFastDecoderKVAE
from glow_pytorch.model import Glow

class KalmanVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, obs_dim, latent_dim, num_modes, beta=1, alpha='mlp', mode='base', device='cuda', gamma=1):
        super(KalmanVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.num_modes = num_modes
        # Beta VAE-like loss: Nll + b*KLD
        self.beta = beta
        self.alpha = alpha
        self.mode = mode
        self.device = device
        self.encoder = None
        self.decoder = None
        self.flow_model = None
        self.gamma = gamma
        if self.mode=='base':
            #self.encoder = CNNFastEncoder(self.input_dim, self.obs_dim)
            #self.decoder = CNNFastDecoder(self.obs_dim, self.input_dim)
            self.encoder = MLP(self.input_dim, self.hidden_dim, 2*self.latent_dim)
            self.decoder = MLP(self.latent_dim, self.hidden_dim, self.input_dim)
        else:
            self.flow_model = Glow(1, 32, 4)
            self.split_sizes = None
            self.recalc_split = False
        if self.mode=='greparam':
            #self.C1 = nn.Parameter(torch.randn(self.num_modes, 6, 6)*0.05)
            #self.C2 = nn.Parameter(torch.randn(self.num_modes, 4, 4)*0.05)
            #self.C3 = nn.Parameter(torch.randn(self.num_modes, 4, 4)*0.05)
            self.C1 = nn.Parameter(torch.randn(4,16, 3, 3)*0.05)
            self.C2 = nn.Parameter(torch.randn(16,16, 3, 3)*0.05)
            self.C3 = nn.Parameter(torch.randn(16,16, 3, 3)*0.05)
            self.C4 = nn.Parameter(torch.randn(16,1, 4, 4)*0.05)

            self.C1_indices = self._get_indices_sparse_kernel_matrix(self.C1, 3, 3)
            self.C2_indices = self._get_indices_sparse_kernel_matrix(self.C2, 7, 7)
            self.C3_indices = self._get_indices_sparse_kernel_matrix(self.C3, 15, 15)
            self.C4_indices = self._get_indices_sparse_kernel_matrix(self.C4, 32, 32)
        else:
            self.C = nn.Parameter(torch.randn(self.num_modes, self.obs_dim, self.latent_dim)*0.05)
        #self.decoder = CNNResidualDecoder(self.obs_dim, self.input_dim)
        if self.alpha=='mlp':
            self.parameter_net = MLP(self.obs_dim, self.hidden_dim, self.num_modes)
        else:
            self.parameter_net = nn.LSTM(self.obs_dim, self.hidden_dim, 
                                         2, batch_first=True)
            self.alpha_out = nn.Linear(self.hidden_dim, self.num_modes)

        # Initial latent code a_0
        self.start_code = nn.Parameter(torch.zeros(self.obs_dim))
        self.state_dyn_net = None
        # Initial p(z_1) distribution
        self.mu_1 = (torch.zeros(self.latent_dim)).to(self.device).float()
        self.Sigma_1 = (torch.eye(self.latent_dim)).to(self.device).float()
        # Matrix modes
        self.A = nn.Parameter(torch.eye(self.latent_dim).unsqueeze(0).repeat(self.num_modes,1,1))
        #self.C = nn.Parameter(torch.randn(self.num_modes, self.obs_dim, self.latent_dim)*0.05)

        self.Q = 0.05*torch.eye(self.latent_dim).to(self.device).float()
        self.R = 0.01*torch.eye(self.obs_dim).to(self.device).float()



        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _encode_obs(self, x, variational=True):
        (z_mu, z_log_var) = self.encoder(x).split(self.latent_dim, dim=-1)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = (z_log_var*0.5).exp()
        sample = z_mu
        if variational:
            sample = sample + z_std*eps

        return sample, z_mu, z_log_var

    def _compute_alpha(self, obs, pred, mask):
        B, *_ = obs.size()
        joint_obs = (1-mask)*pred + mask*obs

        if self.alpha=='mlp':
            dyn_emb = self.parameter_net(joint_obs)
        else:
            dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs.unsqueeze(1), (self.state_dyn_net))
            dyn_emb = self.alpha_out(dyn_emb)
        inter_weight = dyn_emb.softmax(-1)

        return inter_weight
    
    def _compute_mixture(self, weight):
        B, *_ = weight.size()
        A_t = torch.matmul(weight, self.A.reshape(self.num_modes,-1)).reshape(B,self.latent_dim,self.latent_dim)
        C_t = torch.matmul(weight, self.C.reshape(self.num_modes,-1)).reshape(B,self.obs_dim,self.latent_dim)
        
        return A_t, C_t

    def _get_indices_sparse_kernel_matrix(self, K, h_X, w_X, s=2):

        # Assuming no channels and stride == 1.
        # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
        # This is a little bit brain-twisting.

        _, _, h_K, w_K = K.size()
        h_Y, w_Y = (h_X - h_K)//s + 1, (w_X - w_K)//s + 1
        index_matrix = []
        index_kernel = []
        for i in range(h_Y):
            for j in range(w_Y):
                for ii in range(h_K):
                    for jj in range(w_K):
                        index_matrix.append( (i * w_Y + j)*h_X*w_X +  s*i * w_X + s*j + ii * w_X + jj)
                        index_kernel.append(ii*w_K + jj)
        return index_matrix, index_kernel
    
    def _get_deconv_from_indices(self, index_mat, index_kernel, K, h_Y, h_X):
            w_Y = h_Y
            w_X = h_X
            n_in, n_out, _, _ = K.size()
            M = torch.zeros((n_in, n_out, h_Y * w_Y*h_X * w_X)).to(K.device)
            M[:,:,index_mat] = K.reshape(n_in,n_out,-1)[:,:,index_kernel]
            M = M.reshape((n_in, n_out, h_Y * w_Y, h_X * w_X)).permute(2,0,3,1).reshape(h_Y * w_Y*n_in, h_X * w_X*n_out).T
            return M

    def _interpolate_matrices(self, obs):
        # obs: (B ,T, N)
        (B, T, _) = obs.size()
        code = self.start_code.reshape(1,1,-1)
        
        joint_obs = torch.cat([code.expand(B,-1,-1),obs[:,:-1,:]],dim=1)
        if self.alpha=='mlp':
            dyn_emb = self.parameter_net(joint_obs.reshape(B*T, -1))
        else:
            dyn_emb, self.state_dyn_net = self.parameter_net(joint_obs)
            dyn_emb = self.alpha_out(dyn_emb.reshape(B*T,-1))
        inter_weight = (dyn_emb/self.gamma).softmax(-1)
        A_t = torch.matmul(inter_weight, self.A.reshape(self.num_modes,-1)).reshape(B,T,self.latent_dim,self.latent_dim)
        if self.mode != 'greparam':
            C_t = torch.matmul(inter_weight, self.C.reshape(self.num_modes,-1)).reshape(B,T,self.obs_dim,self.latent_dim)
        else:
            C1 = self._get_deconv_from_indices(self.C1_indices[0], self.C1_indices[1], self.C1, 1, 3)
            C2 = self._get_deconv_from_indices(self.C2_indices[0], self.C2_indices[1], self.C2, 3, 7)
            C3 = self._get_deconv_from_indices(self.C3_indices[0], self.C3_indices[1], self.C3, 7, 15)
            C4 = self._get_deconv_from_indices(self.C4_indices[0], self.C4_indices[1], self.C4, 15, 32)
            C_t = torch.matmul(C4, torch.matmul(C3, torch.matmul(C2, C1))).unsqueeze(0).unsqueeze(0).expand(B,T,-1,-1)
            #C1 = self._get_sparse_kernel_matrix(self.C1, 32, 32).transpose(1,2)
            #C2 = self._get_sparse_kernel_matrix(self.C2, 14, 14, s=2).transpose(1,2)
            #C3 = self._get_sparse_kernel_matrix(self.C3, 6, 6, s=2).transpose(1,2)
            #C = torch.matmul(C1,C2)#,C3)
            #C = torch.matmul(torch.matmul(C1,C2),C3)
            #C_t =torch.matmul(inter_weight, C.reshape(self.num_modes,-1)).reshape(B,T,self.obs_dim,self.latent_dim)

        return A_t, C_t, inter_weight

    def _filter_posterior(self, obs, A, C):
        # obs: (T ,B, N)
        # A: (T, D, D)
        # C: (T, N, D)
        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.latent_dim, 1).to(obs.device).float()
        Sigma_filt = torch.zeros(T, B, self.latent_dim, self.latent_dim).to(obs.device).float()
        obs = obs.unsqueeze(-1)
        mu_t = self.mu_1.expand(B,-1).unsqueeze(-1)
        Sigma_t = self.Sigma_1.expand(B,-1,-1)

        mu_pred = torch.zeros_like(mu_filt)
        Sigma_pred = torch.zeros_like(Sigma_filt)

        for t in range(T):
            
            # mu/sigma: t | t-1
            mu_pred[t] = mu_t
            Sigma_pred[t] = Sigma_t

            y_pred = torch.matmul(C[:,t,:,:], mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], Sigma_t), torch.transpose(C[:,t,:,:], 1,2))
            S_t += self.R.unsqueeze(0)

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(C[:,t,:,:], 1,2)), torch.inverse(S_t))       
            # filter: t | t
            mu_z = mu_t + torch.matmul(Kalman_gain, r)
            
            I_ = torch.eye(self.latent_dim).to(obs.device) - torch.matmul(Kalman_gain, C[:,t,:,:])
            #Sigma_z = torch.matmul(I_, Sigma_t)
            Sigma_z = torch.matmul(torch.matmul(I_, Sigma_t), torch.transpose(I_, 1,2)) + torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            #Sigma_z = (Sigma_z + Sigma_z.transpose(1,2))/2
            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z
            if t != T-1:
                # mu/sigma: t+1 | t for next step
                mu_t = torch.matmul(A[:,t+1,:,:], mu_z)
                Sigma_t = torch.matmul(torch.matmul(A[:,t+1,:,:], Sigma_z), torch.transpose(A[:,t+1,:,:], 1,2))
                Sigma_t += self.Q.unsqueeze(0)

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred)

    def _filter_posterior_missing(self, obs, mask):
        # obs: (T ,B, N)
        # mask: (B, T)
        (T, B, _) = obs.size()
        mu_filt = torch.zeros(T, B, self.latent_dim, 1).to(obs.device).float()
        Sigma_filt = torch.zeros(T, B, self.latent_dim, self.latent_dim).to(obs.device).float()
        obs = obs.unsqueeze(-1)
        mu_t = self.mu_1.expand(B,-1).unsqueeze(-1)
        Sigma_t = self.Sigma_1.expand(B,-1,-1)

        mu_pred = torch.zeros_like(mu_filt).float()
        Sigma_pred = torch.zeros_like(Sigma_filt).float()

        A_t = torch.zeros(B,T,self.latent_dim,self.latent_dim).to(obs.device).float()
        C_t = torch.zeros(B,T,self.obs_dim,self.latent_dim).to(obs.device).float()
        code = self.start_code.reshape(1,-1).expand(B,-1)
        self.state_dyn_net = None
        alpha = self._compute_alpha(code, code, 
                                    torch.ones(B,self.obs_dim).to(obs.device))
        A, C = self._compute_mixture(alpha)
        A_t[:,0,:,:] = A
        C_t[:,0,:,:] = C

        for t in range(T):
            
            # mu/sigma: t | t-1
            mu_pred[t] = mu_t
            Sigma_pred[t] = Sigma_t

            y_pred = torch.matmul(C, mu_t)
            r = obs[t] - y_pred
            S_t = torch.matmul(torch.matmul(C, Sigma_t), torch.transpose(C, 1,2))
            S_t += self.R.unsqueeze(0)

            Kalman_gain = torch.matmul(torch.matmul(Sigma_t, torch.transpose(C, 1,2)), torch.inverse(S_t))       
            Kalman_gain *= mask[:,t].reshape(B,1,1)
            # filter: t | t
            mu_z = mu_t + torch.matmul(Kalman_gain, r)#*mask[:,t].reshape(B,1,1).expand(-1,self.latent_dim,-1)
            
            I_ = torch.eye(self.latent_dim).to(obs.device) - torch.matmul(Kalman_gain, C)
            #Sigma_z = torch.matmul(I_, Sigma_t)
            Sigma_z = torch.matmul(torch.matmul(I_, Sigma_t), torch.transpose(I_, 1,2)) 
            Sigma_z += torch.matmul(torch.matmul(Kalman_gain, self.R.unsqueeze(0)), torch.transpose(Kalman_gain, 1,2))
            mu_filt[t] = mu_z
            Sigma_filt[t] = Sigma_z
            if t != T-1:
                alpha = self._compute_alpha(obs[t].squeeze(-1), y_pred.squeeze(-1),
                                            mask[:,t].unsqueeze(-1).expand(-1,self.obs_dim))
                A, C = self._compute_mixture(alpha)
                A_t[:,t+1,:,:] = A
                C_t[:,t+1,:,:] = C
                # mu/sigma: t+1 | t for next step
                mu_t = torch.matmul(A, mu_z)
                Sigma_t = torch.matmul(torch.matmul(A, Sigma_z), torch.transpose(A, 1,2))
                Sigma_t += self.Q.unsqueeze(0)

        return (mu_filt, Sigma_filt), (mu_pred, Sigma_pred), A_t, C_t

    
    def _smooth_posterior(self, A, filtered, prediction):
        mu_filt, Sigma_filt = filtered
        mu_pred, Sigma_pred = prediction
        (T, *_) = mu_filt.size()
        mu_z_smooth = torch.zeros_like(mu_filt).float()
        Sigma_z_smooth = torch.zeros_like(Sigma_filt).float()
        mu_z_smooth[-1] = mu_filt[-1]
        Sigma_z_smooth[-1] = Sigma_filt[-1]
        for t in reversed(range(T-1)):
            J = torch.matmul(Sigma_filt[t], torch.matmul(torch.transpose(A[:,t+1,:,:], 1,2), torch.inverse(Sigma_pred[t+1])))
            mu_diff = mu_z_smooth[t+1] - mu_pred[t+1]
            mu_z_smooth[t] = mu_filt[t] + torch.matmul(J, mu_diff)

            cov_diff = Sigma_z_smooth[t+1] - Sigma_pred[t+1]
            Sigma_z_smooth[t] = Sigma_filt[t] + torch.matmul(torch.matmul(J, cov_diff), torch.transpose(J, 1, 2))
        
        return mu_z_smooth, Sigma_z_smooth

    def _kalman_posterior(self, obs, mask=None, filter_only=False, get_weights=False):
        # obs: (B ,T, N)
        if mask is None:
            A, C, weights = self._interpolate_matrices(obs)
            filtered, pred = self._filter_posterior(obs.transpose(0,1), A, C)
        else:
            filtered, pred, A, C = self._filter_posterior_missing(obs.transpose(0,1), mask)
        
        if filter_only:
            return filtered, A, C
        smoothed = self._smooth_posterior(A, filtered, pred)
        if get_weights:
            return smoothed, A, C, weights
        return smoothed, A, C

    def _decode(self, z):
        if self.mode=='base':
            x = self.decoder(z)
        else:
            x = self.flow_model.reverse(z,True)
        return x

    def _decode_latent(self, z_sample, A, C):
        (T, B, *_) = z_sample.size()
        z_next = torch.matmul(A.transpose(0,1), z_sample.unsqueeze(-1)).squeeze(-1)
        a_next = torch.matmul(C.transpose(0,1), z_sample.unsqueeze(-1)).squeeze(-1)
        return a_next, z_next

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def _compute_Kalman_loglikelihood(self, obs, filtered, A, C):
        (T, B, _) = obs.size()
        mu_t = self.mu_1.expand(B,-1).unsqueeze(-1)
        Sigma_t = self.Sigma_1.expand(B,-1,-1)
        y_pred = torch.zeros(T, B, self.obs_dim).to(obs.device).float()
        cov_pred = torch.zeros(T, B, self.obs_dim, self.obs_dim).to(obs.device).float()
        for t in range(T):
            y_pred[t,:,:] = torch.matmul(C[:,t,:,:], mu_t).squeeze(-1)
            S_t = torch.matmul(torch.matmul(C[:,t,:,:], Sigma_t), torch.transpose(C[:,t,:,:], 1,2))
            S_t += self.R.unsqueeze(0)
            cov_pred[t,:,:,:] = S_t

            if t != T-1:
                # mu/sigma: t+1 | t for next step
                mu_t = torch.matmul(A[:,t+1,:,:], filtered[0][t,:,:])
                Sigma_t = torch.matmul(torch.matmul(A[:,t+1,:,:], filtered[1][t,:,:,:]), torch.transpose(A[:,t+1,:,:], 1,2))
                Sigma_t += self.Q.unsqueeze(0)

        # log sum N (a_t; C_t mu_t|t-1, C_t*V_t|t-1*F_t + Q)
        p_y = MultivariateNormal(y_pred, scale_tril=torch.linalg.cholesky(cov_pred))
        log_p = p_y.log_prob(obs)
        return log_p

    def _compute_elbo(self, x, mask_frames, mask_visual, x_hat, a_mu, a_log_var, a_sample, smoothed, A, C):
        #(B, T, ch, H, W) = x.size()
        #if mask_frames is None:
        #    mask_frames = torch.ones(B,T).to(x.device)
        #if mask_visual is None:
        #    mask_visual = torch.ones(B, T, ch, H, W).to(x.device)
        (B, T, D) = x.size()
        if mask_frames is None:
            mask_frames = torch.ones(B,T).to(x.device)
        if mask_visual is None:
            mask_visual = torch.ones(B, T).to(x.device)
            
        # max: ELBO = log p(x_t|a_t) - (log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1))
        # min: -ELBO =  - log p(x_t|a_t) + log q(a) + log q(z) - log p(a_t | z_t) - log p(z_t| z_t-1)
        # Fixed variance
        # Reconstruction Loss p(x_t | a_t)
        decoder_x = MultivariateNormal(x_hat, covariance_matrix=torch.eye(self.input_dim).to(self.device)*0.01)
        #p_x = (decoder_x.log_prob(x)*mask_visual).reshape(B,T,-1).sum(-1)
        #nll = -(p_x*mask_frames).mean(dim=0).sum()
        p_x = (decoder_x.log_prob(x)).sum(-1)
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
        a_pred, z_next = self._decode_latent(z_sample, A, C)
        # -log p(z_t| z_t-1)
        kld = - decoder_z_0.log_prob(z_sample[0]).mean(dim=0)
        kld -= decoder_z.log_prob((z_sample[1:] - z_next[:-1])).mean(dim=1).sum()
        # -log p(a_t| z_t)
        kld -= (decoder_a.log_prob((a_sample - a_pred))*mask_frames.transpose(0,1)).mean(dim=1).sum()
        # log q(z)
        kld += smoothed_z.log_prob(z_sample).mean(dim=1).sum()
        # log q(a)
        kld += (q_a.log_prob(a_sample.transpose(0,1))*mask_frames).mean(dim=0).sum()
        
        elbo = kld + nll
        loss = kld + self.beta*nll
        losses = {
            'kld': kld,
            'elbo': elbo,
            'loss': loss,
            'nll': nll
        }
        return z_sample, losses

    def _compute_elbo_glow(self, logdet, a_sample, filtered, A, C):
        T, B, *_ = a_sample.size()
        log_p = self._compute_Kalman_loglikelihood(a_sample, filtered, A, C)
        nll = (-log_p.sum() - logdet.sum())/B

        losses = {
            'kld': 0,
            'elbo': nll,
            'loss': nll,
            'nll': nll
        }
        return None, losses

    def forward(self, x, mask_frames=None, mask_visual=None, variational=True):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,D) = x.size()
        #(B,T,C,H,W) = x.size()
        # q(a_t|x_t)
        if self.mode=='base':
            a_sample, a_mu, a_log_var = self._encode_obs(x.reshape(B*T,D), variational)
            #a_sample, a_mu, a_log_var = self._encode_obs(x.reshape(B*T,C,H,W), variational)
            a_sample = a_sample.reshape(B,T,-1)
            a_mu = a_mu.reshape(B,T,-1)
            a_log_var = a_log_var.reshape(B,T,-1)
            # q(z|a)
            smoothed, A_t, C_t = self._kalman_posterior(a_sample, mask_frames)
        else:
            _, logdet, a_sample = self.flow_model(x.reshape(B*T,C,H,W))
            if self.split_sizes is None:
                self.recalc_split = True
                self.split_sizes = []
            for i in range(len(a_sample)):
                if self.recalc_split:
                    self.split_sizes.append(a_sample[i].size())
                a_sample[i] = a_sample[i].reshape(B,T,-1)
            self.recalc_split = False   
            a_sample = torch.cat(a_sample,dim=-1)
            # q(z|a) For mode=glow, only filtered distribution is enough
            smoothed, A_t, C_t = self._kalman_posterior(a_sample, mask_frames, filter_only=True)

        # p(x_t|a_t)
        if self.mode=='base':
            x_hat = self._decode(a_sample.reshape(B*T,-1)).reshape(B,T,D)
            #x_hat = self._decode(a_sample.reshape(B*T,-1)).reshape(B,T,C,H,W)
            # ELBO
            z_sample, losses = self._compute_elbo(x, mask_frames, mask_visual, x_hat, a_mu, a_log_var, a_sample.transpose(0,1), smoothed, A_t, C_t)
        else:
            decode_in = []
            idx = 1024
            offset = 0
            for i in range(4):
                if i!=3:
                    idx //= 2
                decode_in.append(a_sample[:,:,offset:offset+idx].reshape(self.split_sizes[i]))
                offset += idx
            x_hat = self._decode(decode_in).reshape(B,T,C,H,W)
            z_sample, losses = self._compute_elbo_glow(logdet, a_sample.transpose(0,1), smoothed, A_t, C_t)
        return x_hat, a_sample, z_sample, losses

    def predict_sequence(self, input, seq_len=None):
        (B,T,C,H,W) = input.size()
        if seq_len is None:
            seq_len = T
        if self.mode=='base':
            a_sample, _, _ = self._encode_obs(input.reshape(B*T,C,H,W))
            a_sample = a_sample.reshape(B,T,-1)
        else:
            _, _, a_sample = self.flow_model(input.reshape(B*T,C,H,W))
            for i in range(len(a_sample)):
                a_sample[i] = a_sample[i].reshape(B,T,-1)
            a_sample = torch.cat(a_sample,dim=-1)
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
        if self.mode=='greparam':
            C1 = self._get_deconv_from_indices(self.C1_indices[0], self.C1_indices[1], self.C1, 1, 3)
            C2 = self._get_deconv_from_indices(self.C2_indices[0], self.C2_indices[1], self.C2, 3, 7)
            C3 = self._get_deconv_from_indices(self.C3_indices[0], self.C3_indices[1], self.C3, 7, 15)
            C4 = self._get_deconv_from_indices(self.C4_indices[0], self.C4_indices[1], self.C4, 15, 32)
            _C = torch.matmul(C4, torch.matmul(C3, torch.matmul(C2, C1))).unsqueeze(0).expand(B,-1,-1)
            #C1 = self._get_sparse_kernel_matrix(self.C1, 32, 32).transpose(1,2)
            #C2 = self._get_sparse_kernel_matrix(self.C2, 14, 14, s=2).transpose(1,2)
            #C3 = self._get_sparse_kernel_matrix(self.C3, 6, 6, s=2).transpose(1,2)
            #_C = torch.matmul(C1,C2)#,C3)
            #_C = torch.matmul(torch.matmul(C1,C2),C3)
            #_C = self._get_sparse_kernel_matrix(self.C).transpose(1,2)
        for t in range(seq_len):
            # Compute alpha from a_0:t-1
            if self.alpha=='mlp':
                dyn_emb = self.parameter_net(obs_prev)
            else:
                alpha_, cell_state = self.state_dyn_net
                dyn_emb, self.state_dyn_net = self.parameter_net(obs_prev.unsqueeze(1), (alpha_, cell_state))
                dyn_emb = self.alpha_out(dyn_emb)
            inter_weight = dyn_emb.softmax(-1).squeeze(1)
            ## Compute A_t, C_t
            A_t = torch.matmul(inter_weight, self.A.reshape(self.num_modes,-1)).reshape(B,self.latent_dim,self.latent_dim)
            if self.mode != 'greparam':
                C_t = torch.matmul(inter_weight, self.C.reshape(self.num_modes,-1)).reshape(B,self.obs_dim,self.latent_dim)
            else:
                #C_t =torch.matmul(inter_weight, _C.reshape(self.num_modes,-1)).reshape(B,self.obs_dim,self.latent_dim)
                C_t = _C
            # Calculate new z_t
            ## Update z_t
            latent_prev = torch.matmul(A_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            latent_seq[:,t] = latent_prev
            # Calculate new a_t
            obs_prev = torch.matmul(C_t, latent_prev.unsqueeze(-1)).squeeze(-1)
            obs_seq[:,t] = obs_prev

        if self.mode=='base':
            image_seq = self._decode(obs_seq.reshape(B*seq_len,-1)).reshape(B,seq_len,C,H,W) 
        else:
            decode_in = []
            idx = 1024
            offset = 0
            for i in range(4):
                if i!=3:
                    idx //= 2
                decode_in.append(obs_seq[:,:,offset:offset+idx].reshape(self.split_sizes[i]))
                offset += idx
            image_seq = self._decode(decode_in).reshape(B,seq_len,C,H,W)

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

    def get_alpha_from_obs(self, x, variational=True):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,D) = x.size()
        #(B,T,C,H,W) = x.size()
        # q(a_t|x_t)
        a_sample, _, _ = self._encode_obs(x.reshape(B*T,D), variational)
        #a_sample, a_mu, a_log_var = self._encode_obs(x.reshape(B*T,C,H,W), variational)
        a_sample = a_sample.reshape(B,T,-1)
        return self._kalman_posterior(a_sample, get_weights=True)[-1]

if __name__=="__main__":
    # Trial run
    """
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
    """
    import time
    import torch.nn.functional as F
    
    def get_sparse_kernel_matrix(K, h_X, w_X):

        # Assuming no channels and stride == 1.
        # Convert the kernel matrix to sparse matrix (dense matrix with lots of zeros in fact).
        # This is a little bit brain-twisting.

        h_K, w_K = K.shape

        h_Y, w_Y = (h_X - h_K)//2 + 1, (w_X - w_K)//2 + 1

        W = torch.zeros((h_Y * w_Y, h_X * w_X))
        for i in range(h_Y):
            for j in range(w_Y):
                for ii in range(h_K):
                    for jj in range(w_K):
                        W[i * w_Y + j, 2*i * w_X + 2*j + ii * w_X + jj] = K[ii, jj]
        return W
    start = time.time()
    K = torch.arange(1,37).reshape((6,6)).float()
    X = torch.randn((6,6))

    M = get_sparse_kernel_matrix(K,16,16)
    b = torch.matmul(M.T,X.flatten()).reshape((16,16))
    print("Elapsed:", time.time()-start)