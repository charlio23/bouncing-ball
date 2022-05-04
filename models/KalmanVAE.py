import torch
import torch.nn as nn
from modules import MLP, CNNEncoder, CNNResidualDecoder

class KalmanVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(KalmanVAE, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.encoder = CNNEncoder(self.input_dim, self.latent_dim, 4)
        self.decoder = CNNResidualDecoder(self.latent_dim)
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

    def filter_posterior(self, obs):
    
    def smooth_posterior(self, obs):


    def kalman_posterior(self, obs):

    def _decode(self, z):
        x = self.decoder(z)
        return x

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def compute_elbo(self):

        # ELBO: 

    def forward(self, x, variational=True):
        # Input is (B,T,C,H,W)
        # Autoencode
        (B,T,C,H,W) = x.size()
        x = x.reshape(B*T,C,H,W)
        a_sample, a_mu, a_log_var = self._encode_obs(x, variational)

        x_hat = self._decode(a_sample.reshape(B*T,-1))
        return x_hat, a_mu, a_log_var

if __name__=="__main__":
    # Trial run
    net = KalmanVAE(input_dim=1, hidden_dim=128, latent_dim=32)

    sample = torch.rand((1,30,1,32,32))
    torch.autograd.set_detect_anomaly(True)
    x_hat, a_mu, a_log_var = net(sample)
    print(x_hat.size())
    print(a_mu.size())
    print(a_log_var.size())