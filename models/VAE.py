import torch
import torch.nn as nn
from models.modules import MLP, CNNEncoder, CNNResidualDecoder

class ImageVAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(ImageVAE, self).__init__()
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

    def _encode(self, x):
        (z_mu, z_log_var) = self.encoder(x)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = torch.minimum((z_log_var*0.5).exp(), torch.FloatTensor([100.]).to(x.device))
        sample = z_mu + z_std*eps
        return sample, z_mu, z_log_var

    def _decode(self, z):
        x = self.decoder(z)
        return x

    def _sample(self, size):
        eps = torch.normal(mean=torch.zeros(size))
        return self._decode(eps)

    def forward(self, x):
        # Autoencode
        z, z_mu, z_log_var = self._encode(x)
        x_hat = self._decode(z)
        return x_hat, z_mu, z_log_var
