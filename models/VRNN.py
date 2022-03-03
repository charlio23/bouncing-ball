import torch
import torch.nn as nn
from models.modules import MLP

class VRNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, input_type='base', decoder='LSTM'):
        super(VRNN, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.prior = MLP(self.hidden_dim, self.hidden_dim, self.latent_dim*2)
        if input_type=='visual':
            self.embedder_x = None
            self.decoder = None
        else:
            self.embedder_x = MLP(self.input_dim, self.hidden_dim, self.hidden_dim)
            self.decoder = MLP(self.hidden_dim*2, self.hidden_dim, self.input_dim*2)
        self.encoder = MLP(self.hidden_dim*2, self.hidden_dim, self.latent_dim*2)
        self.embedder_z = MLP(self.latent_dim, self.hidden_dim, self.hidden_dim)
        if decoder=='vainilla':
            self.hidden_decoder = None
        elif decoder=='LSTM':
            self.hidden_decoder = nn.LSTMCell(self.hidden_dim*2, self.hidden_dim)
        elif decoder=='GRU':
            self.hidden_decoder = None
        else:
            raise NotImplementedError("Decoder '" + decoder + "' not implemented.")
        
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def _inference(self, x, h):
        encoder_in = torch.cat([self.embedder_x(x), h], dim=-1)
        (z_mu, z_log_var) = self.encoder(encoder_in).split(self.latent_dim, dim=-1)
        eps = torch.normal(mean=torch.zeros_like(z_mu))
        z_std = (z_log_var*0.5).exp()
        sample = z_mu + z_std*eps
        return sample, z_mu, z_log_var

    def _decode(self, z, h_prev, c_prev=None):
        embed_z = self.embedder_z(z)
        decoder_in = torch.cat([embed_z, h_prev], dim=-1)
        (x_mu, x_log_var) = self.decoder(decoder_in).split(self.input_dim, dim=-1)
        eps = torch.normal(mean=torch.zeros_like(x_mu))
        # Cap std to 100 for stability
        x_std = torch.minimum((x_log_var*0.5).exp(), torch.FloatTensor([100.]))
        x = x_mu + x_std*eps
        input = torch.cat([self.embedder_x(x), embed_z], dim=-1)
        h, c = self.hidden_decoder(input, (h_prev, c_prev))
        return x, x_mu, x_log_var, h, c

    def _sample(self, h):
        z_mu, z_log_var = self.prior(h)
        eps = torch.normal(mean=torch.zeros_like(z_mu))
        # Cap std to 100 for stability
        z_std = torch.minimum((z_log_var*0.5).exp(), torch.FloatTensor([100.]))
        sample = z_mu + z_std*eps
        return sample

    def forward(self, x):
        b, seq_len, _ = x.size()
        h_prev = torch.zeros((b, self.hidden_dim))
        c_prev = torch.zeros((b, self.hidden_dim))
        reconstr_seq = torch.zeros_like(x)
        sizes_z_params = [b, seq_len, 2, self.latent_dim]
        z_params = torch.zeros(sizes_z_params)
        sizes_x_params = [b, seq_len, 2, self.input_dim]
        x_params = torch.zeros(sizes_x_params)
        for i in range(seq_len):
            last_x = x[:,i,:]

            # Autoencode
            z, z_mu, z_log_var = self._inference(last_x, h_prev)
            x_hat, x_hat_mu, x_hat_log_var, h_prev, c_prev = self._decode(z, h_prev, c_prev)

            # Collect parameters
            reconstr_seq[:, i, :] = x_hat
            z_params[:, i, 0, :] = z_mu
            z_params[:, i, 1, :] = z_log_var
            x_params[:, i, 0, :] = x_hat_mu
            x_params[:, i, 1, :] = x_hat_log_var

        return reconstr_seq, z_params, x_params
