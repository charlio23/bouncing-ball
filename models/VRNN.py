from re import X
import torch
import torch.nn as nn
from models.modules import MLP, CNNEncoder, CNNEncoderPosition, CNNResidualDecoder

class VRNN(nn.Module):
    def __init__(self, input_dim, input_pos, hidden_dim, latent_dim, num_rec_layers=1, input_type='base', decoder='LSTM'):
        super(VRNN, self).__init__()
        self.input_dim = input_dim
        self.input_pos = input_pos
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_rec_layers = num_rec_layers
        self.prior = MLP(self.hidden_dim, self.hidden_dim, self.latent_dim*2)
        self.input_type = input_type
        if input_type=='visual':
            self.embedder_x = CNNEncoder(self.input_dim, self.latent_dim, 4)
            self.decoder = CNNResidualDecoder()
        else:
            #self.embedder_x = CNNEncoderPosition(self.input_dim, input_pos, self.hidden_dim, 4)
            self.embedder_x = MLP(self.input_pos, self.hidden_dim, self.latent_dim)
            self.decoder = MLP(self.hidden_dim + self.latent_dim, self.hidden_dim, self.input_pos)
        self.encoder = MLP(self.hidden_dim + self.latent_dim, self.hidden_dim, self.latent_dim*2)
        self.embedder_z = MLP(self.latent_dim, self.hidden_dim, self.latent_dim)
        if decoder=='vainilla':
            self.hidden_decoder = None
        elif decoder=='LSTM':
            self.hidden_decoder = nn.LSTMCell(self.latent_dim*2, self.hidden_dim)
            if num_rec_layers==3:
                self.hidden_decoder = nn.ModuleList([
                    nn.LSTMCell(self.latent_dim*2, self.hidden_dim),
                    nn.LSTMCell(self.hidden_dim, self.hidden_dim),
                    nn.LSTMCell(self.hidden_dim, self.hidden_dim)
                ])
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
        if self.num_rec_layers == 1:
            encoder_in = torch.cat([self.embedder_x(x), h], dim=-1)
        else:
            encoder_in = torch.cat([self.embedder_x(x), h[-1]], dim=-1)
        (z_mu, z_log_var) = self.encoder(encoder_in).split(self.latent_dim, dim=-1)
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(x.device)
        z_std = torch.minimum((z_log_var*0.5).exp(), torch.FloatTensor([100.]).to(x.device))
        sample = z_mu + z_std*eps
        return sample, z_mu, z_log_var

    def _decode(self, z, h_prev, c_prev=None):
        embed_z = self.embedder_z(z)
        if self.num_rec_layers == 1:
            decoder_in = torch.cat([embed_z, h_prev], dim=-1)
        else:
            decoder_in = torch.cat([embed_z, h_prev[-1]], dim=-1)
        x_ = self.decoder(decoder_in)
        input = torch.cat([self.embedder_x(x_), embed_z], dim=-1)
        if self.num_rec_layers == 1:
            h_prev, c_prev = self.hidden_decoder(input, (h_prev, c_prev))
        else:
            b, *_ = input.size()
            for i in range(self.num_rec_layers):
                if i==0:
                    h_prev[i], c_prev[i] = self.hidden_decoder[i](input, (h_prev[i], c_prev[i]))
                else:
                    h_prev[i], c_prev[i] = self.hidden_decoder[i](h_prev[i-1], (h_prev[i], c_prev[i]))
        return x_, h_prev, c_prev

    def _prior(self, h):
        (z_mu, z_log_var) = self.prior(h).split(self.latent_dim, dim=-1)
        return z_mu, z_log_var

    def _sample(self, h):
        if self.num_rec_layers == 1:
            (z_mu, z_log_var) = self._prior(h)
        else:
            (z_mu, z_log_var) = self._prior(h[-1])
        eps = torch.normal(mean=torch.zeros_like(z_mu)).to(z_mu.device)
        # Cap std to 100 for stability
        z_std = (z_log_var*0.5).exp().to(z_mu.device)
        sample = z_mu + z_std*eps
        return sample
    
    def predict_sequence(self, input, seq_len=None):
        b, T, *_ = input.size()
        h_prev = [torch.zeros((b, self.hidden_dim)).to(input.device) for _ in range(self.num_rec_layers)]
        c_prev = [torch.zeros((b, self.hidden_dim)).to(input.device) for _ in range(self.num_rec_layers)]
        if self.num_rec_layers == 1:
            h_prev = h_prev[0]
            c_prev = c_prev[0]
        if seq_len is None:
            seq_len = T
        for i in range(T):
            last_x = input[:,i,:]
            # Autoencode
            z, _, _ = self._inference(last_x, h_prev)
            _, h_prev, c_prev = self._decode(z, h_prev, c_prev)
        
        pos_shape = [input.size(0), seq_len, input.size(2)]
        reconstr_seq = torch.zeros(pos_shape).to(input.device)
        for i in range(seq_len):
            z_sampled = self._sample(h_prev)
            pos_hat, h_prev, c_prev = self._decode(z_sampled, h_prev, c_prev)
            reconstr_seq[:, i, :] = pos_hat

        return reconstr_seq
        
    def forward(self, x):
        b, seq_len, *_ = x.size()
        h_prev = [torch.zeros((b, self.hidden_dim)).to(x.device) for _ in range(self.num_rec_layers)]
        c_prev = [torch.zeros((b, self.hidden_dim)).to(x.device) for _ in range(self.num_rec_layers)]
        if self.num_rec_layers == 1:
            h_prev = h_prev[0]
            c_prev = c_prev[0]
        reconstr_seq = torch.zeros_like(x).to(x.device)
        sizes_z_params = [b, seq_len, 2, self.latent_dim]
        z_params = torch.zeros(sizes_z_params).to(x.device)
        z_params_prior = torch.zeros(sizes_z_params).to(x.device)
        for i in range(seq_len):

            last_x = x[:,i,:]
            # Autoencode
            z, z_mu, z_log_var = self._inference(last_x, h_prev)
            if self.num_rec_layers == 1:
                z_mu_prior, z_log_var_prior = self._prior(h_prev)
            else:
                z_mu_prior, z_log_var_prior = self._prior(h_prev[-1])
            x_hat, h_prev, c_prev = self._decode(z, h_prev, c_prev)

            # Collect parameters
            reconstr_seq[:, i, :] = x_hat
            z_params[:, i, 0, :] = z_mu
            z_params[:, i, 1, :] = z_log_var
            z_params_prior[:, i, 0, :] = z_mu_prior
            z_params_prior[:, i, 1, :] = z_log_var_prior

        return reconstr_seq, z_params, z_params_prior
