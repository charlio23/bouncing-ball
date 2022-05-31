from pyexpat.errors import XML_ERROR_PARAM_ENTITY_REF
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, n_in, n_hid, n_out):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in, n_hid)
        self.fc2 = nn.Linear(n_hid, n_hid)
        self.fc_final = nn.Linear(n_hid, n_out)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)

    def forward(self, inputs):
        # Input shape: [num_sims, num_things, num_features]
        x = F.relu(self.fc1(inputs))
        x = F.relu(self.fc2(x))
        x = self.fc_final(x)
        return x

class SequentialEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_discr, output_cont, num_layers=4, bidirectional=True, output_type='many'):
        super(SequentialEncoder, self).__init__()
        self.bidirectional = bidirectional
        # Output type: one|many
        self.output_type = output_type
        self.lstm_encoder = nn.LSTM(input_dim, hidden_dim, 
                                    num_layers, batch_first=True, bidirectional=bidirectional)
        
    def forward(self, x):
        x, _ = self.lstm_encoder(x)
        x = x[:,-1] if self.output_type=='one' else x
        return x




class ResidualBlock(nn.Module):
    "Each residual block should up-sample an image x2"
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv2 = nn.Conv2d(in_channels=32,
                                out_channels=32,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.input_channels = input_channels
        if input_channels != 32:
            self.match_conv = nn.Conv2d(in_channels=input_channels,
                        out_channels=32,
                        kernel_size=1,
                        stride=1,
                        padding=0)
    

    def forward(self, x):
        x = F.upsample_bilinear(x, scale_factor=2)
        res = self.match_conv(x) if self.input_channels!=32 else x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + res


class CNNResidualDecoder(nn.Module):
    def __init__(self, latent_dim, out_dim=3):
        super(CNNResidualDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.first_mlp = MLP(latent_dim, latent_dim*4, latent_dim*4*4)
        self.first_block = ResidualBlock(input_channels=latent_dim)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(input_channels=32)
        for i in range(2)])
        self.out_conv = nn.Conv2d(in_channels=32,
                                  out_channels=out_dim,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.first_mlp(x).reshape((b, -1, 4, 4))
        x = self.first_block(x)
        for residual_layer in self.residual_blocks:
            x = residual_layer(x)
        x = torch.sigmoid(self.out_conv(x))
        return x

class CNNFastDecoderKVAE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNFastDecoder, self).__init__()
        self.in_dec = nn.Linear(input_dim, 32*4*4)
        self.hidden_convs = nn.ModuleList([
            nn.Conv2d(in_channels=32,
                      out_channels=32*4,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        for _ in range(3)])
        self.out_conv = nn.Conv2d(in_channels=32,
                      out_channels=output_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.in_dec(x).reshape((b, -1, 4, 4))
        for hidden_conv in self.hidden_convs:
            x = F.relu(hidden_conv(x))
            x = subpixel_reshape(x, 2)

        x = torch.sigmoid(self.out_conv(x))
        return x

class CNNFastDecoder(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(CNNFastDecoder, self).__init__()
        self.in_dec = nn.Linear(input_dim, 32*8*8)
        self.hidden_convs = nn.ModuleList([
            nn.ConvTranspose2d(in_channels=32,
                      out_channels=64,
                      kernel_size=3,
                      stride=2,
                      padding=1),
            nn.ConvTranspose2d(in_channels=64,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1)])
        self.out_conv = nn.Conv2d(in_channels=32,
                      out_channels=output_dim,
                      kernel_size=3,
                      stride=1,
                      padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.in_dec(x).reshape((b, -1, 8, 8))
        for hidden_conv in self.hidden_convs:
            x = F.relu(hidden_conv(x))
            x = F.pad(x, (0,1,0,1))

        x = torch.sigmoid(self.out_conv(x))
        return x

class CNNFastEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, log_var=True):
        super(CNNFastEncoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=32,
                                 kernel_size=3,
                                 stride=2,
                                 padding=1)
        self.hidden_conv = nn.ModuleList([
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=3,
                      stride=2,
                      padding=1)
        for _ in range(1)])

        self.out_mean = nn.Linear(32*8*8, output_dim)
        if log_var:
            self.out_log_var = nn.Linear(32*8*8, output_dim)
        else:
            self.out_log_var = None
    def forward(self, x):
        x = F.relu(self.in_conv(x))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = x.flatten(-3, -1)
        if self.out_log_var is None:
            return self.out_mean(x)
        mean, log_var = self.out_mean(x), self.out_log_var(x)
        return mean, log_var

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, num_layers, log_var=True):
        super(CNNEncoder, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=64,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.hidden_conv = nn.ModuleList([
            nn.Conv2d(in_channels=64,
                      out_channels=64,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        for _ in range(num_layers)])
        self.out_conv = nn.Conv2d(in_channels=64,
                                 out_channels=1,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        self.out_mean = MLP(256, 128, output_dim)
        if log_var:
            self.out_log_var = MLP(256, 128, output_dim)
        else:
            self.out_log_var = None

    def forward(self, x):
        x = F.relu(self.in_conv(x))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = F.relu(self.out_conv(x)).flatten(-3, -1)
        if self.out_log_var is None:
            return self.out_mean(x)
        mean, log_var = self.out_mean(x), self.out_log_var(x)
        return mean, log_var
        
class CNNEncoderPosition(nn.Module):
    def __init__(self, input_channels, n_in_pos, output_dim, num_layers):
        super(CNNEncoderPosition, self).__init__()
        self.in_conv = nn.Conv2d(in_channels=input_channels,
                                 out_channels=16,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.hidden_conv = nn.ModuleList([
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1)
        for _ in range(num_layers)])
        self.out_conv = nn.Conv2d(in_channels=16,
                                 out_channels=1,
                                 kernel_size=4,
                                 stride=2,
                                 padding=1)
        self.out_mean = nn.Linear(256 + 32, output_dim)
        self.pos_net = MLP(n_in_pos, 64, 32)
        

    def forward(self, x, pos):
        x = F.relu(self.in_conv(x))
        pos = F.relu(self.pos_net(pos))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = F.relu(self.out_conv(x)).flatten(-3, -1)
        x = torch.cat([pos, x], dim=1)
        mean = self.out_mean(x)
        return mean

def subpixel_reshape(x, factor):
    """
    Subpixel reshape.
    code adapted from https://github.com/simonkamronn/kvae/blob/849d631dbf2faf2c293d56a0d7a2e8564e294a51/kvae/utils/nn.py
    
    Reshape function for subpixel upsampling
    x: tensorflow tensor, shape = (bs, c, h, w)
    factor: interger, upsample factor
    Return: tensorflow tensor, shape = (bs, c//factor**2, h*factor,w*factor)
    """

    # input and output shapes
    B, C, H, W = x.size()
    out_H, out_W, out_C = H * factor, W * factor, C // factor ** 2

    assert C % factor == 0, "Number of input channels must be divisible by factor"

    intermediateshp = (-1, out_C, factor, factor, H, W)
    x = torch.reshape(x, intermediateshp)
    x = torch.permute(x, (0, 1, 4, 2, 5, 3))
    #                     B, C, H, F, W, F 

    x = torch.reshape(x, (-1, out_C, out_H, out_W))

    return x

if __name__=="__main__":
    net = CNNFastEncoder(3, 2)
    x = torch.randn((2,3,32,32))

    print(net(x)[0].size())