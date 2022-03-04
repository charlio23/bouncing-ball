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

class ResidualBlock(nn.Module):
    "Each residual block should up-sample an image x2"
    def __init__(self, input_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=input_channels,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.conv2 = nn.Conv2d(in_channels=64,
                                out_channels=64,
                                kernel_size=3,
                                stride=1,
                                padding=1)
        self.input_channels = input_channels
        if input_channels != 64:
            self.match_conv = nn.Conv2d(in_channels=input_channels,
                        out_channels=64,
                        kernel_size=1,
                        stride=1,
                        padding=0)
    

    def forward(self, x):
        x = F.upsample_bilinear(x, scale_factor=2)
        res = self.match_conv(x) if self.input_channels!=64 else x
        x = F.relu(self.conv1(x))
        x = self.conv2(x)
        return x + res


class CNNResidualDecoder(nn.Module):
    def __init__(self):
        super(CNNResidualDecoder, self).__init__()
        self.first_mlp = MLP(256, 64, 64)
        self.first_block = ResidualBlock(input_channels=1)
        self.residual_blocks = nn.ModuleList([
            ResidualBlock(input_channels=64)
        for i in range(2)])
        self.out_conv = nn.Conv2d(in_channels=64,
                                  out_channels=3,
                                  kernel_size=3,
                                  stride=1,
                                  padding=1)

    def forward(self, x):
        b, *_ = x.size()
        x = self.first_mlp(x).reshape((b, 1, 8, 8))
        x = self.first_block(x)
        for residual_layer in self.residual_blocks:
            x = residual_layer(x)
        x = torch.sigmoid(self.out_conv(x))
        return x

class CNNEncoder(nn.Module):
    def __init__(self, input_channels, output_dim, num_layers):
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
        self.out_mlp = MLP(1024, 128, 128)

    def forward(self, x):
        x = F.relu(self.in_conv(x))
        for hidden_layer in self.hidden_conv:
            x = F.relu(hidden_layer(x))
        x = F.relu(self.out_conv(x)).flatten(-3, -1)
        x = self.out_mlp(x)
        return x
        
