from curses.ascii import DEL
import torch
from torch.utils.data import DataLoader
from models.VRNN import VRNN
from dataloaders.bouncing_data import BouncingBallDataLoader
from utils.losses import nll_gaussian, kld_loss



dl = BouncingBallDataLoader('datasets/trimmed_datapoints.npy')
train_loader = torch.utils.data.DataLoader(dl, batch_size=16, shuffle=True)
sample = next(iter(train_loader)).float()

print(sample.size())
_, _, input_dim = sample.size()
vrnn = VRNN(input_dim, 128, 32).float()
print(vrnn)
reconstr_seq, z_params, x_params = vrnn(sample)
print(z_params.size())
print(x_params.size())
print(kld_loss(z_params[:,:,0,:], z_params[:,:,1,:]))
print(nll_gaussian(x_params[:,:,0,:], x_params[:,:,1,:], sample))
loss = kld_loss(z_params[:,:,0,:], z_params[:,:,1,:]) + nll_gaussian(x_params[:,:,0,:], x_params[:,:,1,:], sample)
print(loss)