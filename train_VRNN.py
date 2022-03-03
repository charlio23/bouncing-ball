# Imports

import time

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from dataloaders.bouncing_data import BouncingBallDataLoader
from utils.losses import kld_loss, nll_gaussian
from models.VRNN import VRNN

def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def main():

    # Set up writers and device
    seq_len = 100
    device = get_device()

    # Load dataset
    dl = BouncingBallDataLoader('datasets/bouncing_ball/train', False)
    train_loader = DataLoader(dl, batch_size=16, shuffle=True)
    sample = next(iter(train_loader)).float()
    _, _, input_dim = sample.size()

    # Load model

    vrnn = VRNN(input_dim, 128, 32).float().to(device)
    print(vrnn)

    # Set up optimizers
    optimizer = Adam(vrnn.parameters(), lr=1e-5)

    # Train Loop
    vrnn.train()
    for epoch in range(1, 10):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):

            # Forward sample to network
            var = Variable(sample[:,:seq_len].float(), requires_grad=True).to(device)
            optimizer.zero_grad()
            reconstr_seq, z_params, x_params = vrnn(var)
            # Compute loss and optimize params
            loss = kld_loss(z_params[:,:,0,:], z_params[:,:,1,:])
            loss += nll_gaussian(x_params[:,:,0,:], x_params[:,:,1,:], var)
            loss.backward()
            optimizer.step()
            b, seq_len, dim = var.size()
            mse = F.mse_loss(reconstr_seq, var, reduction='sum')/(b*seq_len*dim)
            
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=loss, mse=mse))

if __name__=="__main__":
    main()