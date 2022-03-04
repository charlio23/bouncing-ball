# Imports
import os
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
from models.VAE import ImageVAE

def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("stored_models/", exist_ok=True)
    torch.save(state, "stored_models/" + filename + '_latest.pth.tar')

def main():
    # Set up writers and device
    writer = SummaryWriter(log_dir=os.path.join('runs', 'vrnn_visual'))
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    dl = BouncingBallDataLoader('datasets/bouncing_ball_32/train', images=True)
    train_loader = DataLoader(dl, batch_size=2, shuffle=True)
    sample = next(iter(train_loader)).float()
    _, _, input_dim, *_ = sample.size()

    # Load model

    vae = ImageVAE(input_dim, 128, 32, input_type='visual').float().to(device)
    print(vae)

    # Set up optimizers
    optimizer = Adam(vae.parameters(), lr=1e-5)

    # Train Loop
    vae.train()
    for epoch in range(1, 500):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):

            # Forward sample to network
            b, seq_len, C, H, W = sample.size()
            sample_in = sample.float().reshape((b*seq_len, C, H, W))
            var = Variable(sample_in, requires_grad=True).to(device)
            optimizer.zero_grad()
            x_hat, z_mu, z_log_var = vae(var)
            # Compute loss and optimize params
            b, seq_len, dim, *_ = var.size()
            kld = kld_loss(z_mu, z_log_var)
            mse = F.mse_loss(x_hat, var, reduction='sum')/(b)
            loss = kld + mse
            loss.backward()
            optimizer.step()
            
            
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            video_tensor_hat = x_hat.reshape((b, seq_len, C, H, W)).detach().cpu()
            video_tensor_true = sample.float().detach().cpu()
            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=loss, mse=mse))
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                writer.add_scalar('data/kl_loss', kld, i + epoch*len(train_loader))
                writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))
                writer.add_video('data/Inferred_vid',video_tensor_hat[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_vid',video_tensor_true[:16], i + epoch*len(train_loader))

        save_checkpoint({
            'epoch': epoch,
            'vae': vae.state_dict()
        }, filename='image_vae')


if __name__=="__main__":
    main()