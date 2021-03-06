# Imports
import argparse
from ast import arg
import os
import time

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader

from dataloaders.bouncing_data import BouncingBallDataLoader
from utils.losses import kld_loss_standard
from models.VAE import ImageVAE


parser = argparse.ArgumentParser(description='Image VAE trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='./dataset/train', type=str)
parser.add_argument('--runs_path', default='/data2/users/cb221/runs', type=str)
parser.add_argument('--epochs', default=40, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--determ', action='store_true', help='Use VAE or deterministic AE')
parser.add_argument('--beta', default=1, type=float,metavar='N', help='beta VAE param')
parser.add_argument('--lr', default=5e-4, type=float, metavar='N', help='learning rate')
parser.add_argument('--latent_dim', default=32, type=int, metavar='N', help='dimension of latent space')


def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("/data2/users/cb221/stored_models/", exist_ok=True)
    torch.save(state, "/data2/users/cb221/stored_models/" + filename + '_latest.pth.tar')

def main():
    global args, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join(args.runs_path, args.name))
    print(args)
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    dl = BouncingBallDataLoader(args.train_root, images=True)
    train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True)
    sample = next(iter(train_loader))[0].float()
    _, _, input_dim, *_ = sample.size()
    variational = not args.determ
    # Load model

    vae = ImageVAE(input_dim, 128, args.latent_dim).float().to(device)
    print(vae)

    # Set up optimizers
    optimizer = Adam(vae.parameters(), lr=args.lr)
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

    # Train Loop
    vae.train()
    for epoch in range(0, args.epochs):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):
            sample = sample[0]
            # Forward sample to network
            b, seq_len, C, H, W = sample.size()
            sample_in = sample.float().reshape((b*seq_len, C, H, W))
            var = Variable(sample_in, requires_grad=True).to(device)
            optimizer.zero_grad()
            x_hat, z_mu, z_log_var = vae(var, variational=variational)
            # Compute loss and optimize params
            mse = F.mse_loss(x_hat, var, reduction='sum')/(b)
            loss = mse
            if variational:
                kld = kld_loss_standard(z_mu, z_log_var)
                loss += args.beta*kld
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
                if variational:
                    writer.add_scalar('data/kl_loss', kld, i + epoch*len(train_loader))
                writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))
            if i % 100 == 0:
                writer.add_video('data/Inferred_vid',video_tensor_hat[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_vid',video_tensor_true[:16], i + epoch*len(train_loader))
        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'vae': vae.state_dict()
        }, filename='image_vae')


if __name__=="__main__":
    main()