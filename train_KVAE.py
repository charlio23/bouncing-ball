# Imports
import argparse
import os
import time

from matplotlib import pyplot as plt

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
from models.KalmanVAE import KalmanVAE


parser = argparse.ArgumentParser(description='Image VAE trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='./dataset/train', type=str)
parser.add_argument('--runs_path', default='/data2/users/cb221/runs_KVAE', type=str)
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

    kvae = KalmanVAE(input_dim=3, hidden_dim=128, obs_dim=2, latent_dim=4, num_modes=3, beta=args.beta).float().cuda()
    print(kvae)

    # Set up optimizers
    optimizer = Adam(kvae.parameters(), lr=args.lr)
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)

    # Train Loop
    kvae.train()
    for epoch in range(0, args.epochs):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):
            sample = sample[0][:,:50]
            # Forward sample to network
            b, seq_len, C, H, W = sample.size()
            var = Variable(sample.float(), requires_grad=True).to(device)
            optimizer.zero_grad()
            x_hat, a_mu, _, losses = kvae(var, variational=variational)
            # Compute loss and optimize params
            losses['loss'].backward()
            optimizer.step()
            
            mse = F.mse_loss(x_hat, var, reduction='sum')/(b)
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            video_tensor_hat = x_hat.reshape((b, seq_len, C, H, W)).detach().cpu()
            video_tensor_true = sample.float().detach().cpu()
            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses['elbo'], mse=mse))
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                if variational:
                    writer.add_scalar('data/kl_loss', losses['kld'], i + epoch*len(train_loader))
                writer.add_scalar('data/elbo', losses['elbo'], i + epoch*len(train_loader))
                writer.add_scalar('data/loss', losses['loss'], i + epoch*len(train_loader))
            if i % 100 == 0:
                writer.add_video('data/Inferred_vid',video_tensor_hat[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_vid',video_tensor_true[:16], i + epoch*len(train_loader))
                fig_inferred = plt.figure()
                ax1 = fig_inferred.add_subplot(1,1,1)
                a_mu = a_mu.detach().cpu()
                ax1.scatter(a_mu[0,:,0],a_mu[0,:,1])
                writer.add_figure('data/inferred_latent_cont_state', fig_inferred, i + epoch*len(train_loader))
        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'kvae': kvae.state_dict()
        }, filename='KVAE')


if __name__=="__main__":
    main()