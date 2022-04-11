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

from dataloaders.nascar import NascarDataLoader
from dataloaders.bouncing_data import BouncingBallDataLoader
from models.VRSLDS import VRSLDS

parser = argparse.ArgumentParser(description='VrSLDS trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_path', default='/data2/users/cb221/nascar.npz', type=str)
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--beta', default=1, type=float,metavar='N', help='beta VAE param')
parser.add_argument('--lr', default=1e-4, type=float, metavar='N', help='learning rate')
parser.add_argument('--hidden_dim', default=128, type=int, metavar='N', help='dimension of hidden space')
parser.add_argument('--cont_dim', default=2, type=int, metavar='N', help='dimension of continous latent space')
parser.add_argument('--discr_dim', default=4, type=int, metavar='N', help='dimension of discrete latent space')
parser.add_argument('--seq_len', default=50, type=int, metavar='N', help='length of input sequence')
parser.add_argument('--num_enc_layers', default=5, type=int, metavar='N', help='Number of LSTM encoder layers')
parser.add_argument('--load', action='store', type=str, required=False, help='Path from where to load network.')
parser.add_argument('--SB', action='store_true', help='Use image video as input')
parser.add_argument('--experiment', default='ball', type=str, help='Experiment ball|nascar|video')


def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("/data2/users/cb221/stored_models_vrslds/", exist_ok=True)
    torch.save(state, "/data2/users/cb221/stored_models_vrslds/" + filename + '_latest.pth.tar')

def main():
    global args, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join("/data2/users/cb221/runs_VrSLDS_" + args.experiment, args.name))
    print(args)
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    if args.experiment=='nascar':
        dl = NascarDataLoader(args.train_path, seq_len=args.seq_len)
        train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True, num_workers=4)
        obs_dim = next(iter(train_loader))[0].size(-1)
    elif args.experiment=='ball':
        dl = BouncingBallDataLoader('/data2/users/cb221/bouncing_ball_square/train')
        train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True, num_workers=4)
        obs_dim = next(iter(train_loader))[1].size(-1)
    else:
        raise NotImplementedError(args.experiment + 'not implemented!')
    
    # Load model
    vrslds = VRSLDS(obs_dim=obs_dim, discr_dim=args.discr_dim, cont_dim=args.cont_dim,
                    hidden_dim=args.hidden_dim, num_rec_layers=args.num_enc_layers, 
                    beta=args.beta, SB=args.SB).float().to(device)
    print(vrslds)
    if args.load is not None:
        vrslds.load_state_dict(torch.load(args.load)['vrnn'])
        print("=> Model loaded successfully")

    # Set up optimizers
    optimizer = Adam(vrslds.parameters(), lr=args.lr)
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=50, gamma=gamma)

    # Train Loop
    vrslds.train()
    for epoch in range(0, args.epochs):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):
            if args.experiment=='nascar':
                y, x, z = sample
            elif args.experiment=='ball':
                _, y = sample
            # Forward sample to network
            var = Variable(y.float(), requires_grad=True).to(device)
            optimizer.zero_grad()
            y_pred, x_sample, z_sample, losses = vrslds(var)
            # Compute loss and optimize params
            loss = losses['loss']
            loss.backward()
            optimizer.step()
            
            mse = F.mse_loss(y_pred, var)
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Elbo {loss:.4e}\t MSE: {mse:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses['elbo'], mse=mse))
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                writer.add_scalar('data/kld', losses['kld'], i + epoch*len(train_loader))
                writer.add_scalar('data/loss', loss, i + epoch*len(train_loader))
                writer.add_scalar('data/elbo', losses['elbo'], i + epoch*len(train_loader))
            if i % 100 == 0:
                colors = np.array(['c', 'r', 'g', 'y', 'b'])
                x_sample = x_sample.detach().cpu()
                inferred_states = z_sample.argmax(-1).detach().cpu()
                fig_inferred = plt.figure()
                ax1 = fig_inferred.add_subplot(1,1,1)
                ax1.scatter(x_sample[0,:,0],x_sample[0,:,1], color=colors[inferred_states[0]])
                writer.add_figure('data/inferred_latent_cont_state', fig_inferred, i + epoch*len(train_loader))
                if args.experiment=='ball':
                    fig_real = plt.figure()
                    ax2 = fig_real.add_subplot(1,1,1)
                    ax2.scatter(x[0,:,0],x[0,:,1], color=colors[z[0]])
                    writer.add_figure('data/true_latent_cont_state', fig_real, i + epoch*len(train_loader))
        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'vrnn': vrslds.state_dict()
        }, filename=args.name)


if __name__=="__main__":
    main()