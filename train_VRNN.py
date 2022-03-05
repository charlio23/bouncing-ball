# Imports
import argparse
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
from models.VRNN import VRNN

parser = argparse.ArgumentParser(description='VRNN trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')

def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("stored_models/", exist_ok=True)
    torch.save(state, "stored_models/" + filename + '_latest.pth.tar')

def main():
    global args, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join("runs", args.name))
    print(args)
    input_type = 'visual'
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    dl = BouncingBallDataLoader(args.train_root, images=True)
    train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True)
    sample = next(iter(train_loader)).float()
    _, _, input_dim, *_ = sample.size()

    # Load model

    vrnn = VRNN(input_dim, 128, 32, input_type='visual').float().to(device)
    print(vrnn)

    # Set up optimizers
    optimizer = Adam(vrnn.parameters(), lr=1e-5)

    # Train Loop
    vrnn.train()
    for epoch in range(1, args.epochs):
        
        end = time.time()
        for i, sample in enumerate(train_loader, 1):

            # Forward sample to network
            var = Variable(sample.float(), requires_grad=True).to(device)
            optimizer.zero_grad()
            reconstr_seq, z_params, x_params = vrnn(var)
            # Compute loss and optimize params
            b, seq_len, dim = var.size()
            kld = kld_loss(z_params[:,:,0,:], z_params[:,:,1,:])
            mse = F.mse_loss(reconstr_seq, var, reduction='sum')/(b)
            loss = kld
            if input_type == 'visual':
                loss += mse
            else:
                nll = nll_gaussian(x_params[:,:,0,:], x_params[:,:,1,:], var)
                loss += nll
            loss.backward()
            optimizer.step()
            
            
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % 10 == 0:
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=loss, mse=mse))
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                writer.add_scalar('data/kl_loss', kld, i + epoch*len(train_loader))
                writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))

        save_checkpoint({
            'epoch': epoch,
            'vrnn': vrnn.state_dict()
        }, filename='VRNN_visual')


if __name__=="__main__":
    main()