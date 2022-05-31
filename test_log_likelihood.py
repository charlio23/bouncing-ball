# Imports
import argparse
import os
import time

from matplotlib import pyplot as plt

import numpy as np

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataloaders.bouncing_data import BouncingBallDataLoader, MissingBallDataset, SquareBallDataset
from models.KalmanVAE import KalmanVAE
from models.VRNN import VRNN


parser = argparse.ArgumentParser(description='Image VAE trainer')

parser.add_argument('--train_root', default='./dataset/train', type=str)
parser.add_argument('-b', '--batch-size', default=32, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--seq_len', default=100, type=int, metavar='N', help='length of input sequene')
parser.add_argument('--load', action='store', type=str, required=False, help='Path from where to load network.')
parser.add_argument('--missing', action='store_true', help='Activate missing frames')
parser.add_argument('--corrupt', action='store_true', help='Activate corrupt data')
parser.add_argument('--model', default='kvae', type=str, help='kvae|vrnn')
parser.add_argument('--num_samples', default=20, type=int, help='num samples used to compute LL')

def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def main():
    global args
    args = parser.parse_args()
    print(args)
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    if args.missing:
        dl = MissingBallDataset(args.train_root,
                                gt_dir='/data2/users/hbz15/2_body_black_white_real/test')
    elif args.corrupt:
        dl = SquareBallDataset(args.train_root,
                               '/data2/users/hbz15/2_body_black_white_real/mask_test',
                               gt_dir='/data2/users/hbz15/2_body_black_white_real/test')
    else:
        dl = BouncingBallDataLoader(args.train_root, images=True)
    train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True)
    if args.model == 'kvae':
        model = KalmanVAE(input_dim=1, hidden_dim=32, obs_dim=4, 
                        latent_dim=32, num_modes=8, beta=1, 
                        alpha='rnn').double().cuda()
    else:
        model = VRNN(1, 2, 32, 4, num_rec_layers=3, input_type='visual').float().to(device)
    print(model)
    if args.load is not None:
        model.load_state_dict(torch.load(args.load)[args.model])
        print("=> Model loaded successfully")
    model.eval()
    with torch.no_grad():
        log_likelihood_data = torch.empty(0).to(device)
        for i, sample in enumerate(train_loader, 1):
            b, seq_len, C, H, W = sample[0][:,:args.seq_len].size()
            var = (Variable(sample[0][:,:args.seq_len].double(), requires_grad=False).to(device) > 0.5).double()
            target = sample[2][:,:args.seq_len].double().to(device)
            if args.model=='vrnn':
                target = target.float()
                var = var.float()
            if args.missing:
                mask = sample[1][:,:args.seq_len].cuda().double()
                if args.model=='vrnn':
                    mask = mask.reshape(b,seq_len,1,1,1).float()
                log_likeli = model.test_log_likeli(var, target=target, mask_frames=mask, L=args.num_samples)
            elif args.corrupt:
                log_likeli = model.test_log_likeli(var, target=target, L=args.num_samples)
            log_likelihood_data = torch.cat([log_likelihood_data, log_likeli.detach()])
            if i%10 == 0:
                print(i,"=> test log likelihood:", log_likelihood_data.mean())

    print("=> Final test log likelihood is:", log_likelihood_data.mean())

if __name__=="__main__":
    main()