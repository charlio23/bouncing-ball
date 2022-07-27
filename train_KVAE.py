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

from dataloaders.bouncing_data import BouncingBallDataLoader, MissingBallDataset, SquareBallDataset

from models.KalmanVAE import KalmanVAE
from models.ExtendedKalmanVAE import ExtendedKalmanVAE


parser = argparse.ArgumentParser(description='Image VAE trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='./dataset/train', type=str)
parser.add_argument('--runs_path', default='/data2/users/cb221/runs_KVAE_GLOW', type=str)
parser.add_argument('--epochs', default=80, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--determ', action='store_true', help='Use VAE or deterministic AE')
parser.add_argument('--beta', default=1, type=float,metavar='N', help='beta VAE param')
parser.add_argument('--lr', default=5e-4, type=float, metavar='N', help='learning rate')
parser.add_argument('--latent_dim', default=32, type=int, metavar='N', help='dimension of latent space')
parser.add_argument('--seq_len', default=50, type=int, metavar='N', help='length of input sequene')
parser.add_argument('--load', action='store', type=str, required=False, help='Path from where to load network.')
parser.add_argument('--kf_steps', default=1, type=int, metavar='N', help='steps to wait before training alpha network')
parser.add_argument('--alpha', default='rnn', type=str, help='Alpha network type (mlp|rnn)')
parser.add_argument('--missing', action='store_true', help='Activate missing frames')
parser.add_argument('--corrupt', action='store_true', help='Activate corrupt data')
parser.add_argument('--model', default='kvae', type=str, help='Model type (ekvae|kvae|kglow|greparam)')

def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("/data2/users/cb221/stored_models_GLOW/", exist_ok=True)
    torch.save(state, "/data2/users/cb221/stored_models_GLOW/" + filename + '_latest.pth.tar')

def main():
    global args, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join(args.runs_path, args.name))
    print(args)
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    if args.missing:
        dl = MissingBallDataset(args.train_root)
    elif args.corrupt:
        dl = SquareBallDataset(args.train_root,
                               '/data2/users/hbz15/hmnist/mask_train')
    else:
        dl = BouncingBallDataLoader(args.train_root, images=True)
    train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True)
    sample = next(iter(train_loader)).float()
    _, _, input_dim, *_ = sample.size()
    variational = not args.determ
    # Load model
    if args.model=='kvae':
        kvae = KalmanVAE(input_dim=1, hidden_dim=32, obs_dim=2, 
                     latent_dim=4, num_modes=3, beta=args.beta, 
                     alpha=args.alpha).float().cuda()
    elif args.model=='kglow' or args.model=='greparam':
        kvae = KalmanVAE(input_dim=1, hidden_dim=32, obs_dim=1024, 
                     latent_dim=36, num_modes=3, beta=args.beta, 
                     alpha=args.alpha, mode=args.model).float().cuda()
    else:
        kvae = ExtendedKalmanVAE(input_dim=1, hidden_dim=32, obs_dim=4, 
                     latent_dim=12, beta=args.beta).float().cuda() 
    print(kvae)
    if args.load is not None:
        kvae.load_state_dict(torch.load(args.load)['kvae'])
        print("=> Model loaded successfully")

    # Set up optimizers
    if args.model=='kvae':
        kvae_params = list(kvae.encoder.parameters()) + list(kvae.decoder.parameters()) + [kvae.A, kvae.C, kvae.start_code]
        optimizer = Adam(kvae_params, lr=args.lr)
    else:
        optimizer = Adam(kvae.parameters(), lr=args.lr)
    gamma = 0.85
    scheduler = StepLR(optimizer, step_size=10, gamma=gamma)
    changed = False
    clip = 100
    # Train Loop
    kvae.train()
    for epoch in range(0, args.epochs):
        if epoch >= args.kf_steps and not changed and args.model=='kvae':
            changed = True
            optimizer = Adam(kvae.parameters(), lr=args.lr)
            scheduler = StepLR(optimizer, step_size=20, gamma=gamma)
        end = time.time()
        for i, sample in enumerate(train_loader, 1):
            # Forward sample to network
            b, seq_len, C, H, W = sample[:,:args.seq_len].size()
            var = (Variable(sample[:,:args.seq_len].float(), requires_grad=True).to(device) > 0.5).float()
            optimizer.zero_grad()
            #mask = torch.ones(b,seq_len).to(device)
            #mask[:,7:14] = 0
            if args.missing:
                mask = sample[1][:,:args.seq_len].cuda()
                x_hat, a_mu, _, losses = kvae(var, mask_frames=mask, variational=variational)
            elif args.corrupt:
                mask = sample[1][:,:args.seq_len].cuda()
                x_hat, a_mu, _, losses = kvae(var, mask_visual=mask, variational=variational)
            else:
                if args.model=='kglow' or args.model=='greparam':
                    input = (sample[:,:args.seq_len].to(device) > 0.5).float()
                    n_bins = 32
                    var = Variable(input - 0.5, requires_grad=True)
                    x_hat, a_mu, _, losses = kvae(var + torch.rand_like(var)/n_bins, variational=variational)
                    x_hat = ((x_hat + 0.5) > 0.5).float()
                    
                else:
                    mask = None
                    x_hat, a_mu, _, losses = kvae(var, variational=variational)
            # Compute loss and optimize params
            losses['loss'].backward()
            torch.nn.utils.clip_grad_norm_(kvae.parameters(), clip)
            optimizer.step()
            
            mse = F.mse_loss(x_hat, input, reduction='sum')/(b)
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()
            if i % 10 == 0:
                with torch.no_grad():
                    pred_pos, obs_seq, _ = kvae.predict_sequence(var + torch.rand_like(var)/n_bins, seq_len=args.seq_len)
                    target = (sample[:,args.seq_len:args.seq_len*2].to(pred_pos.device) > 0.5).float()
                    pred_pos = ((pred_pos + 0.5) > 0.5).float()
                    pred_mse = F.mse_loss(pred_pos, target, reduction='sum')/(args.batch_size)
                    writer.add_scalar('data/prediction_loss', pred_mse, i + epoch*len(train_loader))
                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}\t PRED: {pred:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=losses['elbo'], mse=mse, pred=pred_mse))
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                if variational:
                    writer.add_scalar('data/kl_loss', losses['kld'], i + epoch*len(train_loader))
                writer.add_scalar('data/elbo', losses['elbo'], i + epoch*len(train_loader))
                writer.add_scalar('data/loss', losses['loss'], i + epoch*len(train_loader))
            if i % 100 == 0:
                video_tensor_hat = x_hat.reshape((b, seq_len, C, H, W)).detach().cpu()
                video_tensor_true = sample[:,:args.seq_len].detach().cpu()
                writer.add_video('data/Inferred_vid',video_tensor_hat[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_vid',video_tensor_true[:16], i + epoch*len(train_loader))
                fig_inferred = plt.figure()
                ax1 = fig_inferred.add_subplot(1,1,1)
                a_mu = a_mu.detach().cpu()
                #obs_seq = obs_seq.detach().cpu()
                #real_pos = sample[1][0,:args.seq_len*2].double()
                #ax1.scatter(a_mu[0,:,0],a_mu[0,:,1])
                #ax1.scatter(obs_seq[0,:,0],obs_seq[0,:,1])
                #ax1.plot(real_pos[:,0],real_pos[:,1])
                video_tensor_predict = pred_pos.detach().cpu()
                video_tensor_predict_true = sample[:,args.seq_len:args.seq_len*2].detach().cpu()
                #writer.add_figure('data/inferred_latent_cont_state', fig_inferred, i + epoch*len(train_loader))
                writer.add_video('data/Predict_vid',video_tensor_predict[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_Future_vid',video_tensor_predict_true[:16], i + epoch*len(train_loader))

        scheduler.step()
        save_checkpoint({
            'epoch': epoch,
            'kvae': kvae.state_dict()
        }, filename=args.name)

if __name__=="__main__":
    main()