# Imports
import argparse
import os
import time
from matplotlib import image

import numpy as np

from tensorboardX import SummaryWriter

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader
from torch.distributions import Bernoulli

from dataloaders.bouncing_data import BouncingBallDataLoader, SquareBallDataset, MissingBallDataset
from dataloaders.nba_data import NBADataset
from utils.losses import kld_loss, nll_gaussian_var_fixed
from models.VRNN import VRNN

parser = argparse.ArgumentParser(description='VRNN trainer')

parser.add_argument('--name', required=True, type=str, help='Name of the experiment')
parser.add_argument('--train_root', default='/dataset/train', type=str)
parser.add_argument('--train_idx', default='', type=str)
parser.add_argument('--epochs', default=500, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('--step_size', default=150, type=int, metavar='N', help='number of total epochs to run')
parser.add_argument('-b', '--batch-size', default=128, type=int,metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--beta', default=1, type=float,metavar='N', help='beta VAE param')
parser.add_argument('--lr', default=5e-4, type=float, metavar='N', help='learning rate')
parser.add_argument('--hidden_dim', default=256, type=int, metavar='N', help='dimension of hidden space')
parser.add_argument('--latent_dim', default=64, type=int, metavar='N', help='dimension of latent space')
parser.add_argument('--seq_len', default=100, type=int, metavar='N', help='length of input sequene')
parser.add_argument('--load', action='store', type=str, required=False, help='Path from where to load network.')
parser.add_argument('--visual', action='store_true', help='Use image video as input')
parser.add_argument('--missing', action='store_true', help='Activate missing frames')
parser.add_argument('--corrupt', action='store_true', help='Activate corrupt data')
parser.add_argument('--experiment', default='NBA', type=str)


def get_device(cuda=True):
    return 'cuda' if cuda and torch.cuda.is_available() else 'cpu'

def save_checkpoint(state, filename='model'):
    os.makedirs("/data2/users/cb221/stored_models_NBA_ICLR/", exist_ok=True)
    torch.save(state, "/data2/users/cb221/stored_models_NBA_ICLR/" + filename + '_latest.pth.tar')

def main():
    global args, writer
    args = parser.parse_args()
    writer = SummaryWriter(log_dir=os.path.join("/data2/users/cb221/runs_NBA_VRNN", args.name))
    print(args)
    # Set up writers and device
    device = get_device()
    print("=> Using device: " + device)
    # Load dataset
    if args.experiment=='NBA' or args.experiment=='NBACond':
        dl = NBADataset(args.train_root, None if args.train_idx=='' else args.train_idx)
    elif args.missing:
        dl = MissingBallDataset(args.train_root)
    elif args.corrupt:
        dl = SquareBallDataset(args.train_root,
                               '/data2/users/hbz15/2_body_black_white_real_real/mask_train')
    else:
        dl = BouncingBallDataLoader(args.train_root, images=False)
    train_loader = DataLoader(dl, batch_size=args.batch_size, shuffle=True, num_workers=4)
    # Load model
    vrnn = VRNN(1, 40, args.hidden_dim, args.latent_dim, num_rec_layers=3, input_type='visual' if args.visual else 'base', ball_cond=args.experiment=='NBACond').float().to(device)
    print(vrnn)
    if args.load is not None:
        vrnn.load_state_dict(torch.load(args.load)['vrnn'])
        print("=> Model loaded successfully")

    # Set up optimizers
    optimizer = Adam(vrnn.parameters(), lr=args.lr)
    gamma = 0.5
    scheduler = StepLR(optimizer, step_size=args.step_size, gamma=gamma)
    changed = False
    # Train Loop
    vrnn.train()
    best_pred_mse = 1e4
    for epoch in range(0, args.epochs):
        pos = None
        if epoch >= 5 and not changed:
            changed = True
            args.beta = 1
        end = time.time()
        for i, sample in enumerate(train_loader, 1):
            if args.visual:
                im = sample[:,:args.seq_len].float()
                (B, T, ch, H, W) = im.size()
                # Forward sample to network
                var = (Variable(im, requires_grad=True).to(device) > 0.5).float()
            else:

                (B, T, *_) = sample[:,:args.seq_len].size()

                if args.experiment=='NBACond':
                    input = sample[:,:args.seq_len,1:][:,:,:,[0,1,3,4]].reshape(B,T,-1)
                else:
                    input = sample[:,:args.seq_len,:][:,:,:,[0,1,3,4]].reshape(B,T,-1)
                var = Variable(input.float(), requires_grad=True).to(device)
            optimizer.zero_grad()
            if args.experiment=='NBA' or args.experiment=='NBACond':
                if args.experiment=='NBACond':
                    ball_coord = sample[:,:args.seq_len,0,:3].float().to(device)
                else:
                    ball_coord = None
                reconstr_seq, z_params, z_params_prior = vrnn(var, ball_coord=ball_coord)
                kld = kld_loss(z_params[:,:,0,:], z_params[:,:,1,:], z_params_prior[:,:,0,:], z_params_prior[:,:,1,:])
                mse = (F.mse_loss(reconstr_seq, var, reduction='none').reshape(B,T,-1).sum(-1)).sum()/B
                nll = nll_gaussian_var_fixed(reconstr_seq, var, variance=1e-3)
            else:
                if args.missing:
                    mask_visual = torch.ones(B, T, ch, H, W).to(var.device)
                    mask_frames = sample[1][:,:args.seq_len].to(var.device).float()
                    reconstr_seq, z_params, z_params_prior = vrnn(var, 
                                                        mask_frames=mask_frames.reshape(B,T,1,1,1))
                elif args.corrupt:
                    mask_visual = sample[1][:,:args.seq_len].to(var.device).float()
                    mask_frames = torch.ones(B,T).to(var.device)
                    reconstr_seq, z_params, z_params_prior = vrnn(var)
                else:
                    mask_visual = torch.ones(B, T, ch, H, W).to(device)
                    mask_frames = torch.ones(B,T).to(var.device)
                    reconstr_seq, z_params, z_params_prior = vrnn(var)
                # Compute loss and optimize params
                kld = kld_loss(z_params[:,:,0,:], z_params[:,:,1,:], z_params_prior[:,:,0,:], z_params_prior[:,:,1,:])
                mse = (F.mse_loss(reconstr_seq, var, reduction='none').reshape(B,T,-1).sum(-1)*mask_frames).sum()/B
                decoder_x = Bernoulli(reconstr_seq)
                p_x = (decoder_x.log_prob(var)*mask_visual).reshape(B,T,-1).sum(-1)
                nll = -(p_x*mask_frames).mean(dim=0).sum()
            loss = args.beta*kld + nll
            elbo = kld + nll
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vrnn.parameters(), 100)
            optimizer.step()
            
            
            # Measure elapsed time
            batch_time = time.time() - end
            end = time.time()

            if i % 10 == 0:
                writer.add_scalar('data/mse_loss', mse, i + epoch*len(train_loader))
                writer.add_scalar('data/kl_loss', kld, i + epoch*len(train_loader))
                writer.add_scalar('data/total_loss', loss, i + epoch*len(train_loader))
                writer.add_scalar('data/elbo', elbo, i + epoch*len(train_loader))
                with torch.no_grad():
                    if args.experiment=='NBACond':
                        ball_coord = sample[:,:,0,:3].float().to(device)
                    else:
                        ball_coord = None
                    pred_pos = vrnn.predict_sequence(var, ball_coord=ball_coord)
                    if args.visual:
                        target = sample[:,args.seq_len:args.seq_len*2].float().to(pred_pos.device)
                    else:
                        if args.experiment=='NBACond':
                            target = sample[:,args.seq_len:args.seq_len*2,1:][:,:,:,[0,1,3,4]].reshape(B,T,-1)
                        else:
                            target = sample[:,args.seq_len:args.seq_len*2][:,:,:,[0,1,3,4]].reshape(B,T,-1)
                        target = target.float().to(pred_pos.device)
                    pred_mse = F.mse_loss(pred_pos, target, reduction='sum')/(args.batch_size)
                    writer.add_scalar('data/prediction_loss', pred_mse, i + epoch*len(train_loader))

                print('Epoch: [{0}][{1}/{2}]\t'
                    'Time {batch_time:.3f}\t'
                    'Loss {loss:.4e}\t MSE: {mse:.4e}\t PRED: {pred:.4e}'.format(
                    epoch, i, len(train_loader), batch_time=batch_time, loss=loss, mse=mse, pred=pred_mse))
            if i % 100 == 0 and args.visual:
                b, seq_len, C, H, W = sample[:,:args.seq_len].size()
                video_tensor_hat = reconstr_seq.reshape((b, seq_len, C, H, W)).detach().cpu()
                video_tensor_true = im.float().detach().cpu()
                video_tensor_predict = pred_pos.detach().cpu()
                video_tensor_predict_true = sample[:,args.seq_len:].float().detach().cpu()
                writer.add_video('data/Inferred_vid',video_tensor_hat[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_vid',video_tensor_true[:16], i + epoch*len(train_loader))
                writer.add_video('data/Predict_vid',video_tensor_predict[:16], i + epoch*len(train_loader))
                writer.add_video('data/True_Future_vid',video_tensor_predict_true[:16], i + epoch*len(train_loader))
        scheduler.step()
        pred_mse = F.mse_loss(pred_pos, target, reduction='sum')/(args.batch_size)
        if best_pred_mse > pred_mse:
            best_pred_mse = pred_mse
            save_checkpoint({
            'epoch': epoch,
            'vrnn': vrnn.state_dict()
            }, filename=args.name+'best_pred')
        save_checkpoint({
            'epoch': epoch,
            'vrnn': vrnn.state_dict()
        }, filename=args.name)


if __name__=="__main__":
    main()