#############################################
#   @author:                                #
#############################################

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.backends.cudnn as cudnn

cudnn.enabled = False
cudnn.benchmark = True
cudnn.deterministic = True

import wandb

import argparse
import sys
import os
import datetime
import numpy as np

from utils import *
from setup import setup
from test import test

#--------------------------------------------------
# Parse input arguments
#--------------------------------------------------
p = argparse.ArgumentParser(description='Training ANN/SNN for semantic segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

# Processing
p.add_argument('--seed',            default=0,                  type=int,       help='Random seed')
p.add_argument('--num_workers',     default=4,                  type=int,       help='number of workers')
p.add_argument('--gpu',             default=True, const=True,   type=str2bool,  help='use gpu', nargs='?')

# Wandb and file
p.add_argument('--wandb_mode',      default='online',           type=str,       help='wandb mode', choices=['online','offline','disabled'])
p.add_argument('--project',         default='snn-seg',          type=str,       help='project name')
p.add_argument('--file_name',       default='',                 type=str,       help='Add-on for the file name')
p.add_argument('--model_dir',       default='./trained_models/',type=str,       help='Directory for trained models')

# Model
p.add_argument('--model_type',      default='snn',              type=str,       help='model type', choices=['ann','snn'])
p.add_argument('--arch',            default='dl-vgg9',          type=str,       help='architecture', choices=['dl-vgg9','fcn-vgg9','dl-vgg11','dl-vgg16'])
p.add_argument('--kernel_size',     default=3,                  type=int,       help='filter size for the conv layers')
p.add_argument('--pretrained',      default=True, const=True,   type=str2bool,  help='use pretrained parameters from torchvision if possible', nargs='?')
p.add_argument('--bn',              default=True, const=True,   type=str2bool,  help='batch norm', nargs='?')
p.add_argument('--full_prop',       default=False, const=True,  type=str2bool,  help='full propagation', nargs='?')

# Dataset
p.add_argument('--dataset',         default='voc2012',          type=str,       help='dataset', choices=['voc2012','ddd17'])
p.add_argument('--batch_size',      default=16,                 type=int,       help='Batch size')
p.add_argument('--img_size',        default=64,                 type=int,       help='Image size')
p.add_argument('--augment',         default=False, const=True,  type=str2bool,  help='turn on data augmentation', nargs='?')
p.add_argument('--thl',             default=False, const=True,  type=str2bool,  help='target history learning', nargs='?')

# Learning
p.add_argument('--epochs',          default=60,                 type=int,       help='Number of epochs')
p.add_argument('--lr',              default=1e-3,               type=float,     help='Learning rate')
p.add_argument('--optimizer',       default='sgd',              type=str,       help='optimizer', choices=['adam','sgd'])

# LIF neuron
p.add_argument('--timesteps',       default=20,                 type=int,       help='Number of time-step')
p.add_argument('--leak_mem',        default=0.99,               type=float,     help='Leak_mem')
p.add_argument('--def_threshold',   default=1.0,                type=float,     help='Default membrane threshold')
p.add_argument('--leaky',           default=False, const=True,  type=str2bool,  help='use non-leaky or leaky ReLUs', nargs='?')
p.add_argument('--alpha',           default=0.1,                type=float,     help='leaky ReLU alpha')

# Visualization
p.add_argument('--plot',            default=False, const=True,  type=str2bool,  help='plot images', nargs='?')
p.add_argument('--plot_batch',      default=1,                  type=int,       help='batch to plot')
p.add_argument('--train_display',   default=1,                  type=int,       help='freq (in epochs) for printing training progress')
p.add_argument('--test_display',    default=10,                 type=int,       help='freq (in epochs) for evaluating model during training')
p.add_argument('--see_model',       default=False, const=True,  type=str2bool,  help='see model structure', nargs='?')
p.add_argument('--info',            default=True, const=True,   type=str2bool,  help='see training info', nargs='?')

# Dev tools
p.add_argument('--debug',           default=False, const=True,  type=str2bool,  help='enable debugging mode', nargs='?')
p.add_argument('--first',           default=False, const=True,  type=str2bool,  help='only debug first epoch and first ten batches', nargs='?')

global args
args = p.parse_args()

if args.first and (not args.debug):
    raise RuntimeError('You must run the --first command with --debug')

if args.full_prop and (args.model_type != 'snn'):
    raise RuntimeError('You can only use the --full_prop command with the SNN model')

if args.thl and ((args.dataset != 'ddd17') or (args.model_type != 'snn')):
    raise RuntimeError('You can only use the --thl command with the SNN model and DDD17 dataset')

#--------------------------------------------------
# Setup
#--------------------------------------------------
run, f, config, trainloader, testloader, model, criterion, optimizer, scheduler, now, state = setup('train', args)

with run:
    # tell wandb to watch what the model gets up to
    wandb.watch(model, log='all', log_freq=10)

    #--------------------------------------------------
    # Train the using surrogate gradients
    #--------------------------------------------------
    f.write('********** {} training and evaluation **********'.format(config.model_type))
    max_miou = 0
    min_loss = 10
    wandb.run.summary["best_miou"] = max_miou
    wandb.run.summary["best_loss"] = min_loss

    for epoch in range(1, config.epochs+1):
        train_loss = AverageMeter()
        model.train()
        start_time = datetime.datetime.now()

        for batch_idx, (data, labels) in enumerate(trainloader):
            
            if config.gpu:
                data = [datum.cuda() for datum in data] if config.thl else data.cuda()
                labels = labels.cuda()

            optimizer.zero_grad()
            outputs = model(data)

            loss = criterion(outputs, labels)
            train_loss.update(loss.item(), labels.size(0))

            loss.backward()
            optimizer.step()

            if args.debug:
                f.write('Epoch {:03d}/{:03d} | Batch progress: {:05.2f}% [{:04d}/{:04d}]'.format(epoch, config.epochs, round((batch_idx+1)/len(trainloader)*100, 2), batch_idx+1, len(trainloader)), end='\r')

            if args.first and batch_idx == 9:
                break
        
        if args.first:
            print('')

        scheduler.step()

        if args.debug or ((epoch) % args.train_display == 0):
            duration = datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
            f.write('Training progress: {:05.2f}% [Epoch {:03d}/{:03d}] | Loss: {:.6f} | LR: {:.6f} [{}]'.format(round(((epoch/config.epochs)*100), 2), epoch, config.epochs, train_loss.avg, optimizer.param_groups[0]['lr'], duration))
            wandb.log({'epoch': epoch, 'loss': train_loss.avg, 'lr': optimizer.param_groups[0]['lr'], 'train_duration_mins': (duration.seconds / 60)}, step=epoch)

            if train_loss.avg < min_loss:
                min_loss = train_loss.avg
                wandb.run.summary['best_loss'] = min_loss

        if args.first or ((epoch) % args.test_display == 0):
            max_miou = test('train', f, config, args, testloader, model, state, epoch, max_miou, start_time=start_time)

        if args.first and epoch == 1:
            break

f.write('Highest mean IoU: {:.6f}'.format(max_miou))
f.write('Total script time: {}'.format(datetime.timedelta(days=(datetime.datetime.now() - now).days, seconds=(datetime.datetime.now() - now).seconds)))

sys.exit(0)