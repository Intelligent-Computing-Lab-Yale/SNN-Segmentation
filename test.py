#############################################
#   @author:                                #
#############################################

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.nn as nn
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
from matplotlib import pyplot as plt
from glob import glob

from utils import *
from setup import setup

def test(phase, f, config, args, testloader, model, state=None, epoch=0, max_miou=0, start_time=None, num_plot=16):

    if not start_time:
        start_time = datetime.datetime.now()

    with torch.no_grad():
        model.eval()

        gts, preds = [], []
        examples = None

        for batch_idx, (data, labels) in enumerate(testloader):

            if args.debug and (batch_idx + 1) != config.plot_batch:
                if phase == 'test':
                    f.write('Batch {} .................... skipped'.format(batch_idx + 1), end=('\r' if (batch_idx % 10) < 9 else '\n'), r_white=True)
            else:
                if config.gpu:
                    data = [datum.cuda() for datum in data] if config.thl else data.cuda()
                    labels = labels.cuda()
                
                outputs = model(data)
                pred = outputs.max(1,keepdim=True)[1].cpu().numpy()
                gt = labels.cpu().numpy()

                if (batch_idx + 1) == config.plot_batch:
                    temp2 = {}
                    temp2['data'] = data.squeeze().cpu().numpy()
                    temp2['preds'] = outputs.max(1,keepdim=True)[1].squeeze().cpu().numpy()
                    temp2['gts']   = labels.squeeze().cpu().numpy()
                    examples = zip(temp2['data'][:num_plot], temp2['preds'][:num_plot], temp2['gts'][:num_plot])

                gts.append(gt)
                preds.append(pred)

                if phase == 'test':
                    f.write('Batch {} .................... completed'.format(batch_idx + 1), end=('\r' if (batch_idx % 10) < 9 else '\n'), r_white=True)
            
            if args.debug and (batch_idx + 1) == config.plot_batch:
                break

            if phase == 'test':
                f.write('Evaluating progress: {:05.2f}% [Batch {:04d}/{:04d}]'.format(round((batch_idx + 1) / len(testloader) * 100, 2), batch_idx + 1, len(testloader)), end='\r')

    if phase == 'test' and args.count_spikes:
        f.write('Average total spikes per example per layer: {}'.format(model.spikes.average()))
        f.write('Average neuronal spike rate per example per layer: {}'.format(model.spikes.rate()))
        f.write('Neurons per layer: {}'.format(model.spikes.units))

        f.write('Average total spikes per example: {}'.format(model.spikes.totalAverage()))
        f.write('Average neuronal spike rate per example: {}'.format(model.spikes.totalRate()))

        label, value, title = "layer", "total spikes per example", "Total Spikes Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.average())]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"total_spikes_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        value, title = "neuronal spike rate per example", "Neuronal Spike Rate Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.rate())]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"spike_rate_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        value, title = "neurons", "Neurons Per Layer"
        data = [[i+1, val] for (i, val) in enumerate(model.spikes.units)]
        table = wandb.Table(data=data, columns=[label, value])
        wandb.log({"neurons_per_layer" : wandb.plot_table("itsjosh/vertical_bar_chart", table, {"label": label, "value": value}, {"title": title})}, step=epoch)

        wandb.log({"total_spikes": model.spikes.totalAverage(), "spike_rate": model.spikes.totalRate(), "total_neurons": model.spikes.totalUnits()}, step=epoch)

    score, class_iou, count = scores(gts, preds, config.dataset['num_cls'], config.batch_size_test, config.plot_batch, f)

    if score['Mean IoU'] > max_miou:
        max_miou = score['Mean IoU']
        wandb.run.summary["best_miou"] = max_miou

        if (not args.debug) and phase == 'train':

            state = {
                **state,
                'max_miou'              : max_miou,
                'epoch'                 : epoch,
                'state_dict'            : model.state_dict(),
            }

            # if config.conversion:
            #     keys = model.get_keys()
            #     parameters = {}
            #     for key in model.state_dict().keys():
            #         if key in keys.keys():
            #             parameters[keys[key]] = nn.Parameter(model.state_dict()[key].data)
            #     state = { **state, 'parameters' : parameters}

            filename = args.model_dir+config.identifier+'.pth'
            torch.save(state, filename)
            
            filename = os.path.join(wandb.run.dir, config.identifier+'.pth')
            torch.save(state, filename)

            # filename = os.path.join(wandb.run.dir, config.identifier+'.onnx')
            # torch.onnx.export(model, data, filename, export_params=True, opset_version=11)
        
        if phase == 'train':
            identifier = 'examples'
        elif phase == 'test':
            if config.attack:
                identifier = '{}_{}_examples'.format(config.attack, config.atk_factor)
            else:
                identifier = '{}_examples'.format('batch' + str(config.plot_batch))

        # Plot examples
        if args.plot:
            cnt = 0
            columns = 4
            plt.figure(figsize=(45,((30/32)*config.batch_size_test)))
            for i, ex in enumerate(examples):
                for j in range(len(ex)):
                    cnt += 1
                    plt.subplot(config.batch_size_test//columns,len(ex)*columns,cnt)
                    plt.xticks([], [])
                    plt.yticks([], [])
                    if args.plot_labels and i < columns:
                        if j == 0:
                            plt.title("input", fontsize=16)
                        elif j == 1:
                            plt.title("prediction", fontsize=16)
                        elif j == 2:
                            plt.title("ground truth", fontsize=16)
                    image = np.array(ex[j])
                    if j == 0:
                        if args.plot_labels:
                            plt.ylabel("{}/{}".format(i+1,config.batch_size_test), fontsize=16)
                        if config.dataset['name'] == 'voc2012':
                            image = image.transpose(1,2,0)
                            image = ((image*255) + np.array([104.00699, 116.66877, 122.67892])).astype(int)[:, :, ::-1]
                            plt.imshow(image)
                        elif config.dataset['name'] == 'ddd17':
                            image = image[1]
                            plt.imshow(np.clip(image, 0, 1), cmap='gray')
                    else:
                        if args.plot_labels:
                            np.set_printoptions(suppress=True, formatter={'float_kind':'{:0.0f}'.format}, linewidth=40)
                            plt.xlabel("{}".format(count[i][j-1]), fontsize=8)
                        if j == 1:
                            plt.imshow(np.squeeze(image).astype(np.uint8), cmap=(voc_cmap if config.dataset['name'] == 'voc2012' else 'viridis'), vmin=0, vmax=config.dataset['num_cls']-1)
                        elif j == 2:
                            if config.dataset['name'] == 'voc2012':
                                plt.imshow(np.squeeze(image).astype(np.uint8), cmap=voc_gt_cmap)
                            elif config.dataset['name'] == 'ddd17':
                                plt.imshow(np.squeeze(image).astype(np.uint8), cmap='viridis', vmin=0, vmax=config.dataset['num_cls']-1)
                        np.set_printoptions(suppress=False, formatter=None, linewidth=75)
            if args.plot_labels:
                plt.suptitle('{} batch {} examples'.format(config.identifier, config.plot_batch), fontsize=20)
                plt.subplots_adjust(top=0.97)

            wandb.log({identifier: plt}, step=epoch)
        else:
            mask_list = []
            for i, (data, pred, gt) in enumerate(examples):
                if config.dataset['name'] == 'voc2012':
                    data = data.transpose(1,2,0)
                    data = ((data*255) + np.array([104.00699, 116.66877, 122.67892])).astype(int)[:, :, ::-1]
                elif config.dataset['name'] == 'ddd17':
                    data = data[1]

                pred = np.squeeze(pred).astype(np.uint8)
                gt = np.squeeze(gt).astype(np.uint8)

                mask_list.append(wandb_mask(data, pred, gt, config.dataset['labels']))

            wandb.log({identifier: mask_list}, step=epoch)

    if phase == 'train':
        duration = datetime.timedelta(seconds=(datetime.datetime.now() - start_time).seconds)
        f.write('--------------- Evaluation -> miou: {:.3f}, best: {:.3f}, time: {}'.format(score['Mean IoU'], max_miou, duration))
        wandb.log({'miou': score['Mean IoU'], 'max_miou': max_miou, 'test_duration_mins': (duration.seconds / 60)}, step=epoch)

    return max_miou

if __name__ == '__main__':
    #--------------------------------------------------
    # Parse input arguments
    #--------------------------------------------------
    p = argparse.ArgumentParser(description='Evaluating ANN/SNN for semantic segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)

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
    p.add_argument('--model_path',      default='',                 type=str,       help='pretrained model path')
    p.add_argument('--conversion',      default=False, const=True,  type=str2bool,  help='use ann to snn conversion', nargs='?')

    # Dataset
    p.add_argument('--batch_size',      default=32,                 type=int,       help='Batch size')
    p.add_argument('--attack',          default='',                 type=str,       help='adversarial attack', choices=['saltpepper','gaussiannoise'])
    p.add_argument('--atk_factor',      default=None,               type=float,     help='Attack constant (sigma/p/scale)')

    # LIF neuron
    p.add_argument('--timesteps',       default=20,                 type=int,       help='Number of time-step')
    p.add_argument('--leak_mem',        default=0.99,               type=float,     help='Leak_mem')
    p.add_argument('--def_threshold',   default=1.0,                type=float,     help='Default membrane threshold')
    p.add_argument('--threshold_type',  default='layer',            type=str,       help='Threshold type', choices=['layer','channel','neuron'])
    p.add_argument('--scaling_factor',  default=0.7,                type=float,     help='scaling factor for thresholds')

    # Visualization
    p.add_argument('--plot',            default=False, const=True,  type=str2bool,  help='plot images', nargs='?')
    p.add_argument('--plot_batch',      default=1,                  type=int,       help='batch to plot')
    p.add_argument('--plot_labels',     default=True, const=True,   type=str2bool,  help='plot images with labels', nargs='?')
    p.add_argument('--see_model',       default=False, const=True,  type=str2bool,  help='see model structure', nargs='?')
    p.add_argument('--info',            default=True, const=True,   type=str2bool,  help='see training info', nargs='?')
    p.add_argument('--count_spikes',    default=False, const=True,  type=str2bool,  help='count spikes', nargs='?')

    # Dev tools
    p.add_argument('--debug',           default=False, const=True,  type=str2bool,  help='enable debugging mode', nargs='?')
    p.add_argument('--first',           default=False, const=True,  type=str2bool,  help='only debug first epoch and first ten batches', nargs='?')
    p.add_argument('--print_models',    default=False, const=True,  type=str2bool,  help='only print available trained models', nargs='?')
    p.add_argument('--reset_thresholds',default=False, const=True,  type=str2bool,  help='find new thresholds for this number of timesteps', nargs='?')

    global args
    args = p.parse_args()

    #--------------------------------------------------
    # Initialize arguments
    #--------------------------------------------------

    if args.augment and args.attack:
        raise RuntimeError('You can\'t use the --augment command with the --attack command')

    if args.attack and (not args.atk_factor):
        raise RuntimeError('You must provide an attack (sigma/p/scale) constant with the --attack command')

    if args.print_models and args.model_path:
        raise RuntimeError('You can\'t use the --model_path command with the --print_models command')


    if args.model_path and args.model_path.isdigit():
        args.model_path = int(args.model_path)

    if isinstance(args.model_path, str) and args.model_path:
        args.model_path = (args.model_dir + args.model_path)
    else:
        pretrained_models = sorted(glob(args.model_dir + '*.pth'))
        val = args.model_path
        if not val and val != 0:
            print('---- Trained models ----')
            for i, model in enumerate(pretrained_models):
                print('{}: {}'.format(i, model[17:]))
            if args.print_models:
                exit()
            val = int(input('\n Which model do you want to use? '))
            while (val < 0) or (val >= len(pretrained_models)):
                print('That index number is not accepted. Please input one of the index numbers above.')
                val = int(input('\n Which model do you want to use? '))
        args.model_path = pretrained_models[val]
    print(args.model_path)


    #--------------------------------------------------
    # Setup
    #--------------------------------------------------
    factor = 'no_atk' if args.atk_factor == None else args.atk_factor
    if args.attack:
        args.file_name = args.attack + '-' + str(factor)

    run, f, config, testloader, model, now = setup('test', args)

    with run:
        #--------------------------------------------------
        # Evaluate the model
        #--------------------------------------------------
        f.write('********** ({}) {} evaluation **********'.format(factor, config.model_type.upper()))
        max_miou = test('test', f, config, args, testloader, model)

    duration = datetime.timedelta(days=(datetime.datetime.now() - now).days, seconds=(datetime.datetime.now() - now).seconds)
    f.write('({}) Mean IoU: {:.6f}'.format(factor, max_miou), r_white=True)
    f.write('({}) Run time: {}'.format(factor, duration))

    sys.exit(0)