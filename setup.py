"""

Setup an environment for training or testing for semantic segmentation.

@author: Joshua Chough

"""

#--------------------------------------------------
# Imports
#--------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader

import wandb

import sys
import os
import datetime
import numpy as np

from utils import *
from models import *
from data import *

#--------------------------------------------------
# Setup function
#--------------------------------------------------
def setup(phase, args):
    # Initialize seed
    seed = args.seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    #--------------------------------------------------
    # Initialize configuration parameters
    #--------------------------------------------------
    now = datetime.datetime.now() # current date and time
    date_time = now.strftime('%y%m%d-%H%M%S')
    date = now.strftime('%y%m%d')

    try:
        os.mkdir(args.model_dir)
    except OSError:
        pass

    if phase == 'train':
        # Training parameters
        config = dict(
            # Processing
            seed                = args.seed,
            num_workers         = args.num_workers,
            gpu                 = args.gpu if torch.cuda.is_available() else False,
            # Model
            model_path          = None,
            conversion          = None,
            model_type          = args.model_type,
            architecture        = args.arch,
            kernel_size         = args.kernel_size,
            pretrained          = args.pretrained,
            bn                  = args.bn,
            full_prop           = (args.full_prop if args.model_type == 'snn' else None),
            # Dataset
            dataset             = dataset_cfg[args.dataset],
            batch_size          = args.batch_size,
            batch_size_test     = args.batch_size*2,
            img_size            = (img_sizes[args.dataset] if args.img_size == -1 else (args.img_size, args.img_size)),
            augment             = args.augment,
            thl                 = (args.thl if args.model_type == 'snn' else None),
            attack              = None,
            atk_factor          = None,
            # Learning
            epochs              = args.epochs,
            lr                  = args.lr,
            optimizer           = args.optimizer,
            # LIF neuron
            timesteps           = (args.timesteps if args.model_type == 'snn' else None),
            leak_mem            = (args.leak_mem if args.model_type == 'snn' else None),
            def_threshold       = (args.def_threshold if args.model_type == 'snn' else None),
            threshold_type      = None,
            leaky               = (args.leaky if args.model_type == 'ann' else None),
            alpha               = (args.alpha if (args.model_type == 'ann' and args.leaky) else None),
            ibt                 = None,
            scaling_factor      = None,
            # Visualization
            plot_batch          = args.plot_batch,
            count_spikes        = None,
        )
    elif phase == 'test':
        # Testing parameters
        model_path = args.model_path
        conversion = True if (args.conversion or ('conversion' in model_path)) else False

        # Use configuration parameters from pretrained model
        state = torch.load(model_path, map_location='cpu')
        old_config = state['config']

        if conversion and old_config['model_type'] == 'snn':
            raise RuntimeError('You can only do conversion using a pretrained ANN model. Please use --model_path with an ANN path')

        model_type = ('snn' if conversion else old_config['model_type'])

        config = dict(
            # Processing
            seed                = args.seed,
            num_workers         = args.num_workers,
            gpu                 = args.gpu if torch.cuda.is_available() else False,
            # Model
            model_path          = model_path,
            conversion          = conversion,
            model_type          = model_type,
            architecture        = old_config['architecture'],
            kernel_size         = old_config['kernel_size'],
            pretrained          = (None if conversion else old_config['pretrained']),
            bn                  = old_config['bn'],
            full_prop           = old_config['full_prop'],
            # Dataset
            dataset             = old_config['dataset'],
            batch_size          = (args.batch_size if conversion else None),
            batch_size_test     = args.batch_size,
            img_size            = old_config['img_size'],
            augment             = None,
            thl                 = old_config['thl'],
            attack              = (args.attack if args.attack else False),
            atk_factor          = (args.atk_factor if (args.atk_factor or args.atk_factor == 0) else False),
            # Learning
            epochs              = None,
            lr                  = None,
            optimizer           = None,
            # LIF neuron
            timesteps           = (args.timesteps if conversion else old_config['timesteps']),
            leak_mem            = (args.leak_mem if conversion else old_config['leak_mem']),
            def_threshold       = (args.def_threshold if conversion else old_config['def_threshold']),
            threshold_type      = (args.threshold_type if conversion else None),
            leaky               = (None if conversion else old_config['leaky']),
            alpha               = old_config['alpha'],
            ibt                 = (True if (conversion and old_config['leaky']) else False if model_type == 'snn' else None),
            scaling_factor      = (args.scaling_factor if conversion else None),
            # Visualization
            plot_batch          = args.plot_batch,
            count_spikes        = args.count_spikes,
        )

    #--------------------------------------------------
    # Initialize wandb settings
    #--------------------------------------------------
    # Generate tags
    tags = []
    if (args.debug):
        tags += ['development']
    else:
        tags += ['production']
    if phase == 'test':
        if config['conversion']:
            tags += ['conversion']
        if config['count_spikes']:
            tags += ['count spikes']
    if config['attack']:
        tags += ['attack']

    # Start a run, tracking hyperparameters
    run = wandb.init(
        project=args.project,
        group=date,
        job_type=phase,
        reinit=True,
        tags=tags,
        force=True,
        config=config,
        mode=args.wandb_mode
    )

    # Generate model identifier
    identifier = createIdentifier((date, run.name, wandb.config.model_type, wandb.config.architecture, wandb.config.dataset['name'], args.file_name))
    wandb.config.update({'identifier': identifier})

    config = wandb.config

    # Use a wrapper for printing
    f = File(False)

    # Display run information
    if (args.debug):
        f.write('------------ D E V E L O P M E N T   M O D E -------------', start='\n', end='\n\n')
    f.write('Run on time: {}'.format(now))
    f.write('Identifier: {}'.format(config.identifier))
    if phase == 'test':
        f.write('Pretrained {}: {}'.format(config.model_type.upper(), args.model_path))
        if config.conversion:
            f.write('==== Converting ANN -> SNN [{}-wise thresholding] ===='.format(config.threshold_type), terminal=True)
    
    # Display parameters
    if args.info:
        f.write('=== [{}] CONFIGURATION ==='.format(run.name), start='\n')
        for key in config.keys():
            if key == 'dataset':
                f.write('\t {:20} : {}'.format(key, getattr(config, key)['name']))
            else:
                value = getattr(config, key)
                if value != None:
                    f.write('\t {:20} : {}'.format(key, value))

    #--------------------------------------------------
    # Load dataset
    #--------------------------------------------------
    # Create dataloaders from custom datasets for training and/or testing
    if config.dataset['name'] == 'voc2012':
        if phase == 'train' or conversion:
            train_dataset = VOC2012(config.dataset['path'], split="train_aug", is_transform=True, img_size=config.img_size)
            trainloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        test_dataset = VOC2012(config.dataset['path'], split="val", is_transform=True, img_size=config.img_size, attack=config.attack, atk_factor=config.atk_factor)
        testloader = DataLoader(test_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers, shuffle=False)
    elif config.dataset['name'] == 'ddd17':
        if phase == 'train' or conversion:
            train_dataset = DDD17(config.dataset['path'], split="train", is_transform=True, is_augment=config.augment, img_size=config.img_size, mod=True, thl=config.thl, thl_size=config.timesteps)
            trainloader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
        test_dataset = DDD17(config.dataset['path'], split="test", is_transform=True, is_augment=False, attack=config.attack, img_size=config.img_size, mod=True, thl=config.thl, thl_size=config.timesteps, atk_factor=config.atk_factor)
        testloader = DataLoader(test_dataset, batch_size=config.batch_size_test, num_workers=config.num_workers, shuffle=False)
    else:
        raise RuntimeError("dataset not valid..")

    # Display dataset stats
    if phase == 'train' or config.conversion:
        f.write('loaded {} train split [{} samples]'.format(config.dataset['name'], (len(trainloader)*config.batch_size)))
    f.write('loaded {} test split [{} samples]'.format(config.dataset['name'], (len(testloader)*config.batch_size_test)))

    #--------------------------------------------------
    # Instantiate the model and optimizer
    #--------------------------------------------------
    if config.model_type == 'snn':
        model = SNN_VGG(config=config)
    elif config.model_type == 'ann':
        model = ANN_VGG(config=config)
    else:
        raise RuntimeError("architecture not valid..")

    if config.gpu:
        model = model.cuda()

    if args.see_model:
        f.write(model)

    if phase == 'test':
        # Load weights from pretrained model
        state = torch.load(args.model_path, map_location='cpu')
        model.load_state_dict(state['state_dict'], strict=False)

        # If using ANN/SNN conversion, load or find the maximum activation thresholds (threshold normalization)
        if config.conversion:
            if (not args.reset_thresholds) and ('thresholds' in state.keys()) and (config.threshold_type in state['thresholds'].keys()) and (str(config.timesteps) in state['thresholds'][config.threshold_type].keys()):
                # If thresholds present in loaded ANN file, load thresholds
                thresholds = state['thresholds'][config.threshold_type][str(config.timesteps)]
                f.write('Loaded {} thresholds ({}) from {}'.format(config.threshold_type, config.timesteps, args.model_path))
                model.threshold_update(scaling_factor=config.scaling_factor, thresholds=thresholds[:], threshold_type=config.threshold_type)
            else:
                # If thresholds not present in loaded ANN file, find thresholds
                thresholds = find_thresholds(f, trainloader, model, config)
                model.threshold_update(scaling_factor=config.scaling_factor, thresholds=thresholds[:], threshold_type=config.threshold_type)
                
                # Save the thresholds in the ANN file
                if ('thresholds' not in state.keys()) or (not isinstance(state['thresholds'], dict)):
                    state['thresholds'] = {}
                if (config.threshold_type not in state['thresholds'].keys()) or (not isinstance(state['thresholds'][config.threshold_type], dict)):
                    state['thresholds'][config.threshold_type] = {}
                state['thresholds'][config.threshold_type][str(config.timesteps)] = thresholds
                torch.save(state, args.model_path)
                f.write('Saved {} thresholds ({}) in {}'.format(config.threshold_type, config.timesteps, args.model_path))

    # For training, configure the loss function, optimizer, and learning rate scheduler
    if phase == 'train':
        criterion = nn.CrossEntropyLoss()

        if config.optimizer == 'adam':
            optimizer = optim.Adam(model.parameters(), lr=config.lr, amsgrad=True, weight_decay=5e-4)
        elif config.optimizer == 'sgd':
            optimizer = optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=5e-4)
        else:
            raise RuntimeError("optimizer not valid..")

        milestones = [int(milestone*config.epochs) for milestone in [0.5, 0.8]]
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)

    #--------------------------------------------------
    # Prepare state objects
    #--------------------------------------------------
    # Prepare state to be saved with trained model
    if phase == 'train':
        state = {
            'config': config.as_dict()
        }

    if phase == 'train':
        return run, f, config, trainloader, testloader, model, criterion, optimizer, scheduler, now, state
    elif phase == 'test':
        return run, f, config, testloader, model, now