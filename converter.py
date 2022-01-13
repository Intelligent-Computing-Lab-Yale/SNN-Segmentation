import torch
import torch.nn as nn
from glob import glob

model_dir = './trained_models/'

pretrained_models = sorted(glob(model_dir + '*.pth'))
print('---- Trained models ----')
for i, model in enumerate(pretrained_models):
    print('{}: {}'.format(i, model[17:]))
val = int(input('\n Which model do you want to use? '))
while (val < 0) or (val >= len(pretrained_models)):
    print('That index number is not accepted. Please input one of the index numbers above.')
    val = int(input('\n Which model do you want to use? '))
pretrained_model = pretrained_models[val]

state = torch.load(pretrained_model, map_location='cpu')

options = {
    'mode': ['convert state_dict', 'convert thresholds']
}

# -----------------
# SETUP
# -----------------
args = {}
for i, key in enumerate(options.keys()):
    if input('{} [{}] (type \'c\' to change)? '.format(key, options[key][0])) == 'c':
        for j, option in enumerate(options[key]):
            print('{}: {}'.format(j, option))
        args[key] = options[key][int(input('Which {}? '.format(key)))]
        print('Changed {} to {}'.format(key, args[key]), end='\n\n')
    else:
        args[key] = options[key][0]
print('{}'.format(args))

if args['mode'] == 'convert state_dict':

    state_dict = state['state_dict']

    print('\n ---------- Keys for {} state_dict ----------'.format(pretrained_model[17:]))
    for key, value in state_dict.items():
        if (('features' in key) or ('classifier' in key)) and ('weight' in key) and ('bn_features' not in key) and ('bn_classifier' not in key):
            print('{}: {}'.format(key, value.shape))

    temp_features = []
    temp_classifier = []
    for key, value in state_dict.items():
        if ('features' in key) and ('weight' in key) and ('bn_features' not in key):
            temp_features.append(nn.Parameter(value))
        elif ('classifier' in key) and ('weight' in key) and ('bn_classifier' not in key):
            temp_classifier.append(nn.Parameter(value))

    print('\n{} feature layers weights were retrieved'.format(len(temp_features)))
    print('{} classifier layers weights were retrieved'.format(len(temp_classifier)))

    state_dict = {}

    for i, param in enumerate(temp_features):
        state_dict['features.{}.weight'.format(i)] = param
    for i, param in enumerate(temp_classifier):
        state_dict['classifier.{}.weight'.format(i)] = param

    print('\n{} layers weights were compiled'.format(len(state_dict)))

    print('\n ---------- Keys for conversion state_dict ----------')
    for key, value in state_dict.items():
        print('{}'.format(key))

    state['state_dict'] = state_dict

elif args['mode'] == 'convert thresholds':

    if 'thresholds' not in state.keys():
        print('\'thresholds\' not in state')
        exit()

    print('\n ---------- Keys for thresholds ----------')
    for key, value in state['thresholds'].items():
        print('{}'.format(key))

    threshold_type = input('Threshold type: ')

    if threshold_type not in state['thresholds'].keys():
        thresholds = {}
        thresholds[threshold_type] = state['thresholds']
        state['thresholds'] = thresholds

    print('\n ---------- Keys for thresholds ----------')
    for key, value in state['thresholds'].items():
        print('{}'.format(key))
    
    print('\n ---------- Keys for {} thresholds ----------'.format(threshold_type))
    for key, value in state['thresholds'][threshold_type].items():
        print('{}'.format(key))

print('\n ---------- Keys for conversion state ----------')
for key, value in state.items():
    print('{}'.format(key))

if input('Change file name [{}] (c to change)? '.format(pretrained_model)) == 'c':
    path = pretrained_model[:-4] + '_' + input('Name of new file: ') + '.pth'
else:
    path = pretrained_model

torch.save(state, path)

print('\nSaved state in {}'.format(path))
