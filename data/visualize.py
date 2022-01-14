"""

Visualize dataset examples.

@author: Joshua Chough

"""

from voc2012 import *
from ddd17 import *

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from mpl_toolkits.axes_grid1 import make_axes_locatable
import os
from math import ceil
import decimal

def dinterval(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)

def drange(x, y, jump):
    return list(dinterval(x, y, str(jump)))

'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
voc_palette = [(0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128), (192,128,128), (0, 64,0), (128, 64, 0), (0,192,0), (128,192,0), (0,64,128)]
voc_palette = [[r/255, g/255, b/255, 1] for (r, g, b) in voc_palette]

def visualize(dataset, data, mode, indexes, img_bounds, cmap, colorbar):
    cols = 6 if len(indexes) > 6 else len(indexes)
    rows = int(ceil(len(indexes)/cols))

    for i in indexes:
        ax = plt.subplot(rows, cols, i+1)
        # ax.set_title('frame {}'.format(i))
        ax.axis('off')

        if mode == 'visualize images':
            img = data[i][0][0] if dataset == 'DDD17' else data[i][0]
        elif mode == 'visualize ground truth':
            img = data[i][1]

            if cmap == 'voc2012':
                cmap = ListedColormap(voc_palette)

            print(i, np.unique(img))
        
        mappable = ax.imshow(img, cmap=cmap, vmin=img_bounds[0], vmax=img_bounds[1])

        if colorbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            ticks = drange(img_bounds[0], (img_bounds[1] + (1 if type(img_bounds[1]) is int else 0.1)), int((abs(img_bounds[0]) + abs(img_bounds[1]))/5))
            plt.colorbar(mappable, cax=cax, ticks=ticks)

if __name__ == '__main__':
    options = {
        'dataset': ['VOC2012', 'DDD17'],
        'split': ['train_aug', 'train', 'val', 'test'],
        'mode': ['visualize images', 'visualize ground truth'],
        'img_size': [64, 128, 256, 512],
        'img_bounds': ['0,20', '-0.5,0.5', '0,5', '0,255'],
        'cmap': ['viridis', 'voc2012', 'gray', 'none'],
        'colorbar': [False, True],
        'display': ['show', 'save']
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
    print('\n{}'.format(args))

    # -----------------
    # DATA
    # -----------------
    if args['dataset'] == 'VOC2012':
        data = VOC2012('VOC2012/', split=args['split'], is_transform=True, img_size=args['img_size'])
    elif args['dataset'] == 'DDD17':
        data = DDD17('DDD17/', split=args['split'], is_transform=True, img_size=args['img_size'], mod=True)

    # -----------------
    # VISUALIZE
    # -----------------
    print()
    indexes = [int(i) for i in input('Index of images (start:end): ').split(':')]
    indexes = range(indexes[0], indexes[1])

    img_bounds = [(float(bound) if '.' in bound else int(bound)) for bound in args['img_bounds'].split(',')]

    cmap = args['cmap'] if (args['cmap'] != 'none') else None

    visualize(args['dataset'], data, args['mode'], indexes, img_bounds, cmap, args['colorbar'])

    # -----------------
    # RESULT
    # -----------------
    if args['display'] == 'show':
        plt.show()
    elif args['display'] == 'save':
        file_path = './_visualizations/'
        try:
            os.mkdir(file_path)
        except OSError:
            pass
        plt.savefig('{}{}.png'.format(file_path, input('Visualization name: ')), bbox_inches='tight')