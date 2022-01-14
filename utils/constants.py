"""

Constant values.

@author: Joshua Chough

"""

from matplotlib.colors import ListedColormap

# Dataset labels
labels = dict(
    voc2012={
        0: 'background',
        1: 'aeroplane',
        2: 'bicycle',
        3: 'bird',
        4: 'boat',
        5: 'bottle',
        6: 'bus',
        7: 'car',
        8: 'cat',
        9: 'chair',
        10: 'cow',
        11: 'diningtable',
        12: 'dog',
        13: 'horse',
        14: 'motorbike',
        15: 'person',
        16: 'pottedplant',
        17: 'sheep',
        18: 'sofa',
        19: 'train',
        20: 'tv',
    },
    ddd17={
        0:'flat',
        1:'sky+construction',
        2:'object',
        3:'nature',
        4:'human',
        5:'vehicle'
    }
)

# Dataset orig img sizes
img_sizes = dict(
    voc2012=(256, 256),
    ddd17=(200, 346)
)

# Custom dataset configuration class
class DatasetConfig():

    def __init__(self, name, input_dim, hasClasses=True, path=''):
        self.name = name
        self.labels = labels[self.name] if hasClasses else None
        self.num_cls = len(labels[self.name]) if hasClasses else None
        self.input_dim = input_dim
        self.path = path if path else ('./data/' + self.name)
        self.img_size = img_sizes[self.name]

    def dictionary(self):
        return dict(
            name=self.name,
            labels=self.labels,
            num_cls=self.num_cls,
            input_dim=self.input_dim,
            path=self.path,
            img_size=self.img_size
        )

# Dataset configurations
dataset_cfg = dict(
    voc2012=DatasetConfig(
        name='voc2012',
        input_dim=3,
    ).dictionary(),
    ddd17=DatasetConfig(
        name='ddd17',
        input_dim=2,
    ).dictionary()
)

# VOC2012 color map
'''
color map
0=background, 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle, 6=bus, 7=car, 8=cat, 9=chair, 10=cow, 11=diningtable,
12=dog, 13=horse, 14=motorbike, 15=person # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
'''
voc_palette = [(0,0,0), (128,0,0), (0,128,0), (128,128,0), (0,0,128), (128,0,128), (0,128,128), (128,128,128), (64,0,0), (192,0,0), (64,128,0), (192,128,0), (64,0,128), (192,0,128), (64,128,128), (192,128,128), (0, 64,0), (128, 64, 0), (0,192,0), (128,192,0), (0,64,128)]
voc_palette = [[r/255, g/255, b/255, 1] for (r, g, b) in voc_palette]
voc_gt_palette = voc_palette + [[0, 0, 0, 1] for i in range(256 - len(voc_palette) - 1)] + [[1, 1, 1, 1]]

voc_cmap = ListedColormap(voc_palette)
voc_gt_cmap = ListedColormap(voc_gt_palette)
