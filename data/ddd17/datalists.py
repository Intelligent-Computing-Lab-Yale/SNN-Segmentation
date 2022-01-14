"""

Generate datalists for DDD17 dataset.

@author: Joshua Chough

"""

import os
from natsort import natsorted, ns

list_directory = 'datalists/'
splits = ['train', 'test']
event_directory = 'events/'
label_directory = 'labels/'
batch_size = 64
mod = True if (input('Mod? ') == 'yes') else False

for split in splits:
    sequences = []
    paths = []
    lastExample = -1
    
    for path, subdirs, files in os.walk(event_directory + split):
        sequence = []
        for filename in natsorted(files, alg=ns.IGNORECASE):
            curExample = int(filename[filename.rfind('_')+1:filename.index('.npy')])
            if curExample != lastExample + 1:
                sequences.append(sequence)
                sequence = []
            sequence.append(filename[:-4])
            lastExample = curExample
    
    for seq in sequences:
        last = (len(seq) - (len(seq) % batch_size)) if mod else len(seq)
        # print(last)
        for filename in seq[:last]:
            paths.append('{0}{2}/{3}.npy {1}{2}/{3}.png\n'.format(event_directory, label_directory, split, filename))

    print('Train split size: {}'.format(len(paths)))

    list_path = list_directory + input('File name for {} split: '.format(split)) + '.txt'
    with open(list_path, 'w') as a:
        for path in paths:
            a.write(path)