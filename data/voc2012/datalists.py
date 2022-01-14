"""

Generate datalists for VOC2012 dataset.

@author: Joshua Chough

"""

def find_nth(haystack, needle, n):
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

list_directory = 'datalists/'
split = input('split [val-old/train_aug-old]\n> ')

if split not in ('val-old', 'train_aug-old'):
    raise ValueError('Split must be either \'val-old\' or \'train_aug-old\'')

paths = []

f = open(list_directory + split + '.txt', "r")
for line in f:
    if split == 'val-old':
        filename = line[:-1]
    elif split == 'train_aug-old':
        filename = line[find_nth(line, '/', 2) + 1:find_nth(line, '.', 1)]
    paths.append('JPEGImages/{0}.jpg SegmentationClassAug/{0}.png\n'.format(filename))

print('{} split size: {}'.format(split, len(paths)))
# exit()

list_path = list_directory + input('File name for {} split: '.format(split)) + '.txt'
with open(list_path, 'w') as a:
    for path in paths:
        a.write(path)