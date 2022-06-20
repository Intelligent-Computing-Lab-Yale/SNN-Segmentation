# /voc2012

## Directory structure

`/voc2012` is compartmentalized as follows:

- `/JPEGImages`: Directory for the VOC2012 image data
- `/SegmentationClassAug`: Directory for the VOC2012 label data
- `/datalists`: Directory for the VOC2012 datalists
- `dataset.py`: Custom dataset class for VOC2012
- `datalists.py`: Script to generate datalists (lists of specific examples used for training/testing) for VOC2012 dataset

## Data format

Each data file in `/JPEGImages` and `/SegmentationClassAug` will contain the data for one frame.

The naming format will be `<sequence-number>_<frame-number>` (e.g. `2007_000063`).

In `/JPEGImages`, the data files will be `.jpg` files. In `/SegmentationClassAug`, the data files will be `.png` files.