# /ddd17

## Directory structure

`/ddd17` is compartmentalized as follows:

- `/events`: Directory for the DDD17 event (spike) data
- `/images`: Directory for the DDD17 image data
- `/labels`: Directory for the DDD17 label data
- `/datalists`: Directory for the DDD17 datalists
- `dataset.py`: Custom dataset class for DDD17
- `datalists.py`: Script to generate datalists (lists of specific examples used for training/testing) for DDD17 dataset

## Data directory structure

`/events`, `/images`, `/labels` are compartmentalized as follows:

- `/train`: Directory for the training data split
- `/test`: Directory for the testing data split
- `/other`: Directory for other data

## Data format

Each data file in `/train`, `/test`, and `/other` will contain the data for one frame.

The naming format will be `rec<sequence-number>_export_<frame-number>` (e.g. `rec1487417411_export_100`).

In `/events`, the data files will be `.npy` files. In `/images` and `/labels`, the data files will be `.png` files.