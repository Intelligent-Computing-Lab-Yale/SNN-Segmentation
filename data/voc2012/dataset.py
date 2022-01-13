import torch
import numpy as np
import imageio
from skimage.transform import resize
import cv2
from torch.utils import data
from imgaug import augmenters as iaa

class VOC2012(data.Dataset):
    def __init__(self, root, split="train_aug", is_transform=False, img_size=512, attack=False, atk_factor=0):
        self.root = root
        self.split = split
        self.is_transform = is_transform
        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.mean = np.array([104.00699, 116.66877, 122.67892])

        directory = 'ImageSets/Segmentation/'
        file_path = self.root + directory + self.split + '.txt'
        file_list = tuple(open(file_path, 'r'))
        file_list = [tuple(id_.rstrip().split(' ')) for id_ in file_list]
        self.files = file_list

        self.attack = attack
        if self.attack == 'gaussianblur':
            self.gaussianBlur = iaa.GaussianBlur(sigma=atk_factor)
        elif self.attack == 'saltpepper':
            self.saltPepper = iaa.SaltAndPepper(p=atk_factor)
        elif self.attack == 'gaussiannoise':
            self.gaussianNoise = iaa.AdditiveGaussianNoise(scale=atk_factor)

        print('Finished initializing VOC2012 {} dataset [{}]'.format(self.split, self.__len__()))

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        img_path, lbl_path = self.files[index]

        img_path = self.root + img_path
        lbl_path = self.root + lbl_path

        img = imageio.imread(img_path)
        img = np.array(img, dtype=np.uint8)

        lbl = imageio.imread(lbl_path)
        lbl = np.array(lbl, dtype=np.int32)

        if self.is_transform:
            img, lbl = self.transform(img, lbl)
        
        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()

        return img, lbl


    def transform(self, img, lbl):
        # Reverse channels
        img = img[:, :, ::-1]

        # Cast to float32?
        img = img.astype(np.float32)

        # resize img and lbl
        img = resize(img, (self.img_size[0], self.img_size[1]))

        if self.split == "train_aug":
            lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = cv2.resize(lbl, dsize=(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)

        # salt and pepper attack
        if self.attack == 'saltpepper':
            img = self.saltPepper.augment_image(img)

        # gaussian noise attack
        elif self.attack == 'gaussiannoise':
            img = self.gaussianNoise.augment_image(img)

        # Subtract mean for normalization
        img -= self.mean
        img = img.astype(float) / 255.0

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # gaussian blur attack
        if self.attack == 'gaussianblur':
            img = img.transpose(1, 2, 0)

            img = self.gaussianBlur.augment_image(img)

            img = img.transpose(2, 0, 1)

        return img, lbl