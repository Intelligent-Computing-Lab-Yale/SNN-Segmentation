import torch
import numpy as np
import imageio
from skimage.transform import resize
import cv2
from torch.utils import data
from imgaug import augmenters as iaa
import random
from sklearn.preprocessing import MinMaxScaler as normalize

class DDD17(data.Dataset):
    def __init__(self, root, split="train", is_transform=False, is_augment=False, attack=False, img_size=512, mod=False, thl=False, thl_size=20, atk_factor=0):
        self.root = root
        self.split = split

        self.is_transform = is_transform
        self.is_augment = is_augment
        self.attack = attack
        self.img_size = img_size if isinstance(img_size, tuple) else(img_size, img_size)
        self.mod = mod
        self.thl = thl
        self.thl_rt = thl_rt
        self.thl_size = thl_size

        if self.attack == 'gaussianblur':
            self.gaussianBlur = iaa.GaussianBlur(sigma=atk_factor)
        elif self.attack == 'saltpepper':
            self.saltPepper = iaa.SaltAndPepper(p=atk_factor)
        elif self.attack == 'gaussiannoise':
            self.gaussianNoise = iaa.AdditiveGaussianNoise(scale=atk_factor)

        directory = 'datalists/'
        file_path = self.root + directory + self.split + ('-new' if self.mod else '') + '.txt'
        file_list = tuple(open(file_path, 'r'))
        file_list = [tuple(id_.rstrip().split(' ')) for id_ in file_list]
        self.files = file_list

        print('Finished initializing DDD17 {} dataset [{}]'.format(self.split, self.__len__()))

    def __len__(self):
        return (len(self.files) - self.thl_size) if (self.thl or self.thl_rt) else len(self.files)

    def __getitem__(self, index):

        if self.is_augment:
            flip_p = round(random.random())
            scale = random.uniform(0.50, 2)
            translate_x = random.uniform(-0.25, 0.25)
            translate_y = random.uniform(-0.25, 0.25)
            rotate = random.uniform(-15, 15)
            augments = (flip_p, scale, translate_x, translate_y, rotate)
        else:
            augments = None

        if self.thl or self.thl_rt:

            sequence_paths = self.files[index:index + self.thl_size]

            if self.thl_rt:
                curFile = sequence_paths[-1][0]
                curExample = int(curFile[curFile.rfind('_') + 1:curFile.index('.npy')])
                lastFile = sequence_paths[-2][0]
                lastExample = int(lastFile[lastFile.rfind('_') + 1:lastFile.index('.npy')])
                newScenario = True if (index == 0) or (curExample != lastExample + 1) else False

            sequence_paths = [(self.root+img_path, self.root+lbl_path) for (img_path, lbl_path) in sequence_paths]

            sequence = [(np.load(img_path), np.array(imageio.imread(lbl_path), dtype=np.int32)) for (img_path, lbl_path) in sequence_paths]

            if self.is_transform:
                sequence = [self.transform(img, lbl, augments) for (img, lbl) in sequence]

            imgs = [torch.from_numpy(img).float() for (img, lbl) in sequence]
            lbl = torch.from_numpy(sequence[-1][1]).long()

            if self.thl:
                return imgs, lbl
            elif self.thl_rt:
                return imgs, lbl, newScenario
        else:
            img_path, lbl_path = self.files[index]

            img_path = self.root + img_path
            lbl_path = self.root + lbl_path
            # print (img_path, lbl_path)

            img = np.load(img_path) # dtype=float32

            lbl = imageio.imread(lbl_path)
            lbl = np.array(lbl, dtype=np.int32)

            if self.is_transform:
                img, lbl = self.transform(img, lbl, augments)

            img = torch.from_numpy(img).float()
            lbl = torch.from_numpy(lbl).long()

            return img, lbl

    def transform(self, img, lbl, augments=None):
        
        # Take the first 2 channels (out of 6)
        img = img[:,:,:2]

        # print('before augment: ', img.shape, np.unique(img), round(np.mean(img), 3))

        # augment
        if self.is_augment:
            img, lbl = self.augment(img, lbl, augments)

        # print('after augment: ', img.shape, np.unique(img), round(np.mean(img), 3))

        # resize img and label
        img = resize(img, (self.img_size[0], self.img_size[1]))
        lbl = cv2.resize(lbl, dsize=(self.img_size[0], self.img_size[1]), interpolation=cv2.INTER_NEAREST)
        lbl = lbl.astype(int)
        
        # salt and pepper attack
        if self.attack == 'saltpepper':
            img = img.transpose(2, 0, 1)

            norm = normalize((0,255))
            img = np.array([norm.fit_transform(channel) for channel in img])

            img = img.transpose(1, 2, 0)

            img = self.saltPepper.augment_image(img)

        # gaussian noise attack
        elif self.attack == 'gaussiannoise':
            img = img.transpose(2, 0, 1)

            norm = normalize((0,255))
            img = np.array([norm.fit_transform(channel) for channel in img])

            img = img.transpose(1, 2, 0)

            img = self.gaussianNoise.augment_image(img)

        # HWC -> CHW
        img = img.transpose(2, 0, 1)

        # Normalize each img channel to [-0.5, 0.5]
        norm = normalize((-0.5,0.5))
        img = np.array([norm.fit_transform(channel) for channel in img])

        # gaussian blur attack
        if self.attack == 'gaussianblur':
            img = img.transpose(1, 2, 0)

            img = self.gaussianBlur.augment_image(img)

            img = img.transpose(2, 0, 1)

        return img, lbl

    def augment(self, img, lbl, augments=None):
        # use same augmentation values for both img and lbl
        flip_p, scale, translate_x, translate_y, rotate = augments

        seq_lbl = iaa.Sequential([
            iaa.Fliplr(flip_p),  # horizontally flip image based on flip_p
            iaa.Affine(
                scale={"x": scale, "y": scale}, # scale images to 50-200% of their size, individually per axis
                translate_percent={"x": translate_x, "y": translate_y}, # translate by -25 to +25 percent (per axis)
                rotate=rotate,  # rotate by -15 to +15 degrees
                order=0,  # nearest-neighbor interpolation (fast)
                cval=0, # constant value to use when fillig in new pixels
                mode="constant" # pads with constant cval
            )])

        seq_img = iaa.Sequential([
            iaa.Fliplr(flip_p),  # horizontally flip image based on flip_p
            iaa.Affine(
                scale={"x": scale, "y": scale}, # scale images to 50-200% of their size, individually per axis
                translate_percent={"x": translate_x, "y": translate_y}, # translate by -25 to +25 percent (per axis)
                rotate=rotate,  # rotate by -15 to +15 degrees
                order=0,  # nearest-neighbor interpolation (fast)
                cval=0, # constant value to use when fillig in new pixels
                mode="constant" # pads with constant cval
            )])
        
        lbl = seq_lbl.augment_image(lbl)
        img = seq_img.augment_image(img)
        
        return img, lbl