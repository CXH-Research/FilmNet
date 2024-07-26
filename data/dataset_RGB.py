import os
import random

import torch
import torchvision.transforms.functional as F
import torchvision.transforms as TF
from PIL import Image
from torch.utils.data import Dataset
import numpy as np


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_dir, film_class='target', img_options=None):
        super(DataLoaderTrain, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps_w = self.img_options['w']
        self.ps_h = self.img_options['h']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        ratio_h = np.random.uniform(0.6, 1.0)
        ratio_w = np.random.uniform(0.6, 1.0)
        w, h = inp_img._size
        crop_h = round(h * ratio_h)
        crop_w = round(w * ratio_w)
        i, j, h, w = TF.RandomCrop.get_params(inp_img, output_size=(crop_h, crop_w))
        inp_img = F.crop(inp_img, i, j, h, w)
        tar_img = F.crop(tar_img, i, j, h, w)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        # Data Augmentations
        aug = random.randint(0, 8)
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        inp_img = F.resize(inp_img, [self.ps_h, self.ps_w])
        tar_img = F.resize(tar_img, [self.ps_h, self.ps_w])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename


class DataLoaderVal(Dataset):
    def __init__(self, rgb_dir, film_class='target', img_options=None):
        super(DataLoaderVal, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(rgb_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(rgb_dir, film_class)))

        self.inp_filenames = [os.path.join(rgb_dir, 'input', x) for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(rgb_dir, film_class, x) for x in tar_files if is_image_file(x)]

        self.img_options = img_options
        self.sizex = len(self.tar_filenames)  # get the size of target

        self.ps_w = self.img_options['w']
        self.ps_h = self.img_options['h']

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_path = self.inp_filenames[index_]
        tar_path = self.tar_filenames[index_]

        inp_img = Image.open(inp_path)
        tar_img = Image.open(tar_path)

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        inp_img = F.resize(inp_img, [self.ps_h, self.ps_w])
        tar_img = F.resize(tar_img, [self.ps_h, self.ps_w])

        filename = os.path.splitext(os.path.split(tar_path)[-1])[0]

        return inp_img, tar_img, filename


class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, img_options):
        super(DataLoaderTest, self).__init__()

        inp_files = sorted(os.listdir(os.path.join(inp_dir, 'input')))
        tar_files = sorted(os.listdir(os.path.join(inp_dir, 'target')))

        self.inp_filenames = [os.path.join(inp_dir, 'input') for x in inp_files if is_image_file(x)]
        self.tar_filenames = [os.path.join(inp_dir, 'target') for x in tar_files if is_image_file(x)]

        self.inp_size = len(self.inp_filenames)
        self.img_options = img_options

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):
        path_inp = self.inp_filenames[index]
        path_tar = self.tar_filenames[index]
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]
        inp = Image.open(path_inp)
        tar = Image.open(path_tar)

        inp = F.to_tensor(inp)
        tar = F.to_tensor(tar)
        return inp, tar, filename
