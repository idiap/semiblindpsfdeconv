'''
Code for the pytorch implementation of
"Semi-Blind Spatially-Variant Deconvolution in Optical Microscopy
with Local Point Spread Function Estimation By Use Of Convolutional Neural Networks"

Copyright (c) 2018 Idiap Research Institute, http://www.idiap.ch/
Written by Adrian Shajkofci <adrian.shajkofci@idiap.ch>,

This file is part of Semi-blind Spatially-Variant Deconvolution.

This is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License version 3 as
published by the Free Software Foundation.

The software is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with the software. If not, see <http://www.gnu.org/licenses/>.
'''


from PIL import Image
import os
import sys
import bz2
import pickle
import torch
from torchvision import utils, transforms
import torch.utils.data as data
from torchvision.datasets import folder
from pandas.io.parsers import read_csv
import numpy as np
import logging

log = logging.getLogger('')


def grayloader(path):
    """
    PIL loader
    """
    return Image.open(path)


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def find_classes(dir):
    """
    Adapted from PyTorch / Torchvision
    """
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(os.listdir(dir)):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """
    Adapted from PyTorch / Torchvision
    """
    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples

        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm']


class ImageFolder(DatasetFolder):
    """
    Adapted from PyTorch / Torchvision
    """
    def __init__(self, root, file_list=None, transform=None, target_transform=None,
                 loader=folder.default_loader):
        super(ImageFolder, self).__init__(root, loader, folder.IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples
        self.file_list = file_list

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        target = self.file_list[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


def load_crops(batch_size, listname_train = "./filelistpsf_train_regression_gaussian", listname_test="./filelistpsf_test_regression_gaussian",
               foldername_train = "./psf_train_regression_gaussian", foldername_test = "./psf_test_regression_gaussian"):
    """
    Loads the training library of cropped files and a CSV file with ground truth value
    :param batch_size: batch size for the data loader
    :param listname_train: filename of the CSV file with the regression values
    :param foldername_train: folder name of the PNG files with a filename 00NNNNN.png, N begin the number of the line in the CSV file (excepted header)
    """
    train_file = read_csv(os.path.expanduser(listname_train)).values.astype(np.float)
    test_file = read_csv(os.path.expanduser(listname_test)).values.astype(np.float)
    train_set = ImageFolder(foldername_train, transform=transforms.ToTensor(),
                                   loader=grayloader, file_list=train_file)
    test_set = ImageFolder(foldername_test, transform=transforms.ToTensor(),
                                  loader=grayloader, file_list=test_file)

    train_loader = torch.utils.data.DataLoader(dataset=train_set, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_set, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader, train_file, test_file


def load_oneimage(size=64, filename='image.tif'):
    """
    Creates a dataloader with only one image
    """
    global patch_size, batch_size
    patch_size = size
    image = grayloader(filename)
    train_loader = torch.utils.data.DataLoader(dataset=image, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=image, batch_size=1, shuffle=True)
    return train_loader, test_loader


def scale(v):
    '''
    Normalize a 2D matrix with a maximum of 1 per pixel
    :param v:
    :return: normalized vector
    '''
    norm = np.linalg.norm(v, ord=1)
    if norm == 0:
        norm = np.finfo(v.dtype).eps
    out = v / norm
    return out * (1/np.max(np.abs(out)))


def normalize(v):
    '''
    Normalize a 2D matrix with a sum of 1
    :param v:
    :return: normalized vector
    '''
    norm=v.sum()
    if norm==0:
        norm=np.finfo(v.dtype).eps
    return v/norm


def gaussian_kernel(size, fwhmx = 3, fwhmy = 3, center=None):
    """ Make a square gaussian kernel.
    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """

    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]

    if center is None:
        if size % 2 == 0:
            print("WARNING gaussian_kernel : you have chosen a even kernel size and therefore the kernel is not centered.")
        x0 = y0 = size // 2
    else:
        x0 = center[0]
        y0 = center[1]

    return normalize(np.exp(-4 * np.log(2) * ( ((x - x0) ** 2) / fwhmx**2 + ((y - y0) ** 2) / fwhmy**2 )))


def pickle_save(filename, obj, compressed=True):
    """
    save object to file using pickle
    """

    try:
        if compressed:
            f = bz2.BZ2File(filename, 'wb')
        else:
            f = open(filename, 'wb')
    except IOError as details:
        sys.stderr.write('File {} cannot be written\n'.format(filename))
        sys.stderr.write(details)
        return

    pickle.dump(obj, f, protocol=2)
    f.close()


def pickle_load(filename, compressed=True):
    """
    Load from filename using pickle
    """

    try:
        if compressed:
            f = bz2.BZ2File(filename, 'rb')
        else:
            f = open(filename, 'rb')
    except IOError as details:
        sys.stderr.write('File {} cannot be read\n'.format(filename))
        sys.stderr.write(details)
        return

    obj = pickle.load(f)
    f.close()
    return obj


def unpad(img, npad):
    '''
    Revert the np.pad command
    '''
    return img[npad:-npad,npad:-npad]


def save(model, run_nb):
    log.info("Saving the model...")
    torch.save(model, 'model_{}.pt'.format(run_nb))


def load(name):
    logging.info("Loading the model {}... ".format(name))
    model = torch.load(name)
    return model
