import cv2
import numpy as np
from matplotlib import pyplot as plt
import sklearn.cluster as sk
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils import data as D
import csv

import os
import glob
import os.path as osp
import pandas as pd
import gc
import random
import progressbar as prgs
import warnings
warnings.filterwarnings('ignore')




def loadnpyfiles(file_names, rootpath=''):
    npzimg = np.load(rootpath+file_names[0])
    npzlbl = np.load(rootpath+file_names[1], allow_pickle=True)
    npzlen = np.load(rootpath+file_names[2])
    return npzimg, npzlbl, npzlen


########################################### DATASET ################
 # Swap commmented with active code in case of issues
class KD_DL(D.Dataset):

    def __init__(self, root, file_names):
        """ Intialize the dataset """
        self.root = root
        self.imgarray, self.labels, self.lens = loadnpyfiles(
            file_names, rootpath=root)
        self.len = len(self.imgarray)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        t = transforms.ToTensor()
        image = t(self.imgarray[index])
        label = self.labels[index]
        lengths = self.lens[index]
        return image, label, lengths

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len

####################################################################



def get_loaders(file_names, split_perc=0.7, batch_size=32, mode=0, rootpath=''):
    # Simple dataset. Only save path to image and load it and transform to tensor when call getitem.
    if mode == 0:
        dataset = KD_DL(rootpath,file_names)           # Numpy File
    else:
        raise Exception(
            'Parameter MODE given was {}. But needs to be an integer (0-2)\n0: Numpy, 1: Npy-Tensor, 2: Raw'.format(mode))

    # total images in set
    print(dataset.len, 'images from the dataset')

    # divide dataset into training and validation subsets
    train_len = int(split_perc*dataset.len)
    valid_len = dataset.len - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])
    print(len(train), len(valid))

    # Use the torch dataloader to iterate through the dataset
    trainloader = D.DataLoader(
        train, batch_size=batch_size, shuffle=False, num_workers=0)
    validloader = D.DataLoader(
        valid, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, validloader


def main_func():

    fileNames = ['no_0_kothaddekha_ImageArray_openslrbn6k3seco.npy',
                'no_0_kothaddekha_LabelArray_openslrbn6k3seco.npy',
                'no_0_kothaddekha_LenthArray_openslrbn6k3seco.npy']

    rpath = r'data/openSLR/Splitted/Train/'
    # Simple dataset. Only save path to image and load it and transform to tensor when call __getitem__.
    dlt, dlv = get_loaders(fileNames, mode=0, rootpath=rpath)

    # get some images
    dataiter_tr = iter(dlt)
    dataiter_vl = iter(dlv)
    images_t, labels_t, lens_t = dataiter_tr.next()
    images_v, labels_v, lens_v = dataiter_vl.next()


    print(lens_t[0])
    print(labels_t[0])

if __name__ == "__main__":
    main_func()