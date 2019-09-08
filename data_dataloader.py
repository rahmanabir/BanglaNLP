import cv2
import numpy as np
from matplotlib import pyplot as plt
import sklearn.cluster as sk
from PIL import Image
import torch
import torchvision
from torchvision import transforms
from torch.utils import data as D

import os
import glob
import os.path as osp
import pandas as pd
import gc
import random
import warnings
warnings.filterwarnings('ignore')

# ------------------------------------------------

def displayimage(imgarr, r=1, c=3):
    plt.figure(figsize=(18,32))
    count = 1
    for i in range(1,r+1):
        for j in range(1,c+1):
            plt.subplot(r,c,count)
            plt.imshow(imgarr[count-1])
            count += 1
    plt.show()

def torchimshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))

def torchdatagrid(images):
    plt.figure(figsize=(16,8))
    torchimshow(torchvision.utils.make_grid(images))

# ------------------------------------------------------------------

def saveimagesasnpy(dir='images/'):
    filearray = []
    labels = []
    filenames = glob.glob(osp.join(dir, '*.png'))
    for fn in filenames:
        filearray.append(fn)
        labels.append(1)

    length = len(filearray)
    imgarr = []
    for index in range(0,length):
        image = Image.open(filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((320, 200), Image.NEAREST)
        nimage = nimage.convert('RGB')
        img = np.array(nimage)
        imgarr.append(img)
    imgarr = np.array(imgarr)
    # imgarr = np.moveaxis(imgarr,3,1)
    np.save('kothaddekha_ImageArray_'+len(imgarr),imgarr)


def saveimagesasnpy_modular(path='images/', name='MOD', length=200, shuffle=False):
    filearray = []
    filenames = glob.glob(osp.join(path, '*.png'))
    count = length
    for fn in filenames:
        if count > 0:
            filearray.append(fn)
            count -= 1
    if shuffle is True: random.shuffle(filearray)
    imgarr = []
    for index in range(0,length):
        image = Image.open(filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((320, 200), Image.NEAREST)
        nimage = nimage.convert('RGB')
        img = np.array(nimage)
        imgarr.append(img)
    imgarr = np.array(imgarr)
    # imgarr = np.moveaxis(imgarr,3,1)
    np.save('kothaddekha_ImageArray_'+name+'_'+str(length)+'.npy',imgarr)


def loadnpyfiles(npyname):
    npzimg = np.load('kothaddekha_ImageArray_'+npyname+'.npy')
    npzlbl = np.load('kothaddekha_LabelArray_'+npyname+'.npy')
    return npzimg, npzlbl


def convertimagestotensor(dirname='images/'):
    filearray = []
    filenames = glob.glob(osp.join(dirname, '*.png'))
    for fn in filenames:
        filearray.append(fn)
    length = len(filearray)
    for index in range(0,length):
        image = Image.open(filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((320, 200), Image.NEAREST)
        nimage = nimage.convert('RGB')
        t = transforms.ToTensor()
        img = t(nimage)
        torch.save(img, dirname+'/tensorImages/'+'tensor_'+str(index)+'.pt')


########################################### DATASET ################
""" Swap commmented with active code in case of issues """
class KD_DL(D.Dataset):

    def __init__(self, root):
        """ Intialize the dataset """
        self.root = root
        self.imgarray = np.load('kothaddekha_ImageArray_'+root+'.npy')
        self.labels = np.load('picsle8_LabelArray_'+root+'.npy')
        self.len = len(self.labels)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        t = transforms.ToTensor()
        image = t(self.imgarray[index])
        label = self.labels[index]
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class KD_DL_Tensor(D.Dataset):

    def __init__(self, root):
        """ Intialize the dataset """
        self.root = root
        self.labels = np.load('picsle8_LabelArray_'+root+'.npy')
        nparray = np.load('kothaddekha_ImageArray_'+root+'.npy')
        self.imgarray = []
        t = transforms.ToTensor()
        for im in nparray:
            self.imgarray.append(t(im))
        self.len = len(self.labels)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        # t = transforms.ToTensor()
        image = self.imgarray[index]
        label = self.labels[index]
        return image, label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


class KD_DL_Raw(D.Dataset):

    def __init__(self, root):
        """ Intialize the dataset """
        self.filearray = []
        self.labels = []
        self.root = root
        self.transform = transforms.ToTensor()
        filenames = glob.glob(osp.join(self.root, '*.png'))
        for fn in filenames:
            self.filearray.append(fn)
            self.labels.append(1)        
        self.len = len(self.filearray)

    def __getitem__(self, index):
        """ Get a sample from the dataset """
        image = Image.open(self.filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((320,200), Image.NEAREST)
        nimage = nimage.convert('RGB')
        label = self.labels[index]
        return self.transform(nimage), label

    def __len__(self):
        """ Total number of samples in the dataset """
        return self.len


# class KD_DL_RawTensor(D.Dataset):

#     def __init__(self, root):
#         """ Intialize the dataset """
#         self.filearray = []
#         self.imgarray = []
#         self.labels = []
#         self.root = root
#         self.tens = transforms.ToTensor()
#         filenames = glob.glob(osp.join(self.root+'Pixelart/', '*.jpg'))
#         for fn in filenames:
#             self.filearray.append(fn)
#             self.labels.append(1)
#         filenames = glob.glob(osp.join(self.root+'Realpix/', '*.jpg'))
#         for fn in filenames:
#             self.filearray.append(fn)
#             self.labels.append(0)
#         self.len = len(self.filearray)

#         for index in range(0,self.len):
#             image = Image.open(self.filearray[index])
#             if image.size[0] != image.size[1]:
#                 sqrsize = min(image.size)
#                 croptrans = transforms.CenterCrop((sqrsize,sqrsize))
#                 image = croptrans(image)
#             nimage = image.resize((128, 128), Image.NEAREST)
#             nimage = nimage.convert('RGB')
#             self.imgarray.append(self.tens(nimage))

#     def __getitem__(self, index):
#         """ Get a sample from the dataset """
#         image = Image.open(self.filearray[index])
#         if image.size[0] != image.size[1]:
#             sqrsize = min(image.size)
#             croptrans = transforms.CenterCrop((sqrsize,sqrsize))
#             image = croptrans(image)
#         nimage = image.resize((128, 128), Image.NEAREST)
#         nimage = nimage.convert('RGB')
#         label = self.labels[index]
#         return self.tens(nimage), label

####################################################################


####################################################################

def get_loaders(path,split_perc=0.7,batch_size=32,mode=0):

    # Simple dataset. Only save path to image and load it and transform to tensor when call getitem.
    if mode==0:
        dataset = KD_DL(path)           # Numpy File
    elif mode==1:
        dataset = KD_DL_Tensor(path)    # Numpy to Tensor
    elif mode==2:
        dataset = KD_DL_Raw(path)       # Raw Images
    else:
        raise Exception('Parameter MODE given was {}. But needs to be an integer (0-2)\n0: Numpy, 1: Npy-Tensor, 2: Raw'.format(mode))

    # total images in set
    print(dataset.len,'images from the dataset')
    
    # divide dataset into training and validation subsets
    train_len = int(split_perc*dataset.len)
    valid_len = dataset.len - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])
    print(len(train), len(valid))
    
    # Use the torch dataloader to iterate through the dataset
    trainloader = D.DataLoader(train, batch_size=batch_size, shuffle=False, num_workers=0)
    validloader = D.DataLoader(valid, batch_size=batch_size, shuffle=False, num_workers=0)
    return trainloader, validloader

def main_func():
    path = 'images/'
    # Simple dataset. Only save path to image and load it and transform to tensor when call __getitem__.
    dataset = KD_DL(path)
    # total images in set
    print(dataset.len,'images from the dataset')
    # divide dataset into training and validation subsets
    train_len = int(0.7*dataset.len)
    valid_len = dataset.len - train_len
    train, valid = D.random_split(dataset, lengths=[train_len, valid_len])
    len(train), len(valid)
    # Use the torch dataloader to iterate through the dataset
    trainloader = D.DataLoader(train, batch_size=32, shuffle=False, num_workers=0)
    validloader = D.DataLoader(valid, batch_size=32, shuffle=False, num_workers=0)

    # get some images
    dataiter_tr = iter(trainloader)
    dataiter_vl = iter(validloader)
    images_t, labels_t = dataiter_tr.next()
    images_v, labels_v = dataiter_vl.next()

    # show images and match labels 4 fun
    plt.figure(figsize=(16,8))
    torchimshow(torchvision.utils.make_grid(images_t))
    print('Train:',labels_t)
    plt.figure(figsize=(16,8))
    torchimshow(torchvision.utils.make_grid(images_v))
    print('Valid:',labels_v)

# tl, vl = get_loaders('Pixel_750', mode=1);    #mode 0/1 numpy+tensor
# tl, vl = get_loaders('images/', mode=3);      #mode 2/3 raw+rawtensor
# print(tl)
