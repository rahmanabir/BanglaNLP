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

# ------------------------------------------------


def displayimage(imgarr, r=1, c=3):
    plt.figure(figsize=(18, 32))
    count = 1
    for i in range(1, r+1):
        for j in range(1, c+1):
            plt.subplot(r, c, count)
            plt.imshow(imgarr[count-1])
            count += 1
    plt.show()


def torchimshow(img):
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))


def torchdatagrid(images):
    plt.figure(figsize=(16, 8))
    torchimshow(torchvision.utils.make_grid(images))

# ------------------------------------------------------------------


def saveimagesasnpy(dir='data/spectographs/', csvf='dekhabet_dataLabelsRanged.csv', name='2k2sec'):
    labels = []
    fnames = []
    lens = []
    with open(csvf, 'r') as csvFile:
        reader = csv.reader(csvFile)
        for row in reader:
            ctext = []
            text = row[4]
            text = text.strip("'-!$[]")
            text = text.split(',')
            i = 0
            z = 0
            for t in range(43):
                if 'Token' in row[4]:
                    pass
                else:
                    try:
                        if int(text[t]) != 0:
                            ctext.append(int(text[t]))
                            i += 1
                        else:
                            z += 1
                            # print(text[t], type(text[t]))
                    except IndexError:
                        ctext.append(0)
                        # print('zero')
            for t in range(z):
                ctext.append(0)
            labels.append(ctext)
            lens.append(i)
            fnames.append(row[0])
            print(len(ctext))
    labels.pop(0)
    lens.pop(0)
    fnames.pop(0)
    print('lfn:', len(fnames))
    print('lbl:', len(labels))
    print('lln:', len(lens))
    print('lll:', len(labels[55]))
    length = len(fnames)
    imgarr = []
    for index in range(0, length):
        image = Image.open(dir+fnames[index]+'.wav.jpg')
        nimage = image.resize((256, 128), Image.NEAREST)
        nimage = nimage.convert('RGB')
        img = np.array(nimage)
        imgarr.append(img)
        if index % 100 == 0:
            print(index, 'imgs added to array')
    imgarr = np.array(imgarr)
    csvFile.close()
    print(fnames[0], lens[0], labels[0], len(labels[0]))
    print(fnames[1], lens[1], labels[1], len(labels[1]))
    print(fnames[55], lens[55], labels[55], len(labels[55]))
    l = np.array(labels)
    print(type(l), type(l[0]), l[0])
    np.save('kothaddekha_ImageArray_'+name+'.npy', imgarr)
    np.save('kothaddekha_LabelArray_'+name+'.npy', l)
    np.save('kothaddekha_LenthArray_'+name+'.npy', np.array(lens))



def saveimagesasnpy2(dir='data/openslr_bengali/spectrograms_mel/', csvf='data/openslr_bengali/transcript_openslr_ranged.csv', name='openslrbn6k3seco'):
    import data_dekhabet as dkb
    labels = []
    fnames = []
    lens = []

    df = pd.read_csv(csvf)
    l = len(df)
    prgs.printProgressBar(0, l, 'Token Vectorization')
    for index, row in df.iterrows():
        ctext = []
        text = row['Vector']
        text = text.strip("'-!.?$[] ")
        text = text.split(',')
        i = 0
        z = 0
        for t in range(43):
            try:
                if int(text[t]) != 0:
                    ctext.append(int(text[t]))
                    i += 1
                else:
                    z += 1
            except IndexError:
                ctext.append(0)
        for t in range(z):
            ctext.append(0)
        labels.append(ctext)
        lens.append(i)
        fnames.append(row['Filename'])
        prgs.printProgressBar(index, l, 'Token Vectorization')
    
    print('lfn:', len(fnames))
    print('lbl:', len(labels))
    print('lln:', len(lens))

    length = len(fnames)
    imgarr = []
    prgs.printProgressBar(0, l, 'Image to Numpy')
    for index in range(0, length):
        image = Image.open(dir+row['Filename']+'.flac.png')
        nimage = image.resize((256, 128), Image.NEAREST)
        nimage = nimage.convert('RGB')
        img = np.array(nimage)
        imgarr.append(img)
        prgs.printProgressBar(index, l, 'Image to Numpy')
    imgarr = np.array(imgarr)
    print(fnames[0], lens[0], labels[0], len(labels[0]))
    print(fnames[1], lens[1], labels[1], len(labels[1]))
    print(fnames[55], lens[55], labels[55], len(labels[55]))
    l = np.array(labels)
    print(type(l), type(l[0]), l[0])
    np.save('kothaddekha_ImageArray_'+name+'.npy', imgarr)
    np.save('kothaddekha_LabelArray_'+name+'.npy', l)
    np.save('kothaddekha_LenthArray_'+name+'.npy', np.array(lens))

# def saveimagesasnpy(dir='data/spectographs/', csvf='dekhabet_dataLabelsRanged.csv', name='2k2sec'):
#     # filearray = []
#     # filenames = glob.glob(osp.join(dir, '*.jpg'))
#     labels = []
#     fnames = []
#     lens = []
#     with open(csvf, 'r') as csvFile:
#         reader = csv.reader(csvFile)
#         for row in reader:
#             ctext = []
#             text = row[4]
#             text = text.strip("'-!$[]")
#             text = text.split(',')
#             i=0
#             print(text)
#             for t in range(0,43):
#                 if text[t]=='Tokens':
#                     pass
#                 else:
#                     try:
#                         if:
#                             ctext.append(int(text[t]))
#                             i+=1
#                     except:
#                         ctext.append(0)
#             labels.append(ctext)
#             lens.append(i-1)
#             fnames.append(row[0])
#
#     labels.pop(0)
#     lens.pop(0)
#     fnames.pop(0)
#     print('lfn:',len(fnames))
#     print('lbl:',len(labels))
#     print('lln:',len(lens))
#
#     length = len(fnames)
#     imgarr = []
#     for index in range(0,length):
#         image = Image.open(dir+fnames[index]+'.wav.jpg')
#         nimage = image.resize((256, 128), Image.NEAREST)
#         nimage = nimage.convert('RGB')
#         img = np.array(nimage)
#         imgarr.append(img)
#         if index%100==0:
#             print(index,'imgs added to array')
#     imgarr = np.array(imgarr)
#
#     csvFile.close()
#     print(fnames[0],lens[0],labels[0])
#     print(fnames[1],lens[1],labels[1])
#     print(fnames[55],lens[55],labels[55])
#     np.save('kothaddekha_ImageArray_'+name+'.npy',imgarr)
#     np.save('kothaddekha_LabelArray_'+name+'.npy',labels)
#     np.save('kothaddekha_LenthArray_'+name+'.npy',lens)


def saveimagesasnpy_modular(path='data/spectographs/', name='MOD', length=200, shuffle=False):
    filearray = []
    filenames = glob.glob(osp.join(path, '*.jpg'))
    count = length
    for fn in filenames:
        if count > 0:
            filearray.append(fn)
            count -= 1
    if shuffle is True:
        random.shuffle(filearray)
    imgarr = []
    for index in range(0, length):
        image = Image.open(filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((128, 96), Image.NEAREST)
        nimage = nimage.convert('RGB')
        img = np.array(nimage)
        imgarr.append(img)
    imgarr = np.array(imgarr)
    # imgarr = np.moveaxis(imgarr,3,1)
    np.save('kothaddekha_ImageArray_'+name+'_'+str(length)+'.npy', imgarr)


def loadnpyfiles(npyname, rootpath=''):
    npzimg = np.load(rootpath+'kothaddekha_ImageArray_'+npyname+'.npy')
    npzlbl = np.load(rootpath+'kothaddekha_LabelArray_' +
                     npyname+'.npy', allow_pickle=True)
    npzlen = np.load(rootpath+'kothaddekha_LenthArray_'+npyname+'.npy')
    return npzimg, npzlbl, npzlen


def convertimagestotensor(dirname='data/spectographs/'):
    filearray = []
    filenames = glob.glob(osp.join(dirname, '*.png'))
    for fn in filenames:
        filearray.append(fn)
    length = len(filearray)
    for index in range(0, length):
        image = Image.open(filearray[index])
        # if image.size[0] != image.size[1]:
        #     sqrsize = min(image.size)
        #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
        #     image = croptrans(image)
        nimage = image.resize((128, 96), Image.NEAREST)
        nimage = nimage.convert('RGB')
        t = transforms.ToTensor()
        img = t(nimage)
        torch.save(img, dirname+'/tensorImages/'+'tensor_'+str(index)+'.pt')


########################################### DATASET ################
 # Swap commmented with active code in case of issues
class KD_DL(D.Dataset):

    def __init__(self, root, rpath=''):
        """ Intialize the dataset """
        self.root = root
        self.imgarray, self.labels, self.lens = loadnpyfiles(
            root, rootpath=rpath)
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


class KD_DL_Tensor(D.Dataset):

    def __init__(self, root):
        """ Intialize the dataset """
        self.root = root
        self.labels = np.load('kothaddekha_LabelArray_'+root+'.npy')
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


# class KD_DL_Raw(D.Dataset):

#     def __init__(self, root):
#         """ Intialize the dataset """
#         self.filearray = []
#         self.labels = []
#         self.root = root
#         self.transform = transforms.ToTensor()
#         filenames = glob.glob(osp.join(self.root, '*.jpg'))
#         for fn in filenames:
#             self.filearray.append(fn)
#             self.labels.append(1)
#         with open(csvf, 'r') as csvFile:
#             reader = csv.reader(csvFile)
#             for row in reader:
#                 self.labels.append(row[4])
#         csvFile.close()
#         self.len = len(self.filearray)

#     def __getitem__(self, index):
#         """ Get a sample from the dataset """
#         image = Image.open(self.filearray[index])
#         # if image.size[0] != image.size[1]:
#         #     sqrsize = min(image.size)
#         #     croptrans = transforms.CenterCrop((sqrsize,sqrsize))
#         #     image = croptrans(image)
#         nimage = image.resize((256, 128), Image.NEAREST)
#         nimage = nimage.convert('RGB')
#         label = self.labels[index]
#         return self.transform(nimage), label

#     def __len__(self):
#         """ Total number of samples in the dataset """
#         return self.len


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

def create_sepeate_static_numpy_valid(name, rp):
    npim, nplb, npln = loadnpyfiles(name, rp)

    npim_val, nplb_val, npln_val = npim[:50], nplb[:50], npln[:50]
    npim_trtst, nplb_trtst, npln_trtst = npim[50:], nplb[50:], npln[50:]

    print('saving!')

    np.save('kothaddekha_ImageArray_'+name+'val.npy', npim_val)
    np.save('kothaddekha_LabelArray_'+name+'val.npy', nplb_val)
    np.save('kothaddekha_LenthArray_'+name+'val.npy', npln_val)

    np.save('kothaddekha_ImageArray_'+name+'trtst.npy', npim_trtst)
    np.save('kothaddekha_LabelArray_'+name+'trtst.npy', nplb_trtst)
    np.save('kothaddekha_LenthArray_'+name+'trtst.npy', npln_trtst)


def get_loaders(path, split_perc=0.7, batch_size=32, mode=0, rootpath=''):
    # Simple dataset. Only save path to image and load it and transform to tensor when call getitem.
    if mode == 0:
        dataset = KD_DL(path, rootpath)           # Numpy File
    elif mode == 1:
        dataset = KD_DL_Tensor(path)    # Numpy to Tensor
    # elif mode==2:
    #     dataset = KD_DL_Raw(path)       # Raw Images
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
    path = '2k2sec_22class'
    rpath = 'data/numpy_arrays/22_class/'
    # Simple dataset. Only save path to image and load it and transform to tensor when call __getitem__.
    dlt, dlv = get_loaders(path, mode=0, rootpath=rpath)

    # get some images
    dataiter_tr = iter(dlt)
    dataiter_vl = iter(dlv)
    images_t, labels_t, lens_t = dataiter_tr.next()
    images_v, labels_v, lens_v = dataiter_vl.next()

    # show images and match labels 4 fun
    # plt.figure(figsize=(16,8))
    # torchimshow(torchvision.utils.make_grid(images_t))
    # print('Train:',labels_t)
    print(lens_t[0])
    print(labels_t[0])
    # plt.figure(figsize=(16,8))
    # torchimshow(torchvision.utils.make_grid(images_v))
    # print('Valid:',labels_v)

# tl, vl = get_loaders('Pixel_750', mode=1);    #mode 0/1 numpy+tensor
# tl, vl = get_loaders('images/', mode=3);      #mode 2/3 raw+rawtensor
# print(tl)
# print(os.getcwd())


# saveimagesasnpy()

# i, l, n = loadnpyfiles('2k2sec', '')

# print(l[1], type(l[1]), len(l[1]))
# print(n[1], type(n[1]))

# main_func()


# path = '2k2sec43'
# rpath = 'data/numpy_arrays/22_class_43sec_fixed/'
# create_sepeate_static_numpy_valid(path,rpath)

saveimagesasnpy2()