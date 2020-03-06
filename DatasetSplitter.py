##############################
#
# @desc:    
#        List of functions designed to read a split/process npy files
#        and convert them to h5py files. 
#
##############################


import pandas as pd 
import numpy as np 
import math

def loadnpyfiles(npyname, rootpath=''):
    npzimg = np.load(rootpath+'kothaddekha_ImageArray_'+npyname+'.npy')
    npzlbl = np.load(rootpath+'kothaddekha_LabelArray_' +
                     npyname+'.npy', allow_pickle=True)
    npzlen = np.load(rootpath+'kothaddekha_LenthArray_'+npyname+'.npy')
    return npzimg, npzlbl, npzlen

# def create_sepeate_static_numpy_valid(name, rp):
#     npim, nplb, npln = loadnpyfiles(name, rp)

#     npim_val, nplb_val, npln_val = npim[:50], nplb[:50], npln[:50]
#     npim_trtst, nplb_trtst, npln_trtst = npim[50:], nplb[50:], npln[50:]

#     print('saving!')

#     np.save('kothaddekha_ImageArray_'+name+'val.npy', npim_val)
#     np.save('kothaddekha_LabelArray_'+name+'val.npy', nplb_val)
#     np.save('kothaddekha_LenthArray_'+name+'val.npy', npln_val)

#     np.save('kothaddekha_ImageArray_'+name+'trtst.npy', npim_trtst)
#     np.save('kothaddekha_LabelArray_'+name+'trtst.npy', nplb_trtst)
#     np.save('kothaddekha_LenthArray_'+name+'trtst.npy', npln_trtst)

# def customSplit(npyArr,splitcount):
    
#     len = npyArr.shape[0]

#     size = math.floor(len/splitcount)

#     start = 0
#     end = size

#     for i in range(splitcount-1):

        



#     print(len)
    
    
#     return 0

def splitFiles(npyArr,directory,savefilename,splitcount):
    

    len = npyArr.shape[0]
    size = math.floor(len/splitcount)

    npyArr = npyArr[:splitcount*size]
    split = np.split(npyArr, splitcount)

    i = 0
    for arr in split:

        np.save(directory+"no_"+str(i)+"_"+savefilename, arr)
        i+=1
    
    return 0



def main():

    npzimg, npzlbl, npzlen = loadnpyfiles('openslrbn6k3seco','data/openSLR/')
    

    # customSplit(npzimg,3)

    splitFiles(npzimg,'data/openSLR/Splitted/','kothaddekha_ImageArray_openslrbn6k3seco.npy',7)
    splitFiles(npzlbl,'data/openSLR/Splitted/','kothaddekha_LabelArray_openslrbn6k3seco.npy',7)
    splitFiles(npzlen,'data/openSLR/Splitted/','kothaddekha_LenthArray_openslrbn6k3seco.npy',7)


main()