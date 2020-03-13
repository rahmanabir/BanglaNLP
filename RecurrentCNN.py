import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import matplotlib.pyplot as plt
import numpy as np

from BILSTM import BidirectionalLSTM

class R_CNN(nn.Module):

    def __init__(self, train=True):
        super(R_CNN, self).__init__()

        in_nc = 3
        nf = 64
        hdn = 256
        nclass = 22 #dekhabet class
        
        
        # custom vesion of the CNN poposed 
        # here: https://arxiv.org/pdf/1507.05717.pdf
        self.convs = nn.Sequential(
            
            nn.Conv2d(in_nc, nf, 3, 1, 1),                                                                                                                 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2, 2), #64 filters, 64*128
            
            nn.Conv2d(nf, nf*2, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d(2,2), #128 filters, 32*64
            
            nn.Conv2d(nf*2, nf*4, 3, 1, 1), 
            nn.BatchNorm2d(nf*4),
            
            nn.Conv2d(nf*4, nf*4, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,1),(2,1)), #256 filters, 16*32
            
            
            nn.Conv2d(nf*4, nf*4, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((1,21),(2,1)),
            
            
            nn.Conv2d(nf*4, nf*8, 3, 1, 1), 
            nn.BatchNorm2d(nf*8),
            
            
            nn.Conv2d(nf*8, nf*8, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2,1),(2,1)),
            
            
            nn.Conv2d(nf*8, nf*8, 3, 1, 1), 
            nn.LeakyReLU(0.2, True),
            nn.MaxPool2d((2, 1),(2,1)),  
            
            nn.Conv2d(nf*8, nf*8, 2, 1,0), 
            
            
        )
        
        self.bilstm = nn.Sequential(
                        BidirectionalLSTM(nf*8, hdn, hdn),
                        BidirectionalLSTM(hdn, hdn, nclass),
                    )
        
        self.lgsftMx = nn.LogSoftmax(dim=2)
        self.sftMx = nn.Softmax(dim=2)
        
    def forward(self, x):

        out = self.convs(x)
        out = out.squeeze(2)
        out = out.permute(2, 0, 1) #ctc expects [width,batch,label]
        
        
        out = self.bilstm(out)
        
        if(self.train):
            out = self.lgsftMx(out)
        else:
            out = self.sftMx(out)
            
        
        
        return out
