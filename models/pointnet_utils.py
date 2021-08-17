import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
class Conv(nn.Module):
    def __init__(self,in_channel,out_channel):
        super(Conv, self).__init__()
        self.conv = nn.Conv1d(in_channel,out_channel,1)
    def forward(self,x):
        return F.relu(self.conv(x))
class PointNetEncoder(nn.Module):
    def __init__(self,in_channel = 131,k = 12):
        super(PointNetEncoder, self).__init__()
        self.mlp = nn.Sequential(Conv(in_channel,128),
                                 Conv(128,128),
                                 Conv(128,256),
                                 Conv(256,512))
    def forward(self,x):
        '''
        x: 3 normalized coordinates + 128 semantic features for each proposal
        (B,M,131) assume B=1 single batch
        '''
        B,M,D = x.shape
        x = x.transpose(2,1)
        x = self.mlp(x) # B,512,M
        # maxpool
        x = torch.max(x,dim = -1)[0] # B,512 (maxpool for points)
        return x