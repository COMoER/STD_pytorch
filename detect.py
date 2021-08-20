import torch
from models.STD import PGM,STD
import numpy as np


class STD_whole():
    def __init__(self,weight1="/data/usr/zhengyu/exp/STD//checkpoints/best.pt",
                 weight2 = "/data/usr/zhengyu/exp/STD_SECOND/2021-08-16_23-22/checkpoints/best.pt"):
        self.model1 = PGM(0).cuda().eval()
        checkpoint = torch.load(weight1)
        self.model1.load_state_dict(checkpoint['model_state_dict'])
        self.model2 = STD().cuda().eval()
        checkpoint = torch.load(weight2)
        self.model2.load_state_dict(checkpoint['model_state_dict'])

    def detect(self,x):
        '''
        the main function for detection
        :param x: (1,8*1024,3) raw pc numpy.ndarray
        :return: bbox [N,score,iou_score,center,size,yaw]
        '''
        assert isinstance(x,np.ndarray),"Please input numpy.ndarray"
        x = torch.from_numpy(x).to(torch.float32).cuda()
        proposals, features = self.model1(x)
        pred = self.model2(proposals,x,features)
        return pred.cpu().numpy()
