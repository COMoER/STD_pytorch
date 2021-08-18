import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstractionMsg,PointNetFeaturePropagation
from models.pointnet_utils import PointNetEncoder

'''
one class
two angle
3d iou reference to define
'''

# This code only support single batch training, that is we assume all B is one
# PointNet++
# using the implement of https://github.com/yanx27/Pointnet_Pointnet2_pytorch
class PointNet2(nn.Module):
    def __init__(self,num_classes):
        super(PointNet2, self).__init__()

        self.sa1 = PointNetSetAbstractionMsg(4096, [0.2,0.4,0.6], [32,64,64], 1, [[32, 32, 64],[64,64,128],[64,96,128]])
        self.sa2 = PointNetSetAbstractionMsg(1024, [0.4, 0.8, 1.6], [16, 32, 32], 64+128+128, [[64, 64, 128], [128,128,256],[128,128,256]])
        self.sa3 = PointNetSetAbstractionMsg(256, [0.8, 1.6, 3.2], [16, 32, 64], 128+256+256, [[128,128,256], [128,128,256], [128, 256, 256]])
        self.sa4 = PointNetSetAbstractionMsg(64, [1.6, 3.2, 6.4], [16, 32, 64], 256+256+256, [[256, 256, 512], [256, 256, 512], [256,512, 1024]])
        self.fp4 = PointNetFeaturePropagation(256+256+256+512+512+1024, [128,128])
        self.fp3 = PointNetFeaturePropagation(128+256+256+128, [128,128])
        self.fp2 = PointNetFeaturePropagation(64+128+128+128, [128, 128])
        self.fp1 = PointNetFeaturePropagation(1+128, [128, 128])

        # for point-wise cls
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        B,N,D = xyz.shape
        l0_points = xyz[:,:,3:].view(B,1,N) # point feature reflection
        l0_xyz = xyz[:,:,:3].transpose(2,1)

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, l0_points, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = torch.sigmoid(x)
        x = x.permute(0, 2, 1)

        return x,l0_points.transpose(2,1) # score and feature

# PointNet used for cls and reg
class PointNet(nn.Module):
    def __init__(self):
        super(PointNet, self).__init__()
        self.feat = PointNetEncoder()
        # reg
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 3+3+1) # xyz,hwl,theta_reg_y,theta_reg
        # self.bn1 = nn.BatchNorm1d(512)
        # self.bn2 = nn.BatchNorm1d(256)
        # cls_1
        self.fc4 = nn.Linear(512, 256)
        self.fc5 = nn.Linear(256, 12) # 12 bins

        # proposal_classification
        self.fc6 = nn.Linear(512, 256)
        self.fc7 = nn.Linear(256, 1)
        # self.bn3 = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(p=0.4)

    def forward(self, x):
        '''
        x (M,131)
        '''
        x = self.feat(x) # maxpool feature B,512
        # reg
        x_reg = F.relu(self.dropout(self.fc1(x)))
        x_reg = F.relu(self.dropout(self.fc2(x_reg)))
        x_reg = self.fc3(x_reg)
        # cls
        x_cls = F.relu(self.dropout(self.fc4(x)))
        x_cls = self.fc5(x_cls)
        # score
        x_p = F.relu(self.dropout(self.fc6(x)))
        x_p = torch.sigmoid(self.fc7(x_p))
        return x_p,x_reg,x_cls









