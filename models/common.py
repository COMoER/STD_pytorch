import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


bin_angle = 2*np.pi/12
pre_defined_size = [[1.6,0.8,0.8],
                    [1.6,0.8,1.6],
                    [1.6,1.6,3.9]]


def in_side(src,feature,center,size,rotate_y,need_mask = False):
    '''

    points: B,N,4
    feature: B,N,128
    center: P,3
    size: P,3
    rotate_y: P

    return list of PointsPool input for each proposal
    '''
    src = src[:,:,:3]
    # h = size[:,0] # y
    # w = size[:,1] # z
    # l = size[:,2] # x
    R_yaw = torch.stack([torch.cos(rotate_y), torch.zeros(rotate_y.shape, device=rotate_y.device), torch.sin(rotate_y),
                     torch.zeros(rotate_y.shape, device=rotate_y.device),
                     torch.ones(rotate_y.shape, device=rotate_y.device),
                     torch.zeros(rotate_y.shape, device=rotate_y.device),
                     -torch.sin(rotate_y), torch.zeros(rotate_y.shape, device=rotate_y.device), torch.cos(rotate_y)],
                    dim=1).view(-1,3,3)
    # R_roll = torch.stack([torch.cos(rotate_z), -torch.sin(rotate_z),torch.zeros(rotate_z.shape, device=rotate_z.device),
    #                  torch.sin(rotate_z), torch.cos(rotate_z),torch.zeros(rotate_z.shape, device=rotate_z.device),
    #                      torch.zeros(rotate_z.shape, device=rotate_z.device),
    #                      torch.zeros(rotate_z.shape, device=rotate_z.device),
    #                       torch.ones(rotate_z.shape, device=rotate_z.device)],
    #                 dim=1).view(-1,3,3)
    src = torch.bmm(R_yaw.transpose(1,2), (src - center.view(-1, 1, 3)).transpose(2, 1)).transpose(2, 1)  # B,N,3
    proposal_points = []
    if need_mask:
        proposal_masks = []
    for proposal_id in range(len(size)):
        h,w,l = size[proposal_id]
        src_p = src[proposal_id].view(1,-1,3)
        mask = (src_p[:, :, 0] <= l / 2) & (src_p[:, :, 0] >= -l / 2) & (src_p[:, :, 1] <= 0) & (src_p[:, :, 1] >= -h) & (
                src_p[:, :, 2] >= -w / 2) & (src_p[:, :, 2] <= w / 2) # inside
        if isinstance(feature,torch.Tensor):
            proposal_points.append(torch.cat([src_p[mask],feature[mask]],dim = 1)) # the input of PointsPool
        else:
            proposal_points.append(src_p[mask])
        if need_mask:
            proposal_masks.append(mask)
    if need_mask:
        return proposal_points,proposal_masks
    else:
        return proposal_points
def compute_target(gt,center,cls_id):
    '''
    :param gt: B,8
    :param center: anchor center 3
    :param cls_id:
    :return: targets B,8
    '''
    cls = torch.floor((gt[:,6] + np.pi) / bin_angle).float()
    angle_reg = (gt[:,6] + np.pi) - cls*bin_angle
    l_reg = gt[:,:3] - center
    size = torch.from_numpy(np.float32(pre_defined_size[cls_id])).to(gt.device)
    size_reg = (gt[:,3:6] - size)/size
    return torch.cat([l_reg,size_reg,cls,angle_reg],dim = 1)
def compute_proposal(pred_reg,pred_cls,center,cls_id):
    '''
    :param pred_reg: N,3+3+1 first is for rotate_y second is for rotate_z
    :param pred_cls_y: N,12
    :param pred_cls_z: N,12
    :param center:
    :param cls_id:
    :return: proposals,G_reg
    '''
    rotate_y = (torch.argmax(pred_cls,dim = 1).float()-6)*bin_angle + pred_reg[:,6]
    pred_center = center + pred_reg[:,:3]
    pred_size = torch.from_numpy(np.float32(pre_defined_size[cls_id])).to(pred_reg.device) + pred_reg[:,3:6] # h,w,l

    return torch.cat([pred_center,pred_size,rotate_y.view(-1,1)],dim = 1)
class Branch(nn.Module):
    def __init__(self):
        super(Branch,self).__init__()
        self.fc1 = nn.Linear(256,512)
        self.fc2 = nn.Linear(512,512)
        self.dp = nn.Dropout(0.4)
    def forward(self,x):
        x = F.relu(self.dp(self.fc1(x)))
        x = F.relu(self.dp(self.fc2(x)))
        return x
class Box_branch(nn.Module):
    def __init__(self):
        super(Box_branch, self).__init__()
        self.feature = Branch()
        # reg
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 3+3+1) # xyz,hwl,theta_reg
        # cls
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 12) # 12 bins

        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        # reg
        x_reg = F.relu(self.dropout(self.fc1(x)))
        x_reg = self.fc2(x_reg)
        # cls
        x_cls = F.relu(self.dropout(self.fc3(x)))
        x_cls = self.fc4(x_cls)
        return x_reg,x_cls

if __name__ == '__main__':
    a = torch.randn(10,2)
    b = torch.randn(2,4)
    # c = in_side(b,a)

