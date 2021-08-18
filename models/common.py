import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F


bin_angle = 2*np.pi/12
pre_defined_size = [[1.6,1.6,3.9],
                    [1.6,0.8,0.8],
                    [1.6,0.8,1.6]]



def in_side_numpy(src,center,size,rotate_y):
    '''

    src: N,4
    label:G,9

    return list of PointsPool input for each proposal
    '''
    src = src[:,:3]
    # h = size[:,0] # y
    # w = size[:,1] # z
    # l = size[:,2] # x
    ry = rotate_y
    R = np.stack([np.cos(ry),0*ry,np.sin(ry),0*ry,np.ones(ry.shape),0*ry,
                        -np.sin(ry),0*ry,np.cos(ry)],axis = 1).reshape((-1,3,3))

    src = np.matmul(R.transpose((0,2,1)), (src.reshape(1,-1,3) - center.reshape(-1,1,3)).transpose((0,2,1))).transpose((0,2,1))  # G,N,3
    points_per_g = []
    for id in range(rotate_y.shape[0]):
        h,w,l = size[id].T
        src_p = src[id]
        mask = np.logical_and(
            np.logical_and(np.logical_and(src_p[ :, 0] <= l / 2,src_p[ :, 0] >= -l / 2),
            np.logical_and((src_p[ :, 1] <= 0),(src_p[ :, 1] >= -h)))
            ,np.logical_and((src_p[ :, 2] >= -w / 2),(src_p[ :, 2] <= w / 2))) # inside
        points_per_g.append((R[id]@src_p[mask].T).T + center[id])
    return points_per_g

def in_side(src,feature,center,size,rotate_y,need_mask = False):
    '''

    points: B,N,4
    feature: B,N,128
    center: P,3
    size: P,3
    rotate_y: P

    return list of PointsPool input for each proposal
    '''
    # p_per_g = in_side_numpy(src.cpu().numpy().reshape(-1,4),center.cpu().numpy(),size.cpu().numpy(),rotate_y.cpu().numpy())

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
def compute_target(gt,center,number,cls_id):
    '''
    :param gt: B,G,9
    :param center: anchor center 3
    :param number: the id of ground_truth
    :param cls_id:
    :return: targets B,7 cls_target B,1
    '''
    gt = gt[:,number]
    cls = torch.floor((gt[:,7] + np.pi) / bin_angle).float()
    angle_reg = (gt[:,7] + np.pi) - cls*bin_angle
    l_reg = gt[:,1:4] - center
    size = torch.from_numpy(np.float32(pre_defined_size[cls_id])).to(gt.device)
    size_reg = (gt[:,4:7] - size)/size
    return torch.cat([l_reg,size_reg,angle_reg.view(-1,1)],dim = 1),cls.to(torch.long)
def compute_proposal(pred_reg,pred_cls,center,cls_id):
    '''
    :param pred_reg: N,3+3+1 first is for rotate_y second is for rotate_z
    :param pred_cls_y: N,12
    :param pred_cls_z: N,12
    :param center:
    :param cls_id:
    :return: proposals 1,7
    '''
    rotate_y = (torch.argmax(pred_cls,dim = 1).float()-6)*bin_angle + pred_reg[:,6]
    pred_center = center + pred_reg[:,:3]
    pred_size = torch.from_numpy(np.float32(pre_defined_size[cls_id])).to(pred_reg.device) + pred_reg[:,3:6] # h,w,l

    return torch.cat([pred_center,pred_size,rotate_y.view(-1,1)],dim = 1)

def compute_box(proposals,reg,cls):
    '''

    :param proposals: P,8
    :param reg: P,7
    :param cls: P,12
    :return: bounding_box with format [center,size,rotate_y] P,7
    '''
    size = proposals[:,3:6] + reg[:,3:6]
    center = proposals[:,:3] + reg[:,:3]
    rotate_y = proposals[:,6] + torch.argmax(cls,dim = 1)[0].float()*bin_angle + reg[:,6]
    return torch.cat([center,size,rotate_y.view(-1,1)],dim = 1)

def compute_reg(proposals,gt):
    '''

    :param proposals: N,8
    :param label: N,8
    :return: reg N,7 cls N
    '''
    rotate_y = gt[:,7] - proposals[:,6]
    cls = torch.floor((rotate_y + np.pi) / bin_angle).float()
    angle_reg = (rotate_y + np.pi) - cls*bin_angle
    l_reg = gt[:,1:4] - proposals[:,:3]
    size = proposals[:,3:6]
    size_reg = (gt[:,4:7] - size)/size
    return torch.cat([l_reg,size_reg,angle_reg.view(-1,1)],dim = 1),cls.to(torch.long)



class Branch(nn.Module):
    def __init__(self):
        super(Branch,self).__init__()
        self.fc1 = nn.Linear(6*6*6*256,512)
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
        # proposal_cls
        self.fc5 = nn.Linear(512, 256)
        self.fc6 = nn.Linear(256, 1)
        # reg
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 3+3+1) # xyz,hwl,theta_reg
        # cls
        self.fc3 = nn.Linear(512, 256)
        self.fc4 = nn.Linear(256, 12) # 12 bins

        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        x = self.feature(x)
        # reg
        x_reg = F.relu(self.dropout(self.fc1(x)))
        x_reg = self.fc2(x_reg)
        # cls
        x_cls = F.relu(self.dropout(self.fc3(x)))
        x_cls = self.fc4(x_cls)
        # proposal cls score
        x_p = F.relu(self.dropout(self.fc5(x)))
        x_p = torch.sigmoid(self.fc6(x_p))
        return x_p,x_reg,x_cls

class IOU_branch(nn.Module):
    def __init__(self):
        super(IOU_branch, self).__init__()
        self.feature = Branch()
        # iou logistic regression
        self.fc1 = nn.Linear(512, 256)
        self.fc2 = nn.Linear(256, 1)

        self.dropout = nn.Dropout(p=0.4)
    def forward(self, x):
        x = self.feature(x)
        # iou score
        x = F.relu(self.dropout(self.fc1(x)))
        x = self.fc2(x)
        return x

if __name__ == '__main__':
    a = torch.randn(10,2)
    b = torch.randn(2,4)
    # c = in_side(b,a)

