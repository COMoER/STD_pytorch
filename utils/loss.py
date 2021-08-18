import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

def smooth_l1(deltas, targets, sigma = 3.0):
    # Reference: https://mohitjainweb.files.wordpress.com/2018/03/smoothl1loss.pdf
    sigma2 = sigma * sigma
    diffs = deltas - targets
    smooth_l1_signs = torch.lt(torch.abs(diffs), 1.0 / sigma2).float()

    smooth_l1_option1 = torch.mul(diffs, diffs) * 0.5 * sigma2
    smooth_l1_option2 = torch.abs(diffs) - 0.5 / sigma2
    smooth_l1_add = torch.mul(smooth_l1_option1, smooth_l1_signs) + torch.mul(smooth_l1_option2, - smooth_l1_signs+1.)
    smooth_l1 = smooth_l1_add

    if smooth_l1.ndimension() > 1:
        smooth_l1 = torch.sum(smooth_l1, dim = 1)

    return smooth_l1

class BCEFocalLoss(nn.Module):
    '''
    from https://zhuanlan.zhihu.com/p/308290543
    '''
    def __init__(self, gamma=2, alpha=0.25, reduction='mean'):
        super(BCEFocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.reduction = reduction

    def forward(self, x, target):
        loss = - self.alpha * (1. - x) ** self.gamma * target * torch.log(x)\
               - (1. - self.alpha) * x ** self.gamma * (1. - target) * torch.log(1. - x)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class BCE(nn.Module):
    def __init__(self, reduction='mean'):
        super(BCE, self).__init__()
        self.reduction = reduction

    def forward(self, x, target):
        loss = - target * torch.log(x)\
               - (1. - target) * torch.log(1. - x)

        if self.reduction == 'mean':
            loss = torch.mean(loss)
        elif self.reduction == 'sum':
            loss = torch.sum(loss)
        return loss

class Reg(nn.Module):
    def __init__(self):
        super(Reg, self).__init__()
    def forward(self, pred_reg,pred_cls,target,cls_target):
        '''

        :param pred_reg: B,7
        :param pred_cls: B,12
        :param target: B,7 location,size,angle_reg
        :return:
        '''
        loss_cls = F.cross_entropy(pred_cls,cls_target)

        loss_reg = smooth_l1(pred_reg,target)

        return (loss_cls+loss_reg).view(-1)

def get_bbox(center,size,rotate_y):
    '''

    :param center: N,3
    :param size: N,3
    :param rotate_y:N
    :return: bbox N,6 eps N,8,3
    '''
    # rotate
    R_yaw = torch.stack([torch.cos(rotate_y), torch.zeros(rotate_y.shape, device=rotate_y.device), torch.sin(rotate_y),
                         torch.zeros(rotate_y.shape, device=rotate_y.device),
                         torch.ones(rotate_y.shape, device=rotate_y.device),
                         torch.zeros(rotate_y.shape, device=rotate_y.device),
                         -torch.sin(rotate_y), torch.zeros(rotate_y.shape, device=rotate_y.device),
                         torch.cos(rotate_y)],
                        dim=1).view(-1, 3, 3)
    # eight points
    h,w,l = size
    xps = torch.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],dim = 1)
    yps = torch.stack([h*0,h*0,h*0,h*0,-h,-h,-h,-h],dim = 1)
    zps = torch.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],dim = 1)
    eps = torch.stack([xps,yps,zps],dim = 2) # N,8,3
    eps = torch.bmm(R_yaw,eps.transpose(1,2)).transpose(1,2) + center.view(-1,1,3)
    bev = eps[:,:4,[0,2]]
    bbox = torch.stack([torch.min(bev[:,:,0],dim = 1)[0],torch.min(bev[:,:,1],dim=1)[0],eps[:,4,2],
                        torch.max(bev[:,:,0],dim=1)[0],torch.max(bev[:,:,1],dim=1)[0],eps[:,0,2]],dim=1) # N,6

    return bbox,eps

def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xyz = torch.min(box_a[:, 3:].unsqueeze(1).expand(A, B, 3)
                        ,box_b[:, 3:].unsqueeze(0).expand(A, B, 3))
    min_xyz = torch.max(box_a[:, :3].unsqueeze(1).expand(A, B, 3),
                        box_b[:, :3].unsqueeze(0).expand(A, B, 3))
    inter = torch.clamp((max_xyz - min_xyz), min=0)
    return inter[:,:,0] * inter[:,:,1] * inter[:,:, 2]


def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 3]-box_a[:, 0]) *
              (box_a[:, 4]-box_a[:, 1]) *
              (box_a[:, 5]-box_a[:, 2])).unsqueeze(1).expand_as(inter)
    area_b = ((box_b[:, 3]-box_b[:, 0]) *
              (box_b[:, 4]-box_b[:, 1]) *
              (box_b[:, 5]-box_b[:, 2])).unsqueeze(0).expand_as(inter)
    union = area_a + area_b - inter
    return inter / union  # A,B

def compute_3DIOU(bbox,ground_truth):
    '''

    :param ground_truth: G,6
    :param bbox: P,6
    :return: iou P,G
    '''

    return jaccard(bbox,ground_truth)

def corner_loss(eps,gt_eps):
    '''

    :param eps: N,8,3
    :param gt_eps: N,8,3
    :return: corner loss
    '''
    return torch.sum(torch.norm(eps-gt_eps,dim = 2),dim=1)






