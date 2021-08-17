import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(self, pred_reg,pred_cls,target):
        '''

        :param pred_reg: B,7
        :param pred_cls: B,12
        :param target: B,8 location,size,angle_cls,angle_reg
        :return:
        '''
        cls = target[:,7].clone().to(torch.long)
        loss_cls = F.cross_entropy(pred_cls,cls)

        loss_reg = smooth_l1(pred_reg[:,:3],target[:,:3]) + \
            smooth_l1(pred_reg[:,3:6],target[:,3:6]) +\
            smooth_l1(pred_reg[:,6],target[:,7])

        return (loss_cls+loss_reg).view(-1)