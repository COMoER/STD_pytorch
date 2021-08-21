import torch
import torch.nn as nn
from models.backbone import PointNet2,PointNet
from models.common import compute_proposal,in_side,compute_target,Box_branch,IOU_branch,compute_box,compute_reg
from utils.anchor import NMS,align_anchor,assign_anchor,NMS_proposal
from models.group_pointcloud import preprocess,PointsPool
from utils.loss import BCEFocalLoss,BCE,Reg,get_bbox,corner_loss,smooth_l1,compute_3DIOU,compute_3DIOU_one2one
import numpy as np

# classes
# 'Car','Pedestrian','Cyclist'

IOU_UP_THRES = 0.3
IOU_DOWN_THRES = 0.1
MAX_PROPOSAL_SIZE = 128
PROPOSAL_SIZE_EVAL = 64

class PGM(nn.Module):
    def __init__(self,cls,second_stage_training = True):
        super(PGM, self).__init__()
        self.model = PointNet2(1)
        self.model2 = PointNet()
        self.cls = cls
        self.focal_loss = BCEFocalLoss()
        self.cls_loss = BCE()
        self.reg_loss = Reg()
        self.second_stage_training = second_stage_training
    def forward(self,points, label = None):
        # xyz B,N,4
        scores, features = self.model(points)
        # for each batch
        B, N, C = scores.shape
        B, N, D = features.shape
        if self.training:
            B, G, M = label.shape
        # single batch
        batch_id = 0
        score = scores[batch_id].view(1, N, C)
        feature = features[batch_id].view(1, N, D)
        xyz = points[batch_id].view(1, N, 4)
        anc,r = NMS(score, xyz, self.cls)
        # get the input of PointNet per anchor
        anchor_points, anchor_masks = align_anchor(anc, r, xyz, feature)
        if self.training:
            ground_truth = label[batch_id].view(1, G, M)
            pos_mask, pos_number, pointwise = assign_anchor(ground_truth, anchor_masks, xyz)
            pointwise_label = (pointwise > 0).float()
            ancwise_label = (pos_mask > 0).float()
            loss = self.focal_loss(score.view(-1), pointwise_label)
            judge_pos = (pos_mask > 0).cpu().numpy()
        proposals = []
        scores_anc = []
        reg_losses = []
        anc = anc.view(-1,3)
        for i,(pts, center) in enumerate(zip(anchor_points, anc)):  # to each anchor
            # using PointNet to predict the proposal
            pred_score,pred_reg, pred_cls = self.model2(pts)
            scores_anc.append(pred_score.view(1, 1))
            if self.training:
                # compute loss
                if judge_pos[i]:
                    targets, cls_targets = compute_target(ground_truth, center, pos_number[i], self.cls)
                    reg_losses.append(self.reg_loss(pred_reg,pred_cls,targets,cls_targets))
            pp = compute_proposal(pred_reg, pred_cls, center, self.cls)
            # pp = torch.cat([pp,number.view(-1,1)],dim = 1) # pp 1,8
            proposals.append(pp)
        if self.training:
            if len(scores_anc):
                scores_anc = torch.cat(scores_anc,dim = 0)
                loss = loss + self.cls_loss(scores_anc.view(-1),ancwise_label)
            if len(reg_losses): # when the number of positive anchor is positive
                loss = loss + torch.mean(torch.cat(reg_losses))
            # NMS of proposals
            return proposals,features,loss
        else:
            if len(scores_anc):
                scores_anc = torch.cat(scores_anc, dim=0).view(-1)
                proposals = torch.cat(proposals,dim = 0)
                # using NMS to delete extra proposal
                proposals,scores_anc = NMS_proposal(proposals,scores_anc)
                index = torch.argsort(scores_anc,descending=True)
                if self.second_stage_training:
                    proposals = proposals[index[:min(MAX_PROPOSAL_SIZE,len(index)-1)]]
                else:
                    proposals = proposals[index[:min(PROPOSAL_SIZE_EVAL, len(index) - 1)]]
            return proposals,features

class STD(nn.Module):
    def __init__(self):
        super(STD, self).__init__()
        self.model3 = PointsPool()
        self.box = Box_branch()
        self.iou = IOU_branch()
        self.box_loss = Reg()
        self.cls_loss = nn.BCELoss()
    def forward(self,proposals,xyz,features,label = None):
        '''
        :param proposals: the proposals generated by PGM P,7
        :param features: feature generated by PointNet++ N,128
        :param label: the ground_truth for each proposal G,9
        :return:
        '''
        if self.training:
            label = label.view(-1,9)
        proposal_points,proposals_no_empty = in_side(xyz, features, proposals[:, :3], proposals[:, 3:6], proposals[:, 6],delete_empty=True)
        proposals = proposals[proposals_no_empty]
        # preprocess for PointsPool
        number, feature = preprocess(proposal_points, proposals)
        # PointsPool
        pred_proposal = self.model3(feature) # P,216,256
        # flatten
        gene_feature = pred_proposal.view(-1,6*6*6*256)

        #TODO: the score(using sigmoid to compute) may output 0, which cause the loss become NAN
        score,pred_reg,pred_cls = self.box(gene_feature)


        pred_iou = self.iou(gene_feature)

        bbox_seven = compute_box(proposals,pred_reg,pred_cls) # using seven param to describe bbox

        bbox,eps = get_bbox(bbox_seven[:,:3],bbox_seven[:,3:6],bbox_seven[:,6])

        p_bbox,_ = get_bbox(proposals[:,:3],proposals[:,3:6],proposals[:,6])

        if self.training:
            if label.shape[0]:
                # use ground_truth 3d IOU to assign the pred bbox
                gt_bbox, gt_eps = get_bbox(label[:, 1:4], label[:, 4:7], label[:, 7])
                gt_p_iou = compute_3DIOU(p_bbox,gt_bbox) # compute iou between proposal and ground_truth

                p_iou_max, gt_assign = torch.max(gt_p_iou, dim=1)  # P
                p_pos = p_iou_max >= IOU_UP_THRES
                p_neg = p_iou_max <= IOU_DOWN_THRES
                N = torch.sum(p_pos)
                if N > 0:
                    bbox_iou = compute_3DIOU_one2one(bbox[p_pos],gt_bbox[gt_assign[p_pos]]) # (N,) iou between bbox and gt

                    reg, cls = compute_reg(proposals[p_pos], label[gt_assign[p_pos]])

                    iou_loss = smooth_l1(pred_iou[p_pos].view(1,-1), bbox_iou.view(1,-1))

                    closs = corner_loss(eps[p_pos], gt_eps[gt_assign[p_pos]])

                    box_loss = self.box_loss(pred_reg[p_pos], pred_cls[p_pos], reg, cls)

                else:
                    box_loss = torch.zeros(torch.Size((1,)),dtype = torch.float,device=proposals.device)
                    iou_loss = torch.zeros(torch.Size((1,)),dtype = torch.float,device=proposals.device)
                    closs = torch.zeros(torch.Size((1,)),dtype = torch.float,device=proposals.device)
                cls_loss = self.cls_loss(score.view(-1), p_pos.float())
            else:
                zeros = torch.zeros(torch.Size((proposals.shape[0],)),dtype = torch.float,device=proposals.device)
                cls_loss = self.cls_loss(score.view(-1), zeros)
                box_loss = torch.zeros(torch.Size((1,)), dtype=torch.float, device=proposals.device)
                iou_loss = torch.zeros(torch.Size((1,)), dtype=torch.float, device=proposals.device)
                closs = torch.zeros(torch.Size((1,)), dtype=torch.float, device=proposals.device)


            # loss = cls_loss + box_loss + iou_loss + closs

            assert not torch.isnan(cls_loss)

            return cls_loss,box_loss,iou_loss,closs
        else:
            return torch.cat([score,pred_iou,proposals,bbox_seven],dim = 1) # [cls_score,iou_score,center,size,rotate_y]





