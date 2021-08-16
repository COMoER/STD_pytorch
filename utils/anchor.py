import torch
import torchvision
from models.pointnet2_utils import square_distance
from models.common import in_side
import cv2
import numpy as np

receptive_region_radius = {  # the unit is meter
    'car': 2,
    'pedestrian': 1,
    'cyclist': 1
}
iou_thres = 0.1

def float2BEV(bbox):
    bbox_numpy = bbox.cpu().numpy().reshape(-1, 4)
    x_min = np.min(bbox_numpy[:, 0])
    x_max = np.max(bbox_numpy[:, 2])
    y_min = np.min(bbox_numpy[:, 1])
    y_max = np.max(bbox_numpy[:, 3])
    x_range = x_max - x_min
    y_range = y_max - y_min
    bbox_numpy[:, [0, 2]] = (bbox_numpy[:, [0, 2]] - x_min) / x_range * 640
    bbox_numpy[:, [1, 3]] = (bbox_numpy[:, [1, 3]] - y_min) / y_range * 640
    canvas = np.zeros((640, 640), np.uint8)

    for bb in bbox_numpy:
        cv2.rectangle(canvas, tuple(bb[:2]), tuple(bb[2:]), 255, 2)

    return canvas


def assign_anchor(G, center_mask, points):
    '''

    :param G: ground_truth [:,cls,x,y,z,h,w,l,theta] 1,G,8
    :param center_mask: list of the points inside anchors mask 1,N
    :param points: 1,N,4
    :return:
    mask tensor INT >0 is the pos
    number tensor indicate which ground_truth to choose
    '''
    G_cat_mask = torch.zeros(torch.Size((points.shape[1],)),dtype = torch.int,device = points.device)
    if G.shape[1] > 0: # if with
        _, G_masks = in_side(points, None, G[:, :, 1:4].view(-1, 3), G[:, :, 4:7].view(-1, 3), G[:, :, 7].view(-1),
                              True)

        G_masks = torch.cat(G_masks, dim=0)
        torch.sum(G_masks,dim = 0,out = G_cat_mask)
        center_mask = torch.cat(center_mask, dim=0)
        T, N = G_masks.shape
        A, N = center_mask.shape
        G_masks = G_masks.view(1, T, N)

        center_mask = center_mask.view(A, 1, N)
        sum_mask = G_masks + center_mask  # 2 is both 0 is neither
        # cls = G[:,:,0].view(-1)
        points_iou = torch.sum(sum_mask == 2, dim=-1).float() / torch.sum(sum_mask > 0, dim=-1).float()
        # assignment of anchor, first the iou must over 0.55 and
        mask = torch.sum(points_iou > 0.55, dim=1)  # A pos or neg
        assign_number = torch.full(torch.Size((A,)), -1, dtype=torch.int, device=mask.device)
        if mask[mask>0].shape[0] > 0:
            assign_number[mask > 0] = torch.argmax(points_iou[mask>0],dim = 1)  # positive ones
        # assign = cls[torch.argmax(points_iou,dim = -1)]
    else:
        A= len(center_mask)
        mask = torch.full(torch.Size((A,)), 0, dtype=torch.int, device=points.device)
        assign_number = torch.full(torch.Size((A,)), -1, dtype=torch.int, device=mask.device)
    return mask,assign_number,G_cat_mask

def NMS(score,points,cls):
    '''
    score B,N,1
    points B,N,4
    '''
    radius_list = list(receptive_region_radius.values())
    radius = radius_list[cls]
    # for each class
    xy = points[:,:,[0,2]]
    bbox = torch.cat([xy-radius,xy+radius],dim = 2)
    mask = torchvision.ops.nms(bbox[0],score[:,:,0][0],iou_thres)
    nms_ps = points[:,mask,:3]

    # assume single batch
    # anchor_per_cls.append((nms_ps,radius))
    return nms_ps,radius,score[:,mask]


def align_anchor(anchors,radius,points,feature):
    '''
    anchors: the anchors for this cls B,M,3
    radius: the radius of anchor for this cls
    points: whole points B,N,4
    feature: the output of PointNet++ B,N,128
    return: points for each anchor list(B,N',131)
    '''
    src = points[:,:,:3]
    feature = feature
    dist = square_distance(src,anchors) # B,N,M
    B,N,M = dist.shape
    dist = dist.view(B,M,N)
    points_per_anchor = []
    mask_per_anchor = []
    for anchor_id in range(M):
        d = dist[:,anchor_id]
        mask = d <= radius # inside
        ps = src[mask].view(B,-1,3)-anchors[:,anchor_id].view(B,1,3) # normalized
        feat = feature[mask].view(B,-1,128)
        points_per_anchor.append(torch.cat([ps,feat],dim = -1)) # B,N',131
        mask_per_anchor.append(mask)
    return points_per_anchor,mask_per_anchor



