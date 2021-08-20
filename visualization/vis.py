import numpy as np
import cv2
import os
import torch
from models.STD import STD,PGM
import torch.nn as nn

os.environ["CUDA_VISIBLE_DEVICES"] = '0'


PC_DIR = "/data/kitti/KITTI/data_object_velodyne/training/velodyne/"
LABEL_DIR = "/data/kitti/KITTI/training/label_2/"
IMAGE_DIR = "/data/kitti/KITTI/data_object_image_2/training/image_2/"
CALIB_DIR = "/data/kitti/KITTI/training/calib/"
POINTS = 8*1024 # 4k

classes = ['Car','Pedestrian','Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']

def in_side(src,label):
    '''

    src: N,4
    label:G,9

    return list of PointsPool input for each proposal
    '''
    src = src[:,:3]
    # h = size[:,0] # y
    # w = size[:,1] # z
    # l = size[:,2] # x
    ry = label[:,7]
    R = np.stack([np.cos(ry),0*ry,np.sin(ry),0*ry,np.ones(ry.shape),0*ry,
                        -np.sin(ry),0*ry,np.cos(ry)],axis = 1).reshape((-1,3,3))

    src = np.matmul(R.transpose((0,2,1)), (src.reshape(1,-1,3) - label[:,1:4].reshape(-1,1,3)).transpose((0,2,1))).transpose((0,2,1))  # G,N,3
    points_per_g = []
    for id in range(label.shape[0]):
        h,w,l = label[id,4:7].T
        src_p = src[id]
        mask = np.logical_and(
            np.logical_and(np.logical_and(src_p[ :, 0] <= l / 2,src_p[ :, 0] >= -l / 2),
            np.logical_and((src_p[ :, 1] <= 0),(src_p[ :, 1] >= -h)))
            ,np.logical_and((src_p[ :, 2] >= -w / 2),(src_p[ :, 2] <= w / 2))) # inside
        points_per_g.append((R[id]@src_p[mask].T).T + label[id,1:4])
    return points_per_g



def float2BEV(ps,bbox,p_bbox,inside):
    '''

    :param ps: N,4 x,y,z,r BEV is X-Z
    bbox G,8,3
    :return:
    '''
    x_min = np.min(ps[:, 0])
    x_max = np.max(ps[:, 0])
    y_min = np.min(ps[:, 2])
    y_max = np.max(ps[:, 2])
    x_range = x_max - x_min
    y_range = y_max - y_min
    ps[:, 0] = (ps[:, 0] - x_min) / x_range * 640
    ps[:, 2] = 640-(ps[:, 2] - y_min) / y_range * 640
    canvas = np.zeros((640, 640,3), np.uint8)

    bbox[:,:,0] = (bbox[:,:, 0] - x_min) / x_range * 640
    bbox[:,:,2] = 640-(bbox[:,:, 2] - y_min) / y_range * 640

    bbox = bbox[:,:,[0,2]]

    p_bbox[:,:,0] = (p_bbox[:,:, 0] - x_min) / x_range * 640
    p_bbox[:,:,2] = 640-(p_bbox[:,:, 2] - y_min) / y_range * 640

    p_bbox = p_bbox[:,:,[0,2]]

    for p in ps:
        cv2.circle(canvas,tuple(p[[0,2]].astype(int)),1,(0,255,0),-1)

    inside[:,0] = (inside[:, 0] - x_min) / x_range * 640
    inside[:,2] = 640-(inside[:, 2] - y_min) / y_range * 640

    inside = inside[:,[0,2]]

    for p in inside:
        cv2.circle(canvas,tuple(p.astype(int)),1,(255,255,255),-1)

    for g in bbox:
        g = g.astype(int)
        for i in range(4):
            cv2.line(canvas,tuple(g[i]),tuple(g[(i+1)%4]),(0,0,255),2)
            cv2.line(canvas,tuple(g[i+4]),tuple(g[(i+1)%4+4]),(0,0,255),2)
            cv2.line(canvas,tuple(g[i]),tuple(g[i+4]),(0,0,255),2)

    for g in p_bbox:
        g = g.astype(int)
        for i in range(4):
            cv2.line(canvas,tuple(g[i]),tuple(g[(i+1)%4]),(255,0,0),2)
            cv2.line(canvas,tuple(g[i+4]),tuple(g[(i+1)%4+4]),(255,0,0),2)
            cv2.line(canvas,tuple(g[i]),tuple(g[i+4]),(255,0,0),2)

    return canvas
def load_calib(calib_path):
    '''
    :param calib_path:
    :return: transform from lidar to camera0, transform from lidar to image2, transform from camera0 to image2
    '''
    with open(calib_path,'r') as f:
        lines = f.readlines()
        P2 = np.float32(lines[2].split(' ')[1:]).reshape(3,4)
        P2 = np.concatenate([P2,np.float32([[0,0,0,1]])],axis = 0)
        P_lc = np.float32(lines[5].split(' ')[1:]).reshape(3,4)
        P_lc = np.concatenate([P_lc,np.float32([[0,0,0,1]])],axis = 0)
        R0_rect = np.float32(lines[4].split(' ')[1:]).reshape(3,3)
        eye4 = np.eye(4)
        eye4[:3,:3] = R0_rect
        R0_rect = eye4.copy()

    return P_lc,P2@R0_rect@P_lc,P2

def convert(label):
    '''

    input format N*15
    return label [cls,G_x,G_y,G_z,G_h,G_w,G_l,r_y]
    '''

    cls_name = label[:,0]
    for i,cls in enumerate(classes):
        cls_name = np.where(cls_name == cls,np.ones(cls_name.shape,dtype = np.float32)*i,cls_name)
    label = np.float32(label[:,1:])
    cls_name = cls_name.astype(np.float32).reshape(-1,1)
    G_l = label[:,10:13]
    v = label[:,13].reshape(-1,1)
    alpha = label[:,2].reshape(-1,1)
    G_size = label[:,7:10]
    mask = (cls_name == 0).reshape(-1) # only train for car class
    return np.concatenate([cls_name,G_l,G_size,v,alpha],axis = 1)[mask]

def drawBox3d(label,P,im,color = (0,0,255)):
    '''

    :param label: G,9
    :param P: 4*4
    :return: im
    '''
    h,w,l = label[:,4:7].T
    xps = np.stack([l/2, l/2, -l/2, -l/2, l/2, l/2, -l/2, -l/2],axis = 1)
    yps = np.stack([h*0,h*0,h*0,h*0,-h,-h,-h,-h],axis = 1)
    zps = np.stack([w/2, -w/2, -w/2, w/2, w/2, -w/2, -w/2, w/2],axis = 1)
    eps = np.stack([xps,yps,zps],axis = 2).transpose((0,2,1)) # G,3,8
    ry = label[:,7]
    R = np.stack([np.cos(ry),0*ry,np.sin(ry),0*ry,np.ones(ry.shape),0*ry,
                        -np.sin(ry),0*ry,np.cos(ry)],axis = 1).reshape((-1,3,3))
    eps = np.matmul(R,eps).transpose((0,2,1)) + label[:,1:4].reshape(-1,1,3) # G,8,3
    bbox = eps.copy()
    eps = eps[np.sum(eps[:,:,2] > 0,axis=1)>1]
    ones = np.ones((*eps.shape[:2],1))
    eps = np.concatenate([eps,ones],axis = 2).transpose((0,2,1)) # G,4,8
    ips = np.matmul(P,eps).transpose((0,2,1))
    ips = (ips/ips[:,:,2].reshape(-1,8,1))[:,:,:2]
    for g in ips:
        g = g.astype(int)
        for i in range(4):
            cv2.line(im,tuple(g[i]),tuple(g[(i+1)%4]),color,2)
            cv2.line(im,tuple(g[i+4]),tuple(g[(i+1)%4+4]),color,2)
            cv2.line(im,tuple(g[i]),tuple(g[i+4]),color,2)
    return im,bbox

if __name__ == '__main__':
    weights = "/data/usr/zhengyu/exp/STD/2021-08-19_09-39/checkpoints/best.pt"
    model = PGM(0).cuda()
    model = STD
    checkpoint = torch.load(weights)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    for lidar_file in os.listdir(PC_DIR)[1000:]:
        # raw point cloud
        T_l,T_image2,T_02 = load_calib(CALIB_DIR+lidar_file.split('.')[0]+'.txt')
        im = cv2.imread(IMAGE_DIR+lidar_file.split('.')[0]+'.png')
        H,W,_ = im.shape
        pc = np.fromfile(PC_DIR + lidar_file, dtype=np.float32).reshape(-1, 4)[:,:3]
        indices = np.random.choice(range(len(pc)), POINTS)
        pc = pc[indices]
        pc_input = np.fromfile(PC_DIR + lidar_file, dtype=np.float32).reshape(-1, 4)[indices]
        pc_input[:, :3] = (T_l[:3, :3] @ (pc[:, :3].T)).T + T_l[:3, 3].reshape(1, 3)
        pc = np.concatenate([pc,np.ones((pc.shape[0],1))],axis = 1)

        pc_camera = (T_l@(pc.T)).T

        ips = (T_image2@(pc.T)).T # N,4
        ips = (ips/ips[:,2].reshape(-1,1))[:,:2]

        mask = np.logical_and(np.logical_and(ips[:,0]>=0,ips[:,0]<=W),np.logical_and(ips[:,1]>=0,ips[:,1]<=H))

        # for p in ips[mask]:
        #     cv2.circle(im,tuple(p.astype(int)),2,(0,255,0),-1)




        label = convert(np.loadtxt(LABEL_DIR+lidar_file.split('.')[0]+'.txt',dtype = object,delimiter=' ').reshape(-1,15))
        label_model = torch.from_numpy(label).view(-1,9).to(torch.device('cuda:0'))
        if label.shape[0] == 0:
            continue
        pc_per_g = in_side(pc_camera,label)

        pcs = np.concatenate(pc_per_g,axis = 0) # N,3

        im,bbox = drawBox3d(label,T_02,im)

        pps = pcs.copy()


        pcs = np.concatenate([pcs,np.ones((pcs.shape[0],1))],axis = 1)
        ips = (T_image2@np.linalg.inv(T_l)@(pcs.T)).T # N,4
        ips = (ips/ips[:,2].reshape(-1,1))[:,:2]
        mask = np.logical_and(np.logical_and(ips[:, 0] >= 0, ips[:, 0] <= W),
                              np.logical_and(ips[:, 1] >= 0, ips[:, 1] <= H))

        for p in ips[mask]:
            cv2.circle(im,tuple(p.astype(int)),2,(0,255,0),-1)

        # using model to predict proposal
        pc = torch.from_numpy(pc_input).view(-1,4).to(torch.device('cuda:0'))
        proposals,feature = model(pc.view(1,-1,4),label_model.view(1,-1,9))
        proposals = proposals.detach().cpu().numpy()
        proposals = np.concatenate([np.ones((proposals.shape[0],1)),proposals],axis = 1)

        im,p_bbox = drawBox3d(proposals,T_02,im,(255,0,0))

        canvas = float2BEV(pc_camera,bbox,p_bbox,pps)




        save_path = "/home/zhengyu/STD/project_model_img/"
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        cv2.imwrite(save_path+lidar_file.split('.')[0]+'_project.jpg',im)
        cv2.imwrite(save_path+lidar_file.split('.')[0]+'_bev.jpg',canvas)

        print("%s saved"%(save_path+lidar_file.split('.')[0]+'_project.jpg'))
