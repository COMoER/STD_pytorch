import torch
import torch.nn as nn
from models.STD import PGM,STD
import numpy as np



class STD_whole():
    def __init__(self,weights):
        self.model1 = PGM(0,False)
        checkpoint = torch.load(weights,torch.device('cuda',0))
        self.model1.load_state_dict(checkpoint['model1_state_dict'])
        self.model1.cuda().eval()
        self.model2 = STD()
        self.model2.load_state_dict(checkpoint['model2_state_dict'])
        self.model2.cuda().eval()
    def detect(self,x):
        '''
        the main function for detection
        :param x: (1,8*1024,3) raw pc numpy.ndarray
        :return: bbox [N,score,iou_score,center,size,yaw]
        '''
        assert isinstance(x,np.ndarray),"Please input numpy.ndarray"
        assert x.shape[1] == 8*1024

        x = torch.from_numpy(x).to(torch.float32).cuda()
        proposals, features = self.model1(x)
        pred = self.model2(proposals,x,features)
        return pred.cpu().numpy()

# a demo to use the model
if __name__ == '__main__':
    import os
    import cv2
    from visualization.vis import load_calib,in_side,convert,drawBox3d,float2BEV

    # parameters
    PC_DIR = "/data/kitti/KITTI/data_object_velodyne/training/velodyne/"
    LABEL_DIR = "/data/kitti/KITTI/training/label_2/"
    IMAGE_DIR = "/data/kitti/KITTI/data_object_image_2/training/image_2/"
    CALIB_DIR = "/data/kitti/KITTI/training/calib/"
    SAVE_PATH = "/data/usr/zhengyu/project_model_img/"
    POINTS = 8 * 1024  # 4k
    WEIGHTS_PATH = "/data/usr/zhengyu/STD_results/weights/2021-08-22_08-55/best.pt"
    CONF_THRES = 0.4
    IOU_THRES = 0.3

    net = STD_whole(WEIGHTS_PATH)

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

        # ips = (T_image2@(pc.T)).T # N,4
        # ips = (ips/ips[:,2].reshape(-1,1))[:,:2]

        # mask = np.logical_and(np.logical_and(ips[:,0]>=0,ips[:,0]<=W),np.logical_and(ips[:,1]>=0,ips[:,1]<=H))

        # for p in ips[mask]:
        #     cv2.circle(im,tuple(p.astype(int)),2,(0,255,0),-1)

        label = convert(np.loadtxt(LABEL_DIR+lidar_file.split('.')[0]+'.txt',dtype = object,delimiter=' ').reshape(-1,15))
        if label.shape[0]==0:
            continue
        pc_per_g = in_side(pc_camera,label)

        pcs = np.concatenate(pc_per_g,axis = 0) # N,3

        pps = pcs.copy()

        pcs = np.concatenate([pcs,np.ones((pcs.shape[0],1))],axis = 1)
        ips = (T_image2@np.linalg.inv(T_l)@(pcs.T)).T # N,4
        ips = (ips/ips[:,2].reshape(-1,1))[:,:2]
        mask = np.logical_and(np.logical_and(ips[:, 0] >= 0, ips[:, 0] <= W),
                              np.logical_and(ips[:, 1] >= 0, ips[:, 1] <= H))

        for p in ips[mask]:
            cv2.circle(im,tuple(p.astype(int)),2,(0,255,0),-1)

        # using model to predict proposal
        with torch.no_grad():
            pred = net.detect(pc_input.reshape(-1,8*1024,4))

        pred = pred[np.logical_and(pred[:,0]>CONF_THRES,pred[:,1]>IOU_THRES)]
        pred_bbox= pred[:,9:]
        proposals = pred[:,2:9]
        proposals = np.concatenate([np.ones((proposals.shape[0],1)),proposals],axis = 1)
        pred_bbox = np.concatenate([np.ones((pred_bbox.shape[0],1)),pred_bbox],axis = 1)
        bbox = drawBox3d(label,T_02,im,(0,0,255))
        p_bbox = drawBox3d(proposals,T_02,im,(0,255,0))
        g_bbox = drawBox3d(pred_bbox,T_02,im,(255,0,0))

        canvas = float2BEV(pc_camera,[bbox,g_bbox,p_bbox],pps,[(0,0,255),(255,0,0),(255,255,255)])

        if not os.path.exists(SAVE_PATH):
            os.mkdir(SAVE_PATH)
        cv2.imwrite(SAVE_PATH+lidar_file.split('.')[0]+'_project.jpg',im)
        cv2.imwrite(SAVE_PATH+lidar_file.split('.')[0]+'_bev.jpg',canvas)

        print("%s saved"%(SAVE_PATH+lidar_file.split('.')[0]+'_project.jpg'))