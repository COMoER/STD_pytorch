import numpy as np
import cv2

classes = ['Car','Pedestrian','Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']

def in_side(src,label):
    '''

    src: N,4
    label:G,9
    return list of  points in each ground truth
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



def float2BEV(ps,all_bbox:list,inside,all_color:list):
    '''
    :param ps: N,4 x,y,z,r BEV is X-Z
    :param all_bbox: list[N,8,3]
    :param inside: points inside the ground truth
    :param color: list[tuple of color]
    :return: the BEV image
    '''
    # range
    x_min = np.min(ps[:, 0])
    x_max = np.max(ps[:, 0])
    y_min = np.min(ps[:, 2])
    y_max = np.max(ps[:, 2])
    x_range = x_max - x_min
    y_range = y_max - y_min

    # create an empty BEV canvas
    canvas = np.zeros((640, 640,3), np.uint8)

    # points
    ps[:, 0] = (ps[:, 0] - x_min) / x_range * 640
    ps[:, 2] = 640-(ps[:, 2] - y_min) / y_range * 640
    for p in ps:
        cv2.circle(canvas,tuple(p[[0,2]].astype(int)),1,(0,255,0),-1)

    inside[:,0] = (inside[:, 0] - x_min) / x_range * 640
    inside[:,2] = 640-(inside[:, 2] - y_min) / y_range * 640

    inside = inside[:,[0,2]]

    for p in inside:
        cv2.circle(canvas,tuple(p.astype(int)),1,(255,255,255),-1)

    for i,(bbox,color) in enumerate(zip(all_bbox,all_color)):
        bbox[:, :, 0] = (bbox[:, :, 0] - x_min) / x_range * 640
        bbox[:, :, 2] = 640 - (bbox[:, :, 2] - y_min) / y_range * 640

        bbox = bbox[:, :, [0, 2]]
        for g in bbox:
            g = g.astype(int)
            for i in range(4):
                cv2.line(canvas,tuple(g[i]),tuple(g[(i+1)%4]),color,2)
                cv2.line(canvas,tuple(g[i+4]),tuple(g[(i+1)%4+4]),color,2)
                cv2.line(canvas,tuple(g[i]),tuple(g[i+4]),color,2)

    return canvas
def load_calib(calib_path):
    '''
    :param calib_path:the calibration file path
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

    :param label: N*7 the box you want to draw on the image [cls,center,size,rotate_y]
    :param P: 4*4 transform matrix
    :param im: the image to draw box on it
    :param color: the corresponding color to each group of boxes
    :return: bbox N,8,3 using eight points format to present the boxes
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
    return bbox
