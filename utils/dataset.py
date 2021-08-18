import torch
from torch.utils.data.dataset import Dataset
import numpy as np
import os
import random
SIZE = 200
POINTS = 8*1024 # 8k
torch.random.manual_seed(0)
np.random.seed(0)
from tqdm import tqdm
classes = ['Car','Pedestrian','Cyclist', 'Van', 'Truck', 'Person_sitting', 'Tram', 'Misc', 'DontCare']
def convert_calib(calib_data):
    '''
    return Lidar2camera 4*4 T
    '''
    T = np.float32(calib_data[0,1:]).reshape(3,4)
    T = np.concatenate([T,np.float32([[0,0,0,1]])],axis = 0)
    return T

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


class pc_dataloader(Dataset):
    def __init__(self,location = "/data/kitti/KITTI/data_object_velodyne/training/velodyne/",
                 label_location = "/data/kitti/KITTI/training/label_2/",
                 calib_location = "/data/kitti/KITTI/training/calib/",
                    device = 0):
        super(pc_dataloader,self).__init__()
        self._device = torch.device('cuda',device)

        print("[INFO] loading Dataset")
        self._data = []
        for lidar_file in tqdm(os.listdir(location)[:SIZE]):
            # raw point cloud
            T = convert_calib(
                np.loadtxt(calib_location + lidar_file.split('.')[0] + '.txt', dtype=object, delimiter=' ',
                           skiprows=5).reshape(-1, 13))
            pc = np.fromfile(location+lidar_file,dtype = np.float32).reshape(-1,4)
            pc[:,:3] = (T[:3,:3]@(pc[:,:3].T)).T + T[:3,3].reshape(1,3)

            label = convert(np.loadtxt(label_location+lidar_file.split('.')[0]+'.txt',dtype = object,delimiter=' ').reshape(-1,15))

            indices = np.random.choice(range(len(pc)), POINTS)
            self._data.append([pc[indices],label]) # random sample POINTS
        # random.shuffle(self._data)

    def __getitem__(self,id):
        pc,label = self._data[id]
        pc = torch.from_numpy(pc).view(-1,4).to(self._device)
        label = torch.from_numpy(label).view(-1,9).to(self._device)
        # data augment
        return pc,label

    def __len__(self):
        return len(self._data)


def std_collate_fn(batch):
    if isinstance(batch[0], torch.Tensor):
        try:
            return torch.stack(batch,dim = 0)
        except:
            # label can't be stack
            return batch
    else:
        transposed = zip(*batch)
        return [std_collate_fn(samples) for samples in transposed]


