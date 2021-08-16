# the implement of VFE by https://github.com/Hqss/VoxelNet_PyTorch.git
import torch
import torch.nn as nn
import numpy as np

def preprocess(points,proposals):
    '''
    points: list of points to each proposal
    proposals: P,8
    '''
    device = proposals.device
    with torch.no_grad():
        proposals_number = []
        proposals_feature = []
        for x,proposal in zip(points,proposals):
            N,_ = x.shape
            # first sample and voxelization , which don't require gradient
            # when the points inside the proposal over 512, then sample to 512
            if N > 512:
                x = x[np.random.choice(range(N), 512)]
            # voxelization 6*6*6
            voxel_size = (proposal[3:6]) / 6
            zero_start = proposal[3:6].clone()
            zero_start[1] = zero_start[1]/2  # w
            zero_start[2] = zero_start[2]/2  # l
            # proposal [-l/2,l/2] [-w/2,w/2] [-h,0]
            voxel_index = torch.floor(
                (x[:, [1,2,0]] + zero_start) / voxel_size).to(torch.int) # N',(l_h,l_w,l_l)

            # [K, 3] coordinate buffer as described in the paper
            coordinate_buffer = torch.unique(voxel_index, dim = 0)

            K = len(coordinate_buffer)
            T = 35

            # [K, 1] store number of points in each voxel grid
            number_buffer = torch.zeros(size = torch.Size((6*6*6,)),dtype = torch.int,device=device)

            # [K, T, 131] feature buffer as described in the paper
            feature_buffer = torch.zeros(size = torch.Size((6*6*6,T,131)),dtype = torch.float32,device=device)

            # build a reverse index for coordinate buffer
            index_buffer = {}
            for i in range(K):
                index = coordinate_buffer[i].cpu().numpy()
                index_buffer[tuple(index)] = index[0]*6 + index[1]*6 + index[2]

            for voxel, point in zip(voxel_index, x):
                index = index_buffer[tuple(voxel.cpu().numpy())]
                number = number_buffer[index]
                if number < T:
                    feature_buffer[index, number, :131] = point
                    number_buffer[index] += 1
            proposals_number.append(number_buffer)
            proposals_feature.append(feature_buffer)
        proposals_number = torch.stack(proposals_number,dim = 0) # B*216
        proposals_feature = torch.stack(proposals_feature,dim = 0) # B*216*35*128
    return proposals_number,proposals_feature




class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(VFELayer, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.units = int(out_channels / 2)

        self.dense = nn.Sequential(nn.Linear(self.in_channels, self.units), nn.ReLU())
        self.batch_norm = nn.BatchNorm2d(self.units)


    def forward(self, inputs, mask):
        B,K,T,D = inputs.shape
        # [ΣK, T, in_ch] -> [ΣK, T, units]
        tmp = self.dense(inputs).permute(0,3,1,2)
        pointwise = self.batch_norm(tmp).permute(0,2,3,1)

        # [ΣK, 1, units]
        aggregated, _ = torch.max(pointwise, dim = 2, keepdim = True)

        # [ΣK, T, units]
        repeated = aggregated.expand(B,K,T,-1)

        # [ΣK, T, 2 * units]
        concatenated = torch.cat([pointwise, repeated], dim = 3)

        # [ΣK, T, 1] -> [ΣK, T, 2 * units]
        mask = mask.expand(B,K, T, 2 * self.units)

        concatenated = concatenated * mask.float()

        return concatenated


class PointsPool(nn.Module):
    def __init__(self):
        super(PointsPool, self).__init__()

        self.vfe1 = VFELayer(131, 128)
        self.vfe2 = VFELayer(128, 128)
        self.vfe3 = VFELayer(128,256)


    def forward(self,x):
        vmax, _ = torch.max(x, dim = 3, keepdim = True)
        mask = ~torch.isclose(vmax,torch.zeros(vmax.shape,device = vmax.device))  # [B,ΣK, T, 1]

        x = self.vfe1(x, mask)
        x = self.vfe2(x, mask)
        x = self.vfe3(x,mask)

        # [B,ΣK, 256] maxpool
        voxelwise, _ = torch.max(x, dim = 2)

        return voxelwise