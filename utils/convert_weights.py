import torch
import torch.nn as nn
from models.STD import PGM,STD
from datetime import datetime
import os

if __name__ == '__main__':
    WEIGHTS1_PATH = "/data/usr/zhengyu/exp/STD/2021-08-19_09-39/checkpoints/best.pt"
    WEIGHTS2_PATH = "/data/usr/zhengyu/exp/STD_SECOND/2021-08-21_13-12/checkpoints/best.pt"
    WEIGHTS_PATH = "/data/usr/zhengyu/STD_results/weights"
    model1 = PGM(0).cuda().eval()
    checkpoint = torch.load(WEIGHTS1_PATH)
    model1.load_state_dict(checkpoint['model_state_dict'])
    model2 = nn.DataParallel(STD().cuda().eval())
    checkpoint = torch.load(WEIGHTS2_PATH)
    model2.load_state_dict(checkpoint['model_state_dict'])
    state_dict = {
        'model1_state_dict':model1.state_dict(),
        'model2_state_dict':model2.module.state_dict()
    }
    save_dir = "{0}/{1}".format(WEIGHTS_PATH,datetime.now().strftime('%Y-%m-%d_%H-%M'))
    os.mkdir(save_dir)
    torch.save(state_dict,"{0}/best.pt".format(save_dir))
    print("Save at {0}/best.pt".format(save_dir))
