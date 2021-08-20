import os

import torch
import torch.nn as nn

from utils.dataset import pc_dataloader,std_collate_fn
from torch.utils.data.dataloader import DataLoader
from models.group_pointcloud import preprocess
from models.common import in_side
from models.STD import PGM,STD


import datetime
import logging
from pathlib import Path
from tqdm import tqdm
import numpy as np

EXP_DIR = '/data/usr/zhengyu/exp'
PGM_WEIGHT_DIR = '/data/usr/zhengyu/exp/STD/2021-08-19_09-39/checkpoints/best.pt'

def main():
    def log_string(str):
        logger.info(str)
        print(str)


    # create dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(EXP_DIR)

    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = experiment_dir.joinpath('STD_SECOND')
    experiment_dir.mkdir(exist_ok=True)

    exp_dir = experiment_dir

    experiment_dir = experiment_dir.joinpath(timestr)
    experiment_dir.mkdir(exist_ok=True)

    checkpoints_dir = experiment_dir.joinpath('checkpoints')
    checkpoints_dir.mkdir(exist_ok=True)

    log_dir = experiment_dir.joinpath('logs')
    log_dir.mkdir(exist_ok=True)

    # LOG
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('{0}/exp_{1}.txt'.format(log_dir,timestr))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # load first stage pt
    frozen_model = PGM(0)
    first_stage_path =
    checkpoint = torch.load(PGM_WEIGHT_DIR)
    frozen_model.load_state_dict(checkpoint['model_state_dict'])
    nn.DataParallel(frozen_model.cuda(1), [1, 2], 2).eval()
    # torch.distributed.init_process_group(backend="nccl")
    model = nn.DataParallel(STD().cuda(1),[1,2],1)

    # model = nn.parallel.DistributedDataParallel(model)

    last_exp = "2021-08-20_11-18"

    try:
        checkpoint = torch.load(str(exp_dir.joinpath(last_exp)) + '/checkpoints/best.pt')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    INTIAL_LR = 0.0001
    DELAY_RATE = 0.1

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=INTIAL_LR,
            betas=(0.9, 0.999),
            eps=1e-08
        )

    EPOCH = 50

    data = pc_dataloader(device=1)
    trainDataloader = DataLoader(data, batch_size=1, shuffle=True, collate_fn=std_collate_fn)

    cls_loss_sum = 0
    box_loss_sum = 0
    iou_loss_sum = 0
    closs_sum = 0
    cls_num = 0
    target_num = 0

    for epoch in range(start_epoch, EPOCH):
        log_string('**** Epoch %d/%s ****' % (epoch+1,EPOCH))

        # adjust lr
        if epoch >= 39 and (epoch+1) % 5 == 0:
            for p in optimizer.param_groups:
                p['lr']  *= DELAY_RATE
        loss_sum = 0
        for i, (points, target) in tqdm(enumerate(trainDataloader), total=len(trainDataloader), smoothing=0.9):
            optimizer.zero_grad()

            proposals,features = frozen_model(points)

            # STD_SECOND forward
            cls_loss,box_loss,iou_loss,closs = model(proposals,points,features,target)
            loss = cls_loss + box_loss + iou_loss + closs
            loss.backward()
            optimizer.step()
            if not np.isclose(cls_loss.detach().cpu().numpy(),0).any():
                cls_num += 1
                cls_loss_sum += cls_loss
                print("target %d cls loss is %.3f"%(i,cls_loss))
            if not np.isclose(box_loss.detach().cpu().numpy(),0).any():
                target_num += 1
                box_loss_sum += box_loss
                print("target %d box loss is %.3f"%(i,box_loss))
                iou_loss_sum += iou_loss
                closs_sum += closs

        log_string('Training mean cls_loss: %f' % (cls_loss_sum / cls_num))
        if target_num > 0:
            log_string('Training mean box_loss: %f' % (box_loss_sum / target_num))
            log_string('Training mean iou_loss: %f' % (iou_loss_sum / target_num))
            log_string('Training mean closs: %f' % (closs_sum / target_num))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best.pt'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')


if __name__ == '__main__':
    main()