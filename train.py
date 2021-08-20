import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel
import torch.distributed

from utils.dataset import pc_dataloader,std_collate_fn
from torch.utils.data.dataloader import DataLoader
from models.STD import PGM

import os
import datetime
import logging
from pathlib import Path
import importlib
import shutil
from tqdm import tqdm
import numpy as np
import time

EXP_DIR = '/data/usr/zhengyu/exp'

def main():
    def log_string(str):
        logger.info(str)
        print(str)

    os.environ["CUDA_VISIBLE_DEVICES"] = '0,1,2'
    # create dir
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path(EXP_DIR)

    experiment_dir.mkdir(exist_ok=True)

    experiment_dir = experiment_dir.joinpath('STD')
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

    # torch.distributed.init_process_group(backend="nccl")
    model = PGM(0).cuda()

    # model = nn.parallel.DistributedDataParallel(model)

    last_exp = "2021-08-18_19-33"

    try:
        checkpoint = torch.load(str(exp_dir.joinpath(last_exp)) + '/checkpoints/best.pt')
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0

    INTIAL_LR = 0.0001
    AFTER_LR = 0.00005

    optimizer = torch.optim.Adam(
            model.parameters(),
            lr=INTIAL_LR,
            betas=(0.9, 0.999),
            eps=1e-08
        )

    EPOCH = 100

    best_loss = 0

    data = pc_dataloader()
    trainDataloader = DataLoader(data, batch_size=1, shuffle=True, collate_fn=std_collate_fn)

    for epoch in range(start_epoch, EPOCH):
        log_string('**** Epoch %d/%s ****' % (epoch+1,EPOCH))

        # adjust lr
        if epoch == 80:
            for p in optimizer.param_groups:
                p['lr']  = AFTER_LR
        loss_sum = 0
        proposal_num = 0
        for i, (points, target) in tqdm(enumerate(trainDataloader), total=len(trainDataloader), smoothing=0.9):
            optimizer.zero_grad()

            proposals,features,loss = model(points,target)
            loss.backward()
            optimizer.step()



            loss_sum += loss
            proposal_num += len(proposals)

        log_string('Training mean loss: %f' % (loss_sum / len(trainDataloader)))
        log_string("Training output proposal: %f"%(proposal_num/len(trainDataloader)))

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