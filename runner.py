#! /usr/bin/python3
# -*- coding: utf-8 -*-

import cv2
import matplotlib
import os
# Fix no $DISPLAY environment variable
matplotlib.use('Agg')
# Fix deadlock in DataLoader
cv2.setNumThreads(0)

from pprint import pprint

from config import cfg
from core.train import train_net
from core.test import test_net
from core.inference import inference_net


if __name__ == '__main__':
    cfg.CONST.WEIGHTS = "/home/chli/github/GRNet/GRNet-KITTI.pth"
    if not os.path.exists(cfg.CONST.WEIGHTS):
        print("Weights file doesn't exists!")
        exit()

    print('Use config:')
    pprint(cfg)

    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.CONST.DEVICE

    inference_net(cfg)

