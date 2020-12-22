#!/bin/bash

### training ###
export TRAIN=1
time python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_tra.yaml"\
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 MODEL.DEVICE cpu

