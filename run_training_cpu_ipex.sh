#!/bin/bash

ARGS=""

if [[ "$1" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$2" == "bf16" ]]
then
    ARGS="$ARGS --mix-precision"
    echo "### running bf16 datatype"
fi

### training ###
time python tools/train_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_tra.yaml" --ipex $ARGS\
       SOLVER.IMS_PER_BATCH 2 TEST.IMS_PER_BATCH 1 SOLVER.MAX_ITER 720000 SOLVER.STEPS "(480000, 640000)" SOLVER.BASE_LR 0.0025 MODEL.DEVICE cpu

