#!/bin/bash

export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""

if [[ "$1" == "bf16" ]]
then
    ARGS="$ARGS --bf16"
    echo "### running bf16 datatype"
fi

if [[ "$2" == "jit" ]]
then
    ARGS="$ARGS --jit"
    echo "### running jit mode"
fi

CORES=`lscpu | grep Core | awk '{print $4}'`
SOCKETS=`lscpu | grep Socket | awk '{print $2}'`
TOTAL_CORES=`expr $CORES \* $SOCKETS`

KMP_SETTING="KMP_AFFINITY=granularity=fine,compact,1,0"

export OMP_NUM_THREADS=$TOTAL_CORES
export $KMP_SETTING

echo -e "### using OMP_NUM_THREADS=$TOTAL_CORES"
echo -e "### using $KMP_SETTING\n\n"
sleep 3

### inference ###
export TRAIN=0
time python tools/test_net.py $ARGS -i 200 --enable-profiling --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" TEST.IMS_PER_BATCH 1 MODEL.DEVICE cpu

