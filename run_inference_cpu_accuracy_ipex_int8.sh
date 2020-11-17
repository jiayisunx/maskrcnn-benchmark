#!/bin/bash
export DNNL_PRIMITIVE_CACHE_CAPACITY=1024

ARGS=""

if [[ "$1" == "dnnl" ]]
then
    ARGS="$ARGS --dnnl"
    echo "### running auto_dnnl mode"
fi

if [[ "$2" == "int8" ]]
then
    ARGS="$ARGS --int8"
    echo "### running int8 datatype"
fi

if [[ "$3" == "jit" ]]
then
    ARGS="$ARGS --jit"
    echo "### running jit mode"
fi

if [[ "$4" == "calibration" ]]
then
   ARGS="$ARGS --calibration --iter-calib 20"
   echo "### running calibration"
fi

# ARGS="$ARGS --configure-dir $5"
# echo "### configure direction: $5"


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
time python tools/test_net.py --config-file "configs/e2e_mask_rcnn_R_50_FPN_1x_coco2017_inf.yaml" --ipex $ARGS TEST.IMS_PER_BATCH 2 MODEL.DEVICE cpu
