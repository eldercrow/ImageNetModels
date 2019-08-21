#!/bin/bash

export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python train.py \
  --network 'ssdnetv3' \
  --lr 0.25 \
  --lr-ratio 0.004 \
  --gpu 0,1,2,3,4,5,6,7 \
  --data ~/dataset/imagenet \
  --batch 1024 \
  --min-crop 0.111 \
  --epoch 300 \
  --parallel 64 \
  # --load './exported/ssdnet_imagenet.npz'
  # --size 224 \
  # --logdir 'ssdnet' \
