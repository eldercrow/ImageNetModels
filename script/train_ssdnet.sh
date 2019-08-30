#!/bin/bash

# export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python train.py \
  --network 'ssdnet' \
  --lr 0.25 \
  --lr-ratio 0.002 \
  --gpu 0,1 \
  --data ~/dataset/imagenet \
  --batch 384 \
  --min-crop 0.111 \
  --epoch 300 \
  --parallel 6 \
  # --load './exported/ssdnet_imagenet.npz'
  # --size 224 \
  # --logdir 'ssdnet' \
