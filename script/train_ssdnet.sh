#!/bin/bash

export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python train.py \
  --network 'ssdnet' \
  --lr 0.5 \
  --lr-ratio 0.002 \
  --gpu 4,5,6,7 \
  --data ~/dataset/imagenet \
  --batch 1024 \
  --min-crop 0.111 \
  --epoch 300 \
  --parallel 40 \
  # --load './exported/ssdnet_imagenet.npz'
  # --size 224 \
  # --logdir 'ssdnet' \
