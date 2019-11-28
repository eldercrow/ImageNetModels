#!/bin/bash

# export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python train.py \
  --network 'ssdnet_ig' \
  --lr 0.25 \
  --lr-ratio 0.001 \
  --gpu 4,5,6,7 \
  --data ~/dataset/imagenet \
  --batch 256 \
  --min-crop 0.111 \
  --epoch 300 \
  --parallel 12 \
  # --load './exported/ssdnet_imagenet.npz'
  # --size 224 \
  # --logdir 'ssdnet' \
