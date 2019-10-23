#!/bin/bash

export TENSORPACK_DATASET='/home/user/dataset/tensorpack_data'

python train.py \
  --network 'ssdnetv2_255' \
  --lr 0.25 \
  --lr-ratio 0.002 \
  --gpu 0,1 \
  --data ~/dataset/imagenet \
  --batch 512 \
  --target-shape 159 \
  --epoch 300 \
  --parallel 6 \
  --min-crop 0.25
