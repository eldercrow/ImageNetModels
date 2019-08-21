#!/bin/bash

export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python train.py \
  --network 'ssdnetv2' \
  --lr 0.25 \
  --lr-ratio 0.002 \
  --gpu 0,1,2,3 \
  --data ~/dataset/imagenet \
  --batch 512 \
  --epoch 300 \
  --parallel 24 \
  --min-crop 0.16
