#!/bin/bash

export TENSORPACK_DATASET='/root/dataset/tensorpack_data'

python ssdnet.py \
  --network 'ssdnet' \
  --lr 0.5 \
  --lr-ratio 0.004 \
  --gpu 4,5,6,7 \
  --data ~/dataset/imagenet \
  --batch 1024 \
  --min-crop 0.16 \
  --epoch 300 \
  --parallel 24 \
  # --size 224 \
  # --logdir 'ssdnet' \
  # --load './exported/ssdnet_ms.npz'
