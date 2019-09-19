#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: shufflenet.py

import argparse
import math
import numpy as np
import os
import cv2

# from memory_saving_gradients import gradients as memsave_gradients
import tensorflow as tf
# from tensorflow.python import ops as tfops

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.scope_utils import under_name_scope
from tensorpack.utils import logger
from tensorpack.utils.gpu import get_num_gpu

from imagenet_utils import ImageNetModel, eval_classification, get_imagenet_dataflow

# def gradients(ys, xs, grad_ys=None, **kwargs):
#     tensors = ['tower0/inc3/conv2/conv_e/bn/output', 'tower0/inc6/conv2/conv_e/bn/output']
#     return memsave_gradients(ys, xs, grad_ys, checkpoints=tensors, **kwargs)
#
# tfops.__dict__['gradients'] = gradients
# tf.__dict__['gradients'] = gradients

class Model(ImageNetModel):
    weight_decay = 4e-5
    conf_regularization = 0.1

    def set_logit_fn(self, logit_fn):
        self.logit_fn = logit_fn

    def get_logits(self, image):
        return self.logit_fn(image)


def get_data(name, batch, min_crop=0.08):
    isTrain = (name == 'train')

    if isTrain:
        augmentors = [
            # use lighter augs if model is too small
            imgaug.GoogleNetRandomCropAndResize(crop_area_fraction=(min_crop, 1.)),
            imgaug.RandomOrderAug(
                [imgaug.BrightnessScale((0.6, 1.4), clip=False),
                 imgaug.Contrast((0.6, 1.4), clip=False),
                 imgaug.Saturation(0.4, rgb=False),
                 # rgb-bgr conversion for the constants copied from fb.resnet.torch
                 imgaug.Lighting(0.1,
                                 eigval=np.asarray(
                                     [0.2175, 0.0188, 0.0045][::-1]) * 255.0,
                                 eigvec=np.array(
                                     [[-0.5675, 0.7192, 0.4009],
                                      [-0.5808, -0.0045, -0.8140],
                                      [-0.5836, -0.6948, 0.4203]],
                                     dtype='float32')[::-1, ::-1]
                                 )]),
            imgaug.Flip(horiz=True),
        ]
    else:
        augmentors = [
            imgaug.ResizeShortestEdge(256, cv2.INTER_CUBIC),
            imgaug.CenterCrop((224, 224)),
        ]
    return get_imagenet_dataflow(
        args.data, name, batch, augmentors, args.parallel)


def get_config(model, nr_tower):
    batch = TOTAL_BATCH_SIZE // nr_tower

    logger.info("Running on {} towers. Batch size per tower: {}".format(nr_tower, batch))
    dataset_train = get_data('train', batch, args.min_crop)
    dataset_val = get_data('val', batch, args.min_crop)

    # max_epoch = int(np.ceil(max_iter / base_step_size))

    step_size = 1280000 // TOTAL_BATCH_SIZE
    max_iter = int(step_size * args.epoch)
    max_epoch = (max_iter // step_size) + 1
    lr = args.lr
    lr_decay = np.exp(np.log(args.lr_ratio) / max_epoch)
    callbacks = [
        ModelSaver(),
        ScheduledHyperParamSetter('learning_rate',
                                  [(0, lr*0.01), (step_size//2, lr)],
                                  interp='linear', step_based=True),
        HyperParamSetterWithFunc('learning_rate',
                                 lambda e, x: x * lr_decay if e > 0 else x),
        ScheduledHyperParamSetter('bn_momentum',
                                  [(0, 0.9), (max_epoch//3, 0.99), (max_epoch//3*2, 0.999)]),
        EstimatedTimeLeft()
    ]
    try:
        callbacks.append(ScheduledHyperParamSetter('dropblock_keep_prob',
                                                   [(0, 0.9), (max_epoch-1, 1.0)],
                                                   interp='linear'))
    except:
        logger.warn('Could not add dropblock_keep_prob callback.')
        pass
    infs = [ClassificationError('wrong-top1', 'val-error-top1'),
            ClassificationError('wrong-top5', 'val-error-top5')]
    if nr_tower == 1:
        # single-GPU inference with queue prefetch
        callbacks.append(InferenceRunner(QueueInput(dataset_val), infs))
    else:
        # multi-GPU inference (with mandatory queue prefetch)
        callbacks.append(DataParallelInferenceRunner(
            dataset_val, infs, list(range(nr_tower))))

    return TrainConfig(
        model=model,
        dataflow=dataset_train,
        callbacks=callbacks,
        steps_per_epoch=step_size,
        max_epoch=max_epoch,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', help='comma separated list of GPU(s) to use.')
    parser.add_argument('--data', help='ILSVRC dataset dir')
    parser.add_argument('--network', help='network name', type=str, default='ssdnet')
    parser.add_argument('--batch', type=int, default=1024, help='total batch size')
    parser.add_argument('--min-crop', type=float, default=0.08, help='minimum crop size for augmentation')
    parser.add_argument('--epoch', type=int, default=300, help='total epoch size')
    parser.add_argument('--lr', type=float, default=0.5, help='initial learning rate')
    parser.add_argument('--lr-ratio', type=float, default=0.004, help='lr ratio between start and end')
    parser.add_argument('--parallel', type=int, default=4, help='cpu workers for preprocessing')
    parser.add_argument('--load', help='path to load a model from')
    parser.add_argument('--eval', action='store_true')
    parser.add_argument('--flops', action='store_true', help='print flops and exit')
    args = parser.parse_args()

    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    if args.batch != 1024:
        logger.warn("Total batch size != 1024, you need to change other hyperparameters to get the same results.")
    TOTAL_BATCH_SIZE = args.batch

    get_logits = getattr(__import__('network.{}'.format(args.network), fromlist=['get_logits']), 'get_logits')

    model = Model()
    model.set_logit_fn(get_logits)

    if args.eval:
        batch = 128    # something that can run on one gpu
        ds = get_data('val', batch)
        eval_classification(model, get_model_loader(args.load), ds)
    elif args.flops:
        # manually build the graph with batch=1
        with TowerContext('', is_training=False):
            model.build_graph(
                tf.placeholder(tf.float32, [1, 224, 224, 3], 'input'),
                tf.placeholder(tf.int32, [1], 'label')
            )
        model_utils.describe_trainable_vars()

        tf.profiler.profile(
            tf.get_default_graph(),
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.float_operation())
        logger.info("Note that TensorFlow counts flops in a different way from the paper.")
        logger.info("TensorFlow counts multiply+add as two flops, however the paper counts them "
                    "as 1 flop because it can be executed in one instruction.")
    else:
        name = args.network
        logger.set_logger_dir(os.path.join('train_log', name))

        nr_tower = max(get_num_gpu(), 1)
        config = get_config(model, nr_tower)
        if args.load:
            config.session_init = get_model_loader(args.load)
        launch_train_with_config(config, SyncMultiGPUTrainerParameterServer(nr_tower))
