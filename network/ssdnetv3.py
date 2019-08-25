# -*- coding: utf-8 -*-
# File: basemodel.py

# import argparse
# import cv2
# import os
import numpy as np
import tensorflow as tf

from contextlib import contextmanager

from tensorpack import *
from tensorpack.dataflow import imgaug
from tensorpack.tfutils import argscope, get_model_loader, model_utils
from tensorpack.tfutils.argscope import argscope #, get_arg_scope
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
# from tensorpack.tfutils.varreplace import custom_getter_scope
from tensorpack.utils.gpu import get_num_gpu
from tensorpack.utils import logger
from tensorpack.models import (
    Conv2D, Deconv2D, MaxPooling, BatchNorm, BNReLU, LinearWrap, GlobalAvgPooling)
from tensorpack.models.regularize import Dropout


@auto_reuse_variable_scope
def get_bn_momentum():
    mom = tf.get_variable('bn_momentum',
                          (),
                          dtype=tf.float32,
                          trainable=False,
                          initializer=tf.constant_initializer(0.9))
    return mom


@layer_register(use_scope=None)
def BNOnly(x):
    return BatchNorm('bn', x)


@contextmanager
def ssdnet_argscope():
    with argscope([Conv2D, MaxPooling, BatchNorm, GlobalAvgPooling], data_format='NHWC'), \
            argscope([Conv2D, FullyConnected], use_bias=False), \
            argscope([BatchNorm], momentum=get_bn_momentum()):
        yield


@layer_register(log_shape=True)
def DropBlock(x, keep_prob=None, block_size=5, drop_mult=1.0, data_format='NHWC'):
    '''
    DropBlock
    '''
    if (keep_prob is None) or (not get_current_tower_context().is_training) or (block_size == 0):
        return x

    drop_prob = (1.0 - keep_prob) * drop_mult

    assert data_format in ('NHWC', 'channels_last')
    feat_size = x.get_shape().as_list()[1]
    N = tf.to_float(tf.size(x))

    f2 = feat_size * feat_size
    b2 = block_size * block_size
    r = (feat_size - block_size + 1)
    r2 = r*r
    gamma = drop_prob * f2 / b2 / r2
    k = [1, block_size, block_size, 1]

    mask = tf.less(tf.random_uniform(tf.shape(x), 0, 1), gamma)
    mask = tf.cast(mask, dtype=x.dtype)
    mask = tf.nn.max_pool(mask, ksize=k, strides=[1, 1, 1, 1], padding='SAME')
    drop_mask = (1. - mask)
    drop_mask *= (N / tf.reduce_sum(drop_mask))
    drop_mask = tf.stop_gradient(drop_mask, name='drop_mask')
    return tf.multiply(x, drop_mask, name='dropped')


@layer_register(log_shape=True)
def AccuracyBoost(x):
    '''
    Accuracy boost block for bottleneck layers.
    '''
    nch = x.get_shape().as_list()[-1]
    g = GlobalAvgPooling('gpool', x)
    W = tf.get_variable('W', shape=(nch,), initializer=tf.variance_scaling_initializer(2.0))
    # g = BatchNorm('bn', tf.multiply(g, W))
    b = tf.get_variable('b', shape=(nch,), initializer=tf.zeros_initializer())
    g = tf.multiply(g, W)
    g = tf.add(g, b)
    ab = tf.reshape(tf.nn.sigmoid(g), (-1, 1, 1, nch))
    return tf.multiply(x, ab, name='res')


@layer_register(log_shape=True)
def DWConv(x, kernel, padding='SAME', stride=1, w_init=None, activation=BNReLU, data_format='NHWC'):
    '''
    Depthwise conv + BN + (optional) ReLU.
    We do not use channel multiplier here (fixed as 1).
    '''
    assert data_format in ('NHWC', 'channels_last')
    channel = x.get_shape().as_list()[-1]
    if not isinstance(kernel, (list, tuple)):
        kernel = [kernel, kernel]
    filter_shape = [kernel[0], kernel[1], channel, 1]

    if w_init is None:
        w_init = tf.variance_scaling_initializer(2.0)
    W = tf.get_variable('W', filter_shape, initializer=w_init)
    out = tf.nn.depthwise_conv2d(x, W, [1, stride, stride, 1], padding=padding, data_format=data_format)

    out = activation(out)
    return out


@layer_register(log_shape=True)
def LinearBottleneck(x, ich, och, kernel,
                     padding='SAME',
                     stride=1,
                     activation=None,
                     t=3,
                     use_ab=False,
                     w_init=None):
    '''
    mobilenetv2 linear bottlenet.
    '''
    if activation is None:
        activation = BNReLU if kernel > 3 else BNOnly

    out = Conv2D('conv_e', x, int(ich*t), 1, activation=BNReLU)
    out = DWConv('conv_d', out, kernel, padding, stride, w_init, activation)
    if use_ab and activation == BNReLU:
        out = AccuracyBoost('ab', out)
    out = Conv2D('conv_p', out, och, 1, activation=BNOnly)
    return out


@layer_register(log_shape=True)
def DownsampleBottleneck(x, ich, och, kernel,
                         padding='SAME',
                         stride=2,
                         activation=None,
                         t=3,
                         use_ab=False,
                         w_init=None):
    '''
    downsample linear bottlenet.
    '''
    if activation is None:
        activation = BNReLU if kernel > 3 else BNOnly

    out_e = Conv2D('conv_e', x, ich*t, 1, activation=BNReLU)
    out_d = DWConv('conv_d', out_e, kernel, padding, stride, w_init, activation)
    out_m = DWConv('conv_m', out_e, kernel, padding, stride, w_init, activation)
    out = tf.concat([out_d, out_m], axis=-1)
    if use_ab and activation == BNReLU:
        out = AccuracyBoost('ab', out)
    out = Conv2D('conv_p', out, och, 1, activation=BNOnly)
    return out


@layer_register(log_shape=True)
def inception(x, ch, k, stride, t=3, swap_block=False, activation=None, use_ab=False):
    '''
    ssdnet inception layer.
    '''
    if stride == 1:
        oi = LinearBottleneck('conv1', x, ch, ch, k, \
                              stride=stride, t=t, activation=activation, use_ab=use_ab)
    else:
        oi = DownsampleBottleneck('conv1', x, ch//2, ch, 4, \
                                  stride=stride, t=t, activation=activation, use_ab=use_ab)
    oi = tf.split(oi, 2, axis=-1)
    o1 = oi[0]
    o2 = oi[1] + LinearBottleneck('conv2', oi[1], ch//2, ch//2, k, \
                                  t=t, activation=activation, use_ab=use_ab)

    if not swap_block:
        out = tf.concat([o1, o2], -1)
    else:
        out = tf.concat([o2, o1], -1)

    # residual if stride is 1
    if stride == 1:
        # lch = x.get_shape().as_list()[-1]
        # rch = out.get_shape().as_list()[-1]
        # if lch != rch:
        #     x = Conv2D('convp', x, rch, 1, activation=None)
        #     x = BatchNorm('convp/bn', x)
        out = tf.add(out, x)
    return out


def get_logits(image, num_classes=1000):
    #
    with ssdnet_argscope():
        # dropblock
        if get_current_tower_context().is_training:
            dropblock_keep_prob = tf.get_variable('dropblock_keep_prob', (),
                                        dtype=tf.float32,
                                        trainable=False)
        else:
            dropblock_keep_prob = None

        l = image #tf.transpose(image, perm=[0, 2, 3, 1])
        # conv1
        l = Conv2D('conv1', l, 32, 4, strides=2, activation=None, padding='SAME')
        with tf.variable_scope('conv1'):
            l = BNReLU(tf.concat([l, -l], axis=-1))
        l = MaxPooling('pool1', l, 2)
        # conv2
        l = LinearBottleneck('conv2', l, 32, 32, 5, t=3, use_ab=True)
        # l = l + LinearBottleneck('conv3', l, 24, 24, 5, t=3, use_ab=True)

        ch_all = [48, 80, 112]
        iters = [2, 4, 4]
        mults = [4, 5, 6]
        bsize = [3, 3, 3]
        strides = [2, 2, 2]

        hlist = []
        for ii, (ch, it, mu, bs, ss) in enumerate(zip(ch_all, iters, mults, bsize, strides)):
            use_ab = (ii < 2)
            for jj in range(it):
                name = 'inc{}/{}'.format(ii, jj)
                stride = ss if jj == 0 else 1
                k = 3 if (jj < it // 2) else 5
                swap_block = True if jj % 2 == 1 else False
                l = inception(name, l, ch, k, stride, t=mu, swap_block=swap_block, use_ab=use_ab)
            l = DropBlock('inc{}/drop'.format(ii), l, keep_prob=dropblock_keep_prob, block_size=bs)

        nch = ch_all[-1] * mults[-1]
        l = Conv2D('convf', l, nch, 1, activation=BNReLU)
        l = GlobalAvgPooling('poolf', l)

        ### position sensitve fc ---
        # hh, ww = l.get_shape().as_list()[1:3]
        # l = tf.reshape(l, [-1, hh*ww, ch_all[-1]])
        #
        # ll = tf.split(l, hh*ww, axis=1)
        # ll = [tf.layers.flatten(li) for li in ll]
        # ll = [FullyConnected('psfc{:02d}'.format(ii), li, 24, activation=BNReLU) for ii, li in enumerate(ll)]
        # l = tf.concat(ll, axis=-1)
        ### --- position sensitve fc

        fc = FullyConnected('fc', l, 1280, activation=BNReLU)
        fc = Dropout(fc, keep_prob=0.8)
        logits = FullyConnected('linear', fc, num_classes, use_bias=True)
    return logits

