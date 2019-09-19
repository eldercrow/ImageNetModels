# -*- coding: utf-8 -*-
# File: basemodel.py

# import argparse
# import cv2
# import os
import numpy as np
import tensorflow as tf

from contextlib import contextmanager

from tensorpack import *
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope
from tensorpack.models import BatchNorm


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


def combined_static_and_dynamic_shape(tensor):
    """Returns a list containing static and dynamic values for the dimensions.
    Returns a list of static and dynamic values for shape dimensions. This is
    useful to preserve static shapes when available in reshape operation.
    Args:
    tensor: A tensor of any type.
    Returns:
    A list of size tensor.shape.ndims containing integers or a scalar tensor.
    """
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(tensor)
    combined_shape = []
    for index, dim in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape
