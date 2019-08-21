#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: dump-model-params.py

import numpy as np
import six
import argparse
import os, sys
import tensorflow as tf

from tensorpack.tfutils import varmanip
from tensorpack.tfutils.common import get_op_tensor_name


def _merge_sparsity_mask(params):
    '''
    Merge weights and masks
    '''
    r_params = {}
    for k, v in params.items():
        rk = k.replace('/mask:0', 'W:0')
        if rk not in r_params:
            r_params[rk] = v.copy()
        else:
            r_params[rk] *= v
    return r_params


def _measure_sparsity(params):
    '''
    '''
    num_w = 0
    num_zw = 0

    for k, v in params.items():
        if k.endswith('W:0'):
            num_w += v.size
            num_zw += np.sum(v == 0)
    sparsity = float(num_zw) / float(num_w)
    return sparsity


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Keep only TRAINABLE and MODEL variables in a checkpoint.')
    # parser.add_argument('--meta', help='metagraph file', required=True)
    parser.add_argument(dest='input', help='input model file, has to be a TF checkpoint')
    parser.add_argument(dest='output', help='output model file, can be npz or TF checkpoint')
    args = parser.parse_args()

    # this script does not need GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = ''

    # loading...
    if args.input.endswith('.npz'):
        dic = np.load(args.input)
    else:
        dic = varmanip.load_chkpt_vars(args.input)
    dic = {get_op_tensor_name(k)[1]: v for k, v in six.iteritems(dic)}

    # dic = _merge_sparsity_mask(dic)
    dic_to_dump = {}
    postfixes = ['W:0', 'b:0', 'beta:0', 'gamma:0', 'EMA:0', \
                 'weight:0', 'kernel:0', 'bias:0', 'moving_mean:0', 'moving_variance:0']
    for k, v in dic.items():
        # print(k)
        found = False
        for p in postfixes:
            if p in k:
                found = True
                break
        if found:
            dic_to_dump[k] = v

    # sparsity = _measure_sparsity(dic_to_dump)
    # print('Net sparsity = {}'.format(sparsity))
    varmanip.save_chkpt_vars(dic_to_dump, args.output)
