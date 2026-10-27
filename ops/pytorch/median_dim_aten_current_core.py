# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact original median_dim ATen callable, positional ABI and core recipe."""
from copy import deepcopy
import torch
from _kernelgen import generate_tensor_input

OP_NAME = 'median_dim_aten_current_core'
IS_INPLACE = False
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int16, torch.int32]
_BINDINGS = [
    {'shape': [9, 7], 'params': {'dim': 0, 'keepdim': False}},
    {'shape': [9, 7], 'params': {'dim': -1, 'keepdim': True}},
    {'shape': [384, 7], 'params': {'dim': 0, 'keepdim': False}},
    {'shape': [5, 9], 'params': {'dim': 1, 'keepdim': False}},
]


def native():
    return torch.ops.aten.median.dim


def grid():
    return deepcopy(_BINDINGS)


def key_shape(binding):
    return str(binding)


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    value = generate_tensor_input(binding['shape'], dtype, device)
    return (value, binding['params']['dim'], binding['params']['keepdim']), {}


def config(binding, dtype):
    shape = binding['shape']
    stride, product = [], 1
    for size in reversed(shape):
        stride.insert(0, product)
        product *= size
    return dict(
        source_commit='ac9a1f703757537a3d54cc1ac1febeca410316c0',
        source='benchmark/test_median.py', seed=0,
        shape=shape, params=binding['params'], dtype=str(dtype),
        native_api='torch.ops.aten.median.dim',
        inputs=dict(args=[dict(shape=shape, dtype=str(dtype), stride=stride,
                               requires_grad=False), binding['params']['dim'],
                          binding['params']['keepdim']], kwargs={}),
        input_recipe='Original _median_dim_build_inputs_fn positional ABI; manual_seed(0), generate_tensor_input',
    )
