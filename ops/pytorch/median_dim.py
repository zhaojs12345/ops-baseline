# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact ac9a1f703757537a3d54cc1ac1febeca410316c0 core PyTorch median recipe.
Seed, shape, dtype, dimensions and keepdim follow benchmark/test_median.py.
"""
import torch
from _kernelgen import generate_tensor_input
OP_NAME = 'median_dim'
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int16, torch.int32]
_BINDINGS = [{'shape': [9, 7], 'params': {'dim': 0, 'keepdim': False}}, {'shape': [9, 7], 'params': {'dim': -1, 'keepdim': True}}, {'shape': [384, 7], 'params': {'dim': 0, 'keepdim': False}}, {'shape': [5, 9], 'params': {'dim': 1, 'keepdim': False}}]

def native():
    return torch.median

def grid():
    from copy import deepcopy
    return deepcopy(_BINDINGS)

def key_shape(binding):
    return str(binding)

def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    inp = generate_tensor_input(binding['shape'], dtype, device)
    return (inp,), dict(binding['params'])

def config(binding, dtype):
    return dict(source_commit='ac9a1f703757537a3d54cc1ac1febeca410316c0',
                source='benchmark/test_median.py', seed=0, shape=binding['shape'],
                params=binding['params'], dtype=str(dtype),
                input_recipe='generate_tensor_input, unchanged dtype/device/layout',
                native_api='torch.median')
