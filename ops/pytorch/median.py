# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact ac9a1f703757537a3d54cc1ac1febeca410316c0 core PyTorch median recipe.
Seed, shape, dtype, dimensions and keepdim follow benchmark/test_median.py.
"""
import torch
from _kernelgen import generate_tensor_input
OP_NAME = 'median'
IS_INPLACE = False
DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int16, torch.int32]
_BINDINGS = [{'shape': [2], 'params': {}}, {'shape': [64], 'params': {}}, {'shape': [65], 'params': {}}, {'shape': [128], 'params': {}}, {'shape': [256], 'params': {}}, {'shape': [257], 'params': {}}, {'shape': [1024], 'params': {}}, {'shape': [4096], 'params': {}}, {'shape': [4, 8, 16, 8], 'params': {}}]

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
