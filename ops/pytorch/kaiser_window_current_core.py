# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact ac9 plain Kaiser benchmark core; native PyTorch CUDA factory only."""
import math
import torch
OP_NAME = 'kaiser_window_current_core'
IS_INPLACE = False
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
SHAPES = [(1073741824,), (64,64), (4096,4096), (64,512,512), (1024,1024,1024)]
SOURCE_COMMIT = 'ac9a1f703757537a3d54cc1ac1febeca410316c0'
def native():
    return torch.kaiser_window
def grid():
    return [dict(shape=list(s), variant=0) for s in SHAPES]
def key_shape(binding):
    return binding['shape']
def build_inputs(binding,dtype,device):
    return (),dict(window_length=math.prod(binding['shape']),periodic=True,beta=12.0,dtype=dtype,device=device)
def config(binding,dtype):
    return dict(source_commit=SOURCE_COMMIT,benchmark='benchmark/test_kaiser_window.py',scope='core',variant=0,descriptor=binding['shape'],native_api='torch.kaiser_window',inputs=dict(args=[],kwargs=dict(window_length=math.prod(binding['shape']),periodic=True,beta=12.0,dtype=str(dtype),device='target')),input_recipe='Exact original _build_inputs_fn; factory has no input tensor/RNG',output=dict(shape=[math.prod(binding['shape'])],dtype=str(dtype),device='target',alias=False,mutation=False))
