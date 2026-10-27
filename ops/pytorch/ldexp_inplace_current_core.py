# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original ac9 ldexp_ core; collect only with the dedicated fresh-input driver."""
import torch
OP_NAME = 'ldexp_inplace_current_core'
IS_INPLACE = True
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
SHAPES = [(1073741824,), (64,64), (4096,4096), (64,512,512), (1024,1024,1024)]
SOURCE_COMMIT = 'ac9a1f703757537a3d54cc1ac1febeca410316c0'
def native():
    return torch.ops.aten.ldexp_.default
def grid():
    return [dict(shape=list(s),variant=0) for s in SHAPES]
def key_shape(binding):
    return binding['shape']
def build_inputs(binding,dtype,device):
    shape=tuple(binding['shape']);torch.manual_seed(0)
    return (torch.randn(shape,dtype=dtype,device=device),torch.randint(-8,9,shape,dtype=torch.int32,device=device)),{}
def config(binding,dtype):
    shape=binding['shape'];stride=[];v=1
    for n in reversed(shape):stride.insert(0,v);v*=n
    tensor=lambda d:dict(shape=shape,dtype=str(d),stride=stride,requires_grad=False)
    return dict(source_commit=SOURCE_COMMIT,benchmark='benchmark/test_ldexp_.py',scope='core',variant=0,descriptor=shape,native_api='torch.ops.aten.ldexp_.default',inputs=dict(args=[tensor(dtype),tensor(torch.int32)],kwargs={}),input_recipe=dict(rng='manual_seed(0)',self='randn',exponent='randint(-8,9,int32)',fresh_inputs=True),measurement_method='official collector + operator-specific untimed-refresh CUDA-event timing',mutation='self only',output_alias='self')
