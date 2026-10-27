# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native cuDNN backward for the supported original current-core joints.

The original grid has six joints. BF16 at both shapes is explicitly unsupported
by this native API on H800; no replacement operator or zero latency is emitted.
"""
import torch
OP_NAME="cudnn_batch_norm_backward_current_core"
DTYPES=[torch.float16,torch.float32]
IS_INPLACE=False
SHAPES=[(16,16,64),(32,64,56,56)]
SOURCE_COMMIT="882edec4093e02573b7bb274c76cbf9352e96ba8"

def native():
    return torch.ops.aten.cudnn_batch_norm_backward.default

def grid():
    return [{"shape":list(s)}for s in SHAPES]

def key_shape(binding):
    return list(binding["shape"])

def build_inputs(binding,dtype,device):
    shape=tuple(binding["shape"])
    torch.manual_seed(0)
    channels=shape[1]
    inp=torch.randn(shape,dtype=dtype,device=device)
    grad=torch.randn(shape,dtype=dtype,device=device)
    weight=torch.ones(channels,dtype=torch.float32,device=device)
    save_mean=torch.randn(channels,dtype=torch.float32,device=device)
    save_invstd=torch.rand(channels,dtype=torch.float32,device=device)+0.1
    reserve=torch.empty((0,),dtype=torch.uint8,device=device)
    return (inp,grad,weight,None,None,save_mean,save_invstd,1e-5,reserve),{}

def config(binding,dtype):
    args,_=build_inputs(binding,dtype,"meta")
    def describe(x):
        return dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),requires_grad=x.requires_grad)if isinstance(x,torch.Tensor)else x
    return dict(inputs=dict(args=[describe(x)for x in args],kwargs={}),source_commit=SOURCE_COMMIT,
                benchmark="benchmark/test_cudnn_batch_norm_backward.py",scope="core",descriptor=list(binding["shape"]),variant=0,
                input_recipe=dict(seed=0,input="torch.randn",grad="torch.randn",weight="ones_fp32",save_mean="randn_fp32",save_invstd="rand_fp32_plus_0.1",reserve="empty_uint8",eps=1e-5))
