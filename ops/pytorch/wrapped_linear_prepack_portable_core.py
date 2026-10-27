# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original FlagGems portable UInt8 packing reference on CUDA.

This measures the original multi-PyTorch-operation packing composition, NOT
aten::_wrapped_linear_prepack's opaque format or native opaque prepack latency.
"""
import torch
OP_NAME = "wrapped_linear_prepack_portable_core"
DTYPES = [torch.float32]
IS_INPLACE = False
SHAPES = [(4,8),(64,128),(256,256),(512,1024),(2048,1024)]
_PACK_MAGIC = 260821.0
_PACK_VERSION = 1.0

def _torch_reference(weight, weight_scale, weight_zero_point, bias):
    quantized = torch.clamp(
        torch.round(weight / weight_scale) + weight_zero_point, -128, 127
    ).to(torch.int8)
    metadata = torch.stack(
        (
            weight_scale.to(torch.float32),
            weight_zero_point.to(torch.float32),
            weight_scale.new_tensor(_PACK_MAGIC),
            weight_scale.new_tensor(_PACK_VERSION),
        )
    )
    return torch.cat(
        (
            metadata.contiguous().view(torch.uint8),
            bias.to(torch.float32).contiguous().view(torch.uint8),
            quantized.contiguous().view(torch.uint8).flatten(),
        )
    )

def native():
    return _torch_reference

def grid():
    return [{"shape": list(s)} for s in SHAPES]

def key_shape(binding):
    return list(binding["shape"])

def build_inputs(binding, dtype, device):
    n,k=binding["shape"]
    torch.manual_seed(0)
    return (torch.randn((n,k),dtype=torch.float32,device=device),
            torch.tensor(0.03125,dtype=torch.float32,device=device),
            torch.tensor(-3,dtype=torch.int64,device=device),
            torch.randn(n,dtype=torch.float32,device=device)), {}

def config(binding, dtype):
    args,_=build_inputs(binding,dtype,"meta")
    return dict(inputs=dict(args=[dict(shape=list(x.shape),dtype=str(x.dtype),stride=list(x.stride()),requires_grad=x.requires_grad) for x in args],kwargs={}),
                source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
                benchmark="benchmark/test_wrapped_linear_prepack.py",scope="core",
                descriptor=list(binding["shape"]),variant=0,
                input_recipe=dict(seed=0,input="torch.randn",bias="torch.randn",scale=0.03125,zero_point=-3),
                reference_kind="portable_uint8_pytorch_cuda_composition")
