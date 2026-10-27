# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native depthwise convolution for FlagGems ac9a1f7 current core recipes.

The historical conv_depthwise2d module and snapshot remain unchanged. This
versioned recipe matches benchmark/test_conv_depthwise2d.py at the source below.
"""
import torch

OP_NAME = "conv_depthwise2d_current_core"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
SOURCE_COMMIT = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
CASES = [(32, 4, 8, 8, 2, 2, 1, 0, 1), (18, 16, 4, 4, 2, 2, 2, 1, 1),
         (1, 64, 224, 224, 3, 3, 1, 1, 1), (1, 128, 112, 112, 5, 5, 2, 2, 1)]


def native():
    return torch.ops.aten._conv_depthwise2d.default


def grid():
    return [{"shape": list(shape)} for shape in CASES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    n, c, h, w, kh, kw, stride, padding, dilation = binding["shape"]
    torch.manual_seed(0)
    inp = torch.randn((n, c, h, w), dtype=dtype, device=device)
    weight = torch.randn((c, 1, kh, kw), dtype=dtype, device=device)
    return (inp, weight, [kh, kw], None, [stride, stride], [padding, padding],
            [dilation, dilation]), {}


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")
    def describe(value):
        if isinstance(value, torch.Tensor):
            return {"shape": list(value.shape), "dtype": str(value.dtype),
                    "stride": list(value.stride()), "requires_grad": value.requires_grad}
        return value
    return {"inputs": {"args": [describe(x) for x in args], "kwargs": kwargs},
            "source_commit": SOURCE_COMMIT,
            "benchmark": "benchmark/test_conv_depthwise2d.py", "scope": "core",
            "descriptor": list(binding["shape"]), "variant": 0,
            "input_recipe": {"seed": 0, "input": "torch.randn", "weight": "torch.randn",
                             "bias": None}}
