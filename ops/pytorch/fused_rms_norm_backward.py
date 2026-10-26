# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU PyTorch-operator composition, NOT a fused vendor RMS backward primitive.

Case recipes: FlagGems kernelgen-dev d20405a359ab31f2c5d856c760850486d858e341,
benchmark/test_fused_rms_norm.py and benchmark/test_fused_rms_norm_backward.py.
The two original core grids contain eight distinct shapes and three dtypes.
No Triton/Gems kernel or CPU transfer is included in the measured callable.
"""

import math

import torch

OP_NAME = "fused_rms_norm_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
SOURCE_COMMIT = "d20405a359ab31f2c5d856c760850486d858e341"
RSTD_RECIPE = "rsqrt(mean(input.float()**2,dim=-1)+1e-5)"
REFERENCE_IMPLEMENTATION = "pytorch_native_operator_composition"
SHAPES = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 4096),
    (1024, 65536),
    (4, 64),
    (8, 256),
    (16, 1024),
]


def rms_norm_backward(grad_out, inp, normalized_shape, rstd, weight, output_mask):
    batch_ndim = inp.ndim - len(normalized_shape)
    norm_dims = tuple(range(batch_ndim, inp.ndim))
    value = inp.to(torch.float32)
    inv_rms = rstd.to(torch.float32).reshape(
        inp.shape[:batch_ndim] + (1,) * len(normalized_shape)
    )
    normalized = value * inv_rms
    grad_input = None
    if output_mask[0]:
        grad = grad_out.to(torch.float32)
        if weight is not None:
            grad = grad * weight.to(torch.float32)
        row_sum = (normalized * grad).sum(dim=norm_dims, keepdim=True)
        grad_input = (
            (grad - normalized / math.prod(normalized_shape) * row_sum) * inv_rms
        ).to(inp.dtype)
    grad_weight = None
    if output_mask[1] and weight is not None:
        grad_weight = (
            (grad_out.to(torch.float32) * normalized)
            .sum(dim=tuple(range(batch_ndim)))
            .reshape(normalized_shape)
            .to(inp.dtype)
        )
    return grad_input, grad_weight


def native():
    return rms_norm_backward


def grid():
    return [
        {
            "shape": list(shape),
            "benchmark": (
                "benchmark/test_fused_rms_norm.py"
                if index < 5
                else "benchmark/test_fused_rms_norm_backward.py"
            ),
        }
        for index, shape in enumerate(SHAPES)
    ]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    m, n = binding["shape"]
    inp = torch.randn((m, n), dtype=dtype, device=device)
    grad_out = torch.randn_like(inp)
    weight = torch.randn(n, dtype=dtype, device=device)
    rstd = torch.rsqrt(inp.float().pow(2).mean(dim=-1) + 1e-5)
    return (grad_out, inp, [n], rstd, weight, [True, True]), {}


def config(binding, dtype):
    m, n = binding["shape"]

    def tensor(shape, tensor_dtype, stride):
        return dict(
            shape=shape, dtype=str(tensor_dtype), stride=stride, requires_grad=False
        )

    return {
        "inputs": {
            "args": [
                tensor([m, n], dtype, [n, 1]),
                tensor([m, n], dtype, [n, 1]),
                [n],
                tensor([m], torch.float32, [1]),
                tensor([n], dtype, [1]),
                [True, True],
            ],
            "kwargs": {},
        },
        "source_commit": SOURCE_COMMIT,
        "benchmark": binding["benchmark"],
        "descriptor": [m, n],
        "scope": "core",
        "variant": 0,
        "seed": 0,
        "rstd_recipe": RSTD_RECIPE,
        "reference_implementation": REFERENCE_IMPLEMENTATION,
    }
