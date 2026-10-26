# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native qint8 pooling with original NCHW core recipes and quantization."""
import math

import torch

OP_NAME = "quantized_max_pool2d"
DTYPES = [torch.qint8]
IS_INPLACE = False
SOURCE_COMMIT = "b9c4b24a83ecc38312d2e6c2779d314b405f69cf"
SOURCE_BENCHMARK = "benchmark/test_quantized_max_pool2d.py"
SHAPES = [(4, 3, 224, 224), (16, 64, 56, 56), (32, 128, 28, 28),
          (64, 256, 14, 14), (128, 512, 7, 7)]
PARAMS = dict(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)


def native():
    return torch.quantized_max_pool2d


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    if dtype != torch.qint8:
        raise ValueError("The original accelerator core catalog is qint8 only")
    raw = torch.randint(-128, 128, binding["shape"], device=device).float()
    inp = torch.quantize_per_tensor(raw, 0.1, 0, dtype)
    return (inp,), dict(PARAMS)


def _quantized_descriptor(shape, dtype):
    return dict(shape=list(shape), dtype=str(dtype),
                stride=[math.prod(shape[i + 1:]) for i in range(len(shape))],
                requires_grad=False, qscheme="torch.per_tensor_affine", scale=0.1, zero_point=0)


def config(binding, dtype):
    shape = binding["shape"]
    return dict(inputs=dict(args=[_quantized_descriptor(shape, dtype)], kwargs=dict(PARAMS)),
                source_commit=SOURCE_COMMIT, benchmark=SOURCE_BENCHMARK,
                descriptor=list(shape), scope="core", variant=0,
                reference_implementation="pytorch_native_operator")
