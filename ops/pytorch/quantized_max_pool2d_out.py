# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native pooling out overload, timing writes to the original preallocated ABI."""
import torch

from quantized_max_pool2d import (
    DTYPES,
    _quantized_descriptor,
    build_inputs as _functional_inputs,
    config as _functional_config,
    grid,
    key_shape,
)

OP_NAME = "quantized_max_pool2d_out"
IS_INPLACE = True


def native():
    return torch.ops.aten.quantized_max_pool2d.out


def _output_shape(shape):
    # Original core: kernel3, stride2, padding1, dilation1, ceil_mode=False.
    return [*shape[:2], (shape[-2] + 1) // 2, (shape[-1] + 1) // 2]


def build_inputs(binding, dtype, device):
    args, kwargs = _functional_inputs(binding, dtype, device)
    kwargs["out"] = torch._empty_affine_quantized(
        _output_shape(binding["shape"]), scale=0.1, zero_point=0, dtype=dtype, device=device)
    return args, kwargs


def config(binding, dtype):
    result = _functional_config(binding, dtype)
    result["inputs"]["kwargs"]["out"] = _quantized_descriptor(_output_shape(binding["shape"]), dtype)
    return result
