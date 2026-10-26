# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native baddbmm_ with original beta=1 and fresh inputs outside event timing."""
import torch

import _inspect216_inplace as recipes

OP_NAME = "baddbmm_"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
measure_latency = recipes.measure_latency


def native():
    return torch.Tensor.baddbmm_


def grid():
    return recipes.grid(OP_NAME)


def key_shape(binding):
    return recipes.key_shape(OP_NAME, binding)


def build_inputs(binding, dtype, device):
    return recipes.build_inputs(OP_NAME, binding, dtype, device)


def config(binding, dtype):
    return recipes.config(OP_NAME, binding, dtype)
