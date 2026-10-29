# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Documented Qwen3.6 native SGLang v0.5.11 workload."""
import _qwen36
import torch

OP_NAME = "fused_recurrent_gated_delta_rule_packed_decode"
DTYPES = [torch.bfloat16]
BENCH_RETURN_MODE = "median"
IS_INPLACE = True


def native():
    return _qwen36.native(OP_NAME)


def grid():
    return _qwen36.grid(OP_NAME)


def build_inputs(binding, dtype, device):
    return _qwen36.inputs(OP_NAME, binding, dtype, device)


def key_shape(binding):
    return _qwen36.key_shape(OP_NAME, binding)


def config(binding, dtype):
    return _qwen36.config(OP_NAME, binding, dtype)
