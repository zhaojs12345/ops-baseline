# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""float_power_tensor_tensor_out: native real-input FloatPower out, original core recipe.
Pinned FlagGems source: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_float_power.py; explicit 5-shape/3-dtype geometry.
The named-constructor parser missed _run_benchmark; this operator-only manual
catalog records the genuine original recipe. Ambient RNG remains unseeded.
"""
import torch
from _kernelgen import export_contract

def _input_iter(shape, dtype, device):
    base_tensor = torch.rand(shape, dtype=dtype, device=device).add_(0.25)
    exponent = torch.rand(shape, dtype=dtype, device=device).mul_(4).sub_(2)
    out = torch.empty(shape, dtype=torch.float64, device=device)
    yield base_tensor, exponent, {"out": out}


def _resolve_native():
    return torch.ops.aten.float_power.Tensor_Tensor_out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]
globals().update(export_contract("float_power_tensor_tensor_out", DTYPES, _resolve_native, _input_iter))
