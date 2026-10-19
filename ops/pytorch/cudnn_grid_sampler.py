# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""cudnn_grid_sampler: native CUDA recipe from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_grid_sampler.py (test_cudnn_grid_sampler).
Preserves all six core input/grid pairs, NVIDIA dtypes and random input ranges.
"""

import torch

from _kernelgen import export_contract

DTYPES = [torch.float32, torch.float16, torch.float64]


def _inputs(shape, dtype, device):
    input_shape, grid_shape = shape
    yield (
        torch.randn(input_shape, dtype=dtype, device=device),
        torch.rand(grid_shape, dtype=dtype, device=device) * 2 - 1,
    )


globals().update(export_contract(
    "cudnn_grid_sampler", DTYPES, lambda: torch.ops.aten.cudnn_grid_sampler, _inputs
))
