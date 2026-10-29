# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native torch.linalg.svdvals matching the current FlagGems original core input contract."""

import torch

OP_NAME = "linalg_svdvals_current_core"
DTYPES = [torch.float32]
IS_INPLACE = False
SOURCE_COMMIT = "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
CASES = [[16, 16], [32, 32], [64, 64], [128, 128], [256, 256], [512, 512]]


def native():
    return torch.linalg.svdvals


def grid():
    return [{"shape": list(shape)} for shape in CASES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    torch.manual_seed(0)
    matrix = torch.randn(shape, dtype=dtype, device=device)
    return (matrix,), {}


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")
    matrix = args[0]
    inputs = [
        {
            "shape": list(matrix.shape),
            "dtype": str(matrix.dtype),
            "stride": list(matrix.stride()),
            "requires_grad": matrix.requires_grad,
        }
    ]
    return {
        "inputs": {"args": inputs, "kwargs": kwargs},
        "source_commit": SOURCE_COMMIT,
        "benchmark": "benchmark/test_linalg_svdvals.py",
        "scope": "core",
        "descriptor": list(binding["shape"]),
        "variant": 0,
        "input_recipe": {"seed": 0, "input": "torch.randn"},
    }
