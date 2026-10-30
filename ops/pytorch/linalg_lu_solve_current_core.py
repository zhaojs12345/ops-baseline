# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native LU solve matching the FlagGems core input recipe and explicit flags."""
import torch

OP_NAME = "linalg_lu_solve_current_core"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
SOURCE_COMMIT = "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
CASES = [(4, 4), (16, 1), (33, 16), (64, 64), (128, 128)]


def native():
    return torch.ops.aten.linalg_lu_solve


def grid():
    return [{"shape": list(shape)} for shape in CASES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    n, k = binding["shape"]
    torch.manual_seed(0)
    matrix = torch.randn(n, n, dtype=dtype, device=device)
    matrix = matrix @ matrix.mT + torch.eye(n, dtype=dtype, device=device) * n
    lu, pivots = torch.linalg.lu_factor(matrix)
    rhs = torch.randn(n, k, dtype=dtype, device=device)
    return (lu, pivots, rhs), {"left": True, "adjoint": False}


def config(binding, dtype):
    n, k = binding["shape"]
    return {
        "inputs": {
            "args": [
                {"shape": [n, n], "dtype": str(dtype), "stride": [1, n], "requires_grad": False},
                {"shape": [n], "dtype": "torch.int32", "stride": [1], "requires_grad": False},
                {"shape": [n, k], "dtype": str(dtype), "stride": [k, 1], "requires_grad": False},
            ],
            "kwargs": {"left": True, "adjoint": False},
        },
        "source_commit": SOURCE_COMMIT,
        "benchmark": "benchmark/test_linalg_lu_solve.py",
        "scope": "core",
        "descriptor": [n, k],
        "variant": 0,
        "input_recipe": {
            "seed": 0,
            "matrix": "torch.randn(n,n); matrix@matrix.mT+n*eye(n)",
            "factorization": "torch.linalg.lu_factor(matrix)",
            "rhs": "torch.randn(n,k) after factorization",
        },
    }
