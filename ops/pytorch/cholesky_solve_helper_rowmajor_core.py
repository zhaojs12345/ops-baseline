# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native H800 Cholesky solve for the original Ascend core input layout.

The target's original Cholesky builder returns row-major factors. Materialize
that same layout on H800 before timing; never include factorization or layout
conversion in the timed solve. Seed0 is the original caller policy, not a claim
that CUDA and NPU produce identical random bytes.
"""

import hashlib

import torch

OP_NAME = "cholesky_solve_helper_rowmajor_core"
DTYPES = [torch.float32]
IS_INPLACE = False
BENCH_RETURN_MODE = "mean"
SHAPES = [(4, 1), (8, 2), (16, 4), (32, 4), (64, 8)]
_samples = {}


def native():
    return torch.ops.aten._cholesky_solve_helper


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    n, nrhs = binding["shape"]
    torch.manual_seed(0)
    matrix = torch.randn((n, n), dtype=dtype, device=device)
    matrix = matrix @ matrix.mT + n * torch.eye(n, dtype=dtype, device=device)
    factor = torch.linalg.cholesky(matrix).contiguous()
    rhs = torch.randn((n, nrhs), dtype=dtype, device=device)
    values = [
        value.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
        for value in (rhs, factor)
    ]
    _samples[((n, nrhs), str(dtype))] = {
        "input_sha256": [hashlib.sha256(value).hexdigest() for value in values],
        "input_bytes": [len(value) for value in values],
        "rng_policy": "original torch.manual_seed(0) for each case; no cross-device byte identity",
    }
    assert rhs.stride() == (nrhs, 1) and factor.stride() == (n, 1)
    return (rhs, factor, False), {}


def config(binding, dtype):
    n, nrhs = binding["shape"]
    return {
        "native_expression": "torch.ops.aten._cholesky_solve_helper",
        "inputs": {
            "args": [
                dict(
                    shape=[n, nrhs],
                    dtype=str(dtype),
                    stride=[nrhs, 1],
                    requires_grad=False,
                ),
                dict(
                    shape=[n, n], dtype=str(dtype), stride=[n, 1], requires_grad=False
                ),
                False,
            ],
            "kwargs": {},
        },
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "benchmark": "benchmark/test_cholesky_solve_helper.py",
        "scope": "core",
        "descriptor": [n, nrhs],
        "variant": 0,
        "input_recipe": {
            "matrix": "randn((n,n),dtype,device) @ matrix.mT + n*eye(n)",
            "factor": "linalg.cholesky outside timed call; row-major factor matches actual original Ascend layout",
            "rhs": "randn((n,nrhs),dtype,device)",
            "seed": "original torch.manual_seed(0) per case",
        },
        "producer_sample": _samples[((n, nrhs), str(dtype))],
    }
