# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""sparse_mask: native CUDA recipe from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sparse_mask.py (test_sparse_mask).
Preserves four square shapes, FLOAT_DTYPES, normal dense input and boolean COO
mask distribution. A local CPU generator fixes the mask across latency/NCU
processes; config records its seed, actual nnz and COO-index SHA256. This is a
reproducible sample of the source distribution, not the source's random mask.
"""

import hashlib
import math

import torch

from _kernelgen import export_contract

DTYPES = [torch.float16, torch.float32, torch.bfloat16]


def _mask(shape):
    generator = torch.Generator(device="cpu").manual_seed(20261019 + shape[0])
    ratio = 0.1 if math.prod(shape) <= 1024 * 1024 else 0.02
    return (torch.rand(shape, generator=generator) > (1 - ratio)).to_sparse()


def _inputs(shape, dtype, device):
    mask = _mask(shape)
    # Reject RNG/version drift rather than attach stale mask identity to timings.
    expected = config({"shape": list(shape), "variant": 0}, dtype)["mask"]
    digest = hashlib.sha256(mask.indices().numpy().tobytes()).hexdigest()
    if mask._nnz() != expected["nnz"] or digest != expected["indices_sha256"]:
        raise ValueError("Sparse mask replay differs from the recorded case identity")
    yield torch.randn(shape, dtype=dtype, device=device), mask.to(device)


globals().update(export_contract(
    "sparse_mask", DTYPES, lambda: torch.ops.aten.sparse_mask, _inputs
))
