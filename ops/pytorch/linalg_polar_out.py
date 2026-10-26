# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""GPU PyTorch SVD polar composition writing the two supplied out buffers.

Recipes: FlagGems cae5c46393b02862c696499e25258518e0cc0e54,
benchmark/test_linalg_polar_out.py. Both output writes are included in timing;
input and output allocation is outside timing. This is not a fused primitive.
"""
import torch

from linalg_polar import (
    SOURCE_COMMIT, REFERENCE_SOURCE_SHA256, SHAPES, _config, linalg_polar,
)

OP_NAME = "linalg_polar_out"
DTYPES = [torch.float32]
IS_INPLACE = True


def linalg_polar_out(inp, *, U, H):
    result_u, result_h = linalg_polar(inp)
    U.copy_(result_u)
    H.copy_(result_h)
    return U, H


def native():
    return linalg_polar_out


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    shape = tuple(binding["shape"])
    inp = torch.randn(shape, dtype=dtype, device=device)
    out_u = torch.empty(shape, dtype=dtype, device=device)
    out_h = torch.empty((*shape[:-2], shape[-1], shape[-1]), dtype=dtype, device=device)
    return (inp,), dict(U=out_u, H=out_h)


def config(binding, dtype):
    return _config(binding, dtype, out=True)
