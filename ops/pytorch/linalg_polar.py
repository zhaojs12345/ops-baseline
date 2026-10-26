# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Right matrix polar decomposition using native GPU PyTorch SVD operations.

This is an explicitly labeled composition, not torch.polar (complex-number
construction) and not a native fused linalg_polar primitive. Core input recipes
come from FlagGems cae5c46393b02862c696499e25258518e0cc0e54,
benchmark/test_linalg_polar.py. No CPU work or tensor transfer is timed.
"""
import math

import torch

OP_NAME = "linalg_polar"
DTYPES = [torch.float32]
IS_INPLACE = False
SOURCE_COMMIT = "cae5c46393b02862c696499e25258518e0cc0e54"
REFERENCE_SOURCE_SHA256 = "c773a1784bd719ba56203cd9b88b3ff78262fcfeccbb5a1b0f0d62f351850910"
SHAPES = [(4, 2), (8, 4), (16, 8), (16, 8, 4)]


def linalg_polar(inp):
    svd_u, singular_values, vh = torch.linalg.svd(inp, full_matrices=False)
    polar_u = svd_u @ vh
    polar_h = vh.mH @ (singular_values.unsqueeze(-1) * vh)
    return polar_u, 0.5 * (polar_h + polar_h.mH)


def native():
    return linalg_polar


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    return (torch.randn(binding["shape"], dtype=dtype, device=device),), {}


def _tensor_descriptor(shape, dtype):
    shape = list(shape)
    return dict(shape=shape, dtype=str(dtype),
                stride=[math.prod(shape[i + 1:]) for i in range(len(shape))],
                requires_grad=False)


def _config(binding, dtype, out=False):
    shape = list(binding["shape"])
    kwargs = {}
    if out:
        kwargs = {"U": _tensor_descriptor(shape, dtype),
                  "H": _tensor_descriptor(shape[:-2] + [shape[-1], shape[-1]], dtype)}
    return dict(
        inputs=dict(args=[_tensor_descriptor(shape, dtype)], kwargs=kwargs),
        reference_implementation="pytorch_native_svd_polar_composition",
        full_matrices=False, side="right", symmetrize_H=True,
        input_recipe="torch.manual_seed(0);torch.randn(shape,dtype=torch.float32,device=device)",
        out_aliases=["U", "H"] if out else [],
    )


def config(binding, dtype):
    return _config(binding, dtype)
