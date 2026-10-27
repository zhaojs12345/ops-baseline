# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact original Suiyuan core grid, timed through native full ATen Flash API.

FlagGems ac9a1f703757537a3d54cc1ac1febeca410316c0, both canonical Flash
benchmark files. Four original Enflame generic shapes and five explicit cases,
with both float16 and bfloat16, are retained without filtering or shape caps.
"""
import math
import torch

OP_NAME = "_scaled_dot_product_flash_attention"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
GENERIC = [(4, 32, 1024, 128), (4, 32, 2048, 128),
           (4, 32, 4096, 128), (4, 32, 8192, 128)]
EXPLICIT = [(1, 2, 128, 128, 64, False, 0.0, False, "default"),
            (2, 4, 256, 256, 64, True, 0.0, False, "explicit"),
            (1, 8, 128, 512, 128, False, 0.0, False, "explicit"),
            (1, 8, 1, 1024, 128, False, 0.0, False, "default"),
            (1, 2, 128, 128, 64, False, 0.2, True, "explicit")]


def native():
    return torch.ops.aten._scaled_dot_product_flash_attention.default


def grid():
    return [dict(family=family, ordinal=i, descriptor=list(shape))
            for family, shapes in [("generic", GENERIC), ("explicit", EXPLICIT)]
            for i, shape in enumerate(shapes)]


def key_shape(binding):
    return binding["family"] + "_" + str(binding["ordinal"]) + "_" + "x".join(
        map(str, binding["descriptor"][:5]))


def build_inputs(binding, dtype, device):
    shape = tuple(binding["descriptor"])
    if binding["family"] == "generic":
        query, key, value = (torch.randn(shape, device=device, dtype=dtype) for _ in range(3))
        return (query, key, value, 0.0, False, False), {}
    batch, heads, qlen, klen, dim, causal, dropout, debug, mode = shape
    torch.manual_seed(0)
    query = torch.empty((batch, heads, qlen, dim), dtype=dtype, device=device).uniform_(-0.05, 0.05)
    key = torch.empty((batch, heads, klen, dim), dtype=dtype, device=device).uniform_(-0.05, 0.05)
    value = torch.empty_like(key).uniform_(-0.05, 0.05)
    scale = None if mode == "default" else 1.0 / math.sqrt(dim)
    return (query, key, value, dropout, causal, debug), {"scale": scale}


def _describe(shape, dtype):
    tensor = torch.empty(shape, dtype=dtype, device="meta")
    return dict(shape=list(shape), dtype=str(dtype), stride=list(tensor.stride()), requires_grad=False)


def config(binding, dtype):
    shape = tuple(binding["descriptor"])
    family = binding["family"]
    if family == "generic":
        qshape = kshape = shape
        args = [_describe(qshape, dtype), _describe(kshape, dtype), _describe(kshape, dtype), 0.0, False, False]
        kwargs = {}
        filename = "benchmark/test_scaled_dot_product_attention.py"
        recipe = dict(distribution="torch.randn", seed="caller-owned RNG state, no source reseed")
    else:
        batch, heads, qlen, klen, dim, causal, dropout, debug, mode = shape
        qshape = (batch, heads, qlen, dim)
        kshape = (batch, heads, klen, dim)
        args = [_describe(qshape, dtype), _describe(kshape, dtype), _describe(kshape, dtype), dropout, causal, debug]
        kwargs = dict(scale=None if mode == "default" else 1.0 / math.sqrt(dim))
        filename = "benchmark/test_scaled_dot_product_flash_attention.py"
        recipe = dict(distribution="torch.empty.uniform_", low=-0.05, high=0.05, seed=0)
    return dict(inputs=dict(args=args, kwargs=kwargs), source_commit="ac9a1f703757537a3d54cc1ac1febeca410316c0",
                source_benchmark=filename, source_vendor_core="enflame", family=family,
                input_recipe=recipe, case_id=filename + "::test_scaled_dot_product_flash_attention::core::" + str(dtype).split(".")[-1] + "::" + str(binding["ordinal"]))
