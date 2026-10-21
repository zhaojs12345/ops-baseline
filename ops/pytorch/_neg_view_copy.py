# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native functional/out _neg_view_copy, matching the original FlagGems core cases.

FlagGems kernelgen-dev source a431434f53bd037f65477a0bae700820becab687:
benchmark/test__neg_view_copy.py, both marker-selected benchmark functions.
The unary and unary-out builders use randn and an empty_like out buffer.
"""

import torch

from _kernelgen import generate_tensor_input

OP_NAME = "_neg_view_copy"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
_SHAPES = ((64, 64), (4096, 4096), (64, 512, 512))
_SOURCE_COMMIT = "a431434f53bd037f65477a0bae700820becab687"
_BENCHMARK_SHA256 = "91f876fe05df4595e1a35c6480af6d63ca24ee070cebe77325cd86ca4323356a"


def native():
    return torch.ops.aten._neg_view_copy


def grid():
    return [
        {"shape": list(shape), "variant": variant}
        for variant in ("functional", "out")
        for shape in _SHAPES
    ]


def _check(binding):
    shape = tuple(binding["shape"])
    variant = binding["variant"]
    if shape not in _SHAPES or variant not in ("functional", "out"):
        raise ValueError(f"Unknown original core case: {binding!r}")
    return shape, variant


def key_shape(binding):
    shape, variant = _check(binding)
    return variant + "__" + "x".join(map(str, shape))


def build_inputs(binding, dtype, device):
    shape, variant = _check(binding)
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported benchmark dtype: {dtype}")
    inp = generate_tensor_input(shape, dtype, device)
    kwargs = {"out": torch.empty_like(inp)} if variant == "out" else {}
    return (inp,), kwargs


def config(binding, dtype):
    shape, variant = _check(binding)
    if dtype not in DTYPES:
        raise ValueError(f"Unsupported benchmark dtype: {dtype}")
    function = "test__neg_view_copy" + ("_out" if variant == "out" else "")
    ordinal = _SHAPES.index(shape)
    return {
        "operator": OP_NAME,
        "variant": variant,
        "shape": list(shape),
        "dtype": str(dtype),
        "input_recipe": "torch.randn(shape, dtype=dtype, device=device)",
        "input_layout": "contiguous",
        "input_requires_grad": False,
        "out_recipe": "torch.empty_like(input)" if variant == "out" else None,
        "case_id": f"benchmark/test__neg_view_copy.py::{function}::core::{str(dtype).removeprefix('torch.')}::{ordinal}",
        "flaggems_source_commit": _SOURCE_COMMIT,
        "flaggems_benchmark_sha256": _BENCHMARK_SHA256,
    }
