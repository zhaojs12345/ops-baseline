# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native NVIDIA _lazy_clone for the unchanged Kunlunxin FlagGems core catalog.

The target native API is unavailable. These six storage-layout plans match all
60 target core case IDs. Allocate the original uninitialized payload; neither
input nor returned output values are inspected by this latency-only recipe.
"""

import torch

OP_NAME = "_lazy_clone"
IS_INPLACE = False
DTYPES = [
    torch.float16,
    torch.float32,
    torch.bfloat16,
    torch.complex64,
    torch.int16,
    torch.int32,
    torch.int8,
    torch.uint8,
    torch.int64,
    torch.bool,
]
_SHAPES = ((64, 64), (4096, 4096), (64, 512, 512))
_LAYOUTS = ("contiguous", "transposed")
_SOURCE_COMMIT = "5313098abb729e22d406072abf795862ed8adaea"
_BENCHMARK_SHA256 = "23459d2b222f59eae9dc2319d638e855f10b10f313395b5130b2d8384b6b78b6"
_TARGET_CORE_YAML_SHA256 = (
    "1cf11e3ee0fa2c7b5ff2762f5c5d5c252f6c0248294b7d1d0db141bfc7304d07"
)
_TARGET_CATALOG_SHA256 = (
    "a09de0813dbbce2d84f9489a80c37abbebea85c4a2871f66437158d25237ac6e"
)


def native():
    return torch.ops.aten._lazy_clone


def grid():
    return [
        {"shape": list(shape), "layout": layout}
        for shape in _SHAPES
        for layout in _LAYOUTS
    ]


def _check(binding, dtype=None):
    shape, layout = tuple(binding["shape"]), binding["layout"]
    if shape not in _SHAPES or layout not in _LAYOUTS:
        raise ValueError(f"Unknown original target core case: {binding!r}")
    if dtype is not None and dtype not in DTYPES:
        raise ValueError(f"Unsupported original target benchmark dtype: {dtype}")
    return shape, layout


def key_shape(binding):
    shape, layout = _check(binding)
    return layout + "__" + "x".join(map(str, shape))


def build_inputs(binding, dtype, device):
    shape, layout = _check(binding, dtype)
    inp = torch.empty(shape, dtype=dtype, device=device)
    if layout == "transposed":
        inp = inp.transpose(-1, -2)
    return (inp,), {}


def config(binding, dtype):
    shape, layout = _check(binding, dtype)
    input_shape = list(shape)
    strides = [1] * len(shape)
    for axis in range(len(shape) - 2, -1, -1):
        strides[axis] = strides[axis + 1] * shape[axis + 1]
    if layout == "transposed":
        input_shape[-1], input_shape[-2] = input_shape[-2], input_shape[-1]
        strides[-1], strides[-2] = strides[-2], strides[-1]
    ordinal = _SHAPES.index(shape) * len(_LAYOUTS) + _LAYOUTS.index(layout)
    return {
        "operator": OP_NAME,
        "storage_shape": list(shape),
        "input_shape": input_shape,
        "input_stride": strides,
        "input_storage_offset": 0,
        "input_layout": layout,
        "dtype": str(dtype),
        "input_device_type": "cuda",
        "input_requires_grad": False,
        "input_is_conj": False,
        "input_is_neg": False,
        "input_recipe": "torch.empty(storage_shape, dtype=dtype, device=device)",
        "input_transform": "transpose(-1, -2)" if layout == "transposed" else None,
        "input_values": "uninitialized; no output value inspection",
        "case_id": (
            "benchmark/test__lazy_clone.py::test__lazy_clone::core::"
            f"{str(dtype).removeprefix('torch.')}::{ordinal}"
        ),
        "flaggems_source_commit": _SOURCE_COMMIT,
        "flaggems_benchmark_sha256": _BENCHMARK_SHA256,
        "target_core_yaml_sha256": _TARGET_CORE_YAML_SHA256,
        "target_catalog_sha256": _TARGET_CATALOG_SHA256,
    }
