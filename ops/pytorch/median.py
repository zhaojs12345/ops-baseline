# Copyright 2026, The FlagOS Contributors.
# SPDX-License-Identifier: Apache-2.0
"""Native no-dim median recipes from FlagGems 9d57a23b4386bd669152a935fe88a96db907da7c.

The default contract retains that source's nine unseeded shapes and five dtypes.
An explicit operator-local selector enables the separate, seed-0 Suiyuan core
grid. Neither dim nor out overloads are included in this module.
"""

import os

import torch

from _kernelgen import export_contract

DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.int16, torch.int32]


def _input(shape, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("Unsupported median input dtype")
    if dtype.is_floating_point:
        return torch.randn(shape, dtype=dtype, device=device)
    # The source creates integer values on CPU before moving to the target.
    # A meta-only enumeration must not allocate the source's large CPU arrays.
    host_device = "meta" if torch.device(device).type == "meta" else "cpu"
    return torch.randint(
        torch.iinfo(dtype).min,
        torch.iinfo(dtype).max,
        shape,
        dtype=dtype,
        device=host_device,
    ).to(device)


def _legacy_inputs(shape, dtype, device):
    yield (_input(shape, dtype, device),)


def _resolve_native():
    return torch.ops.aten.median.default


globals().update(export_contract("median", DTYPES, _resolve_native, _legacy_inputs))
_selector = os.environ.get("OPS_BASELINE_MEDIAN_SUIYUAN_CORE")
if _selector not in (None, "0", "1"):
    raise ValueError("OPS_BASELINE_MEDIAN_SUIYUAN_CORE must be 0 or 1")
if _selector == "1":
    SHAPES = [
        (1073741824,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 1024),
    ]

    def native():
        return torch.ops.aten.median.default

    def grid():
        return [{"shape": list(shape)} for shape in SHAPES]

    def key_shape(binding):
        return list(binding["shape"])

    def _validate(binding, dtype):
        shape = tuple(binding["shape"])
        if set(binding) != {"shape"} or dtype not in DTYPES or shape not in SHAPES:
            raise ValueError("Unknown original Suiyuan median core dtype/shape")
        return shape

    def build_inputs(binding, dtype, device):
        shape = _validate(binding, dtype)
        torch.manual_seed(0)
        return (_input(shape, dtype, device),), {}

    def config(binding, dtype):
        shape = _validate(binding, dtype)
        stride = []
        extent = 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        return {
            "inputs": {
                "args": [
                    {
                        "shape": list(shape),
                        "dtype": str(dtype),
                        "stride": stride,
                        "requires_grad": False,
                    }
                ],
                "kwargs": {},
            },
            "benchmark": "benchmark/test_median.py",
            "scope": "core",
            "input_contract": "median-no-dim-readonly-scalar/v1",
            "grid_contract": "median.suiyuan.kernelgen-dev.core/v1",
            "producer_selector": {"OPS_BASELINE_MEDIAN_SUIYUAN_CORE": "1"},
            "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
            "input_recipe": "manual_seed(0); floating randn(shape,dtype,device); integer randint full iinfo on CPU then target transfer; torch.median no dim",
            "input_builder_hash_method": "ast.get_source_segment(MedianBenchmark.build_inputs),ast.get_source_segment(generate_tensor_input)",
            "input_builder_sha256": "d95708621137c7551aefb533f2d6c5bf94b008b7b2e649a75db89a761abad920",
            "input_helper_sha256": "e991af3667ae315b99d8508a74c52528e478f5bc00ddd96eab06dad01b219495",
        }
