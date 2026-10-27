# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native recipe from FlagGems 9d57a23b4386bd669152a935fe88a96db907da7c.

Default inputs faithfully retain its forward -> ones_like gradient recipe.
The explicit current-core profile follows the later 2ff randn-gradient builder.
Both use the source's ambient random generator without resetting its state.
"""

import os

import torch

OP_NAME = "_adaptive_avg_pool3d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
LEGACY_SOURCE = "9d57a23b4386bd669152a935fe88a96db907da7c"
CURRENT_SOURCE = "2ff0931a646b97ec98ebd50642b5889619125a6b"
CURRENT_BUILDER_SHA256 = (
    "6658ad2c7da71d182bb2a0137ba1de5e0880b12b003e9a8aa5e3afa2a2ace6c1"
)
_SELECTOR = "OPS_BASELINE_ADAPTIVE_AVG_POOL3D_CURRENT_CORE"
_profile = os.environ.get(_SELECTOR)
if _profile not in (None, "0", "1"):
    raise ValueError(f"{_SELECTOR} must be 0 or 1")
CASES = [
    ((1, 3, 8, 8, 8), (4, 4, 4)),
    ((2, 3, 16, 16, 16), (8, 8, 8)),
    ((1, 1, 32, 32, 32), (16, 16, 16)),
    ((4, 8, 64, 64, 64), (32, 32, 32)),
]


def native():
    return torch.ops.aten._adaptive_avg_pool3d_backward.default


def grid():
    return [{"shape": list(shape), "variant": 0} for shape, _ in CASES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    if dtype not in DTYPES or binding.get("variant", 0) != 0:
        raise ValueError("Only the original three dtypes and single variant are valid")
    matches = [output for candidate, output in CASES if candidate == shape]
    if len(matches) != 1:
        raise ValueError("Unrecognized original adaptive pool3d backward descriptor")
    inp = torch.randn(shape, dtype=dtype, device=device)
    if _profile == "1":
        grad = torch.randn((*shape[:-3], *matches[0]), dtype=dtype, device=device)
    else:
        output = torch.nn.functional.adaptive_avg_pool3d(inp, matches[0])
        grad = torch.ones_like(output)
    return (grad, inp), {}


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")
    result = {
        "inputs": {
            "args": [
                {
                    "shape": list(value.shape),
                    "dtype": str(value.dtype),
                    "stride": list(value.stride()),
                    "requires_grad": value.requires_grad,
                }
                for value in args
            ],
            "kwargs": kwargs,
        },
        "source_commit": CURRENT_SOURCE if _profile == "1" else LEGACY_SOURCE,
        "benchmark": "benchmark/test__adaptive_avg_pool3d_backward.py",
        "descriptor": list(binding["shape"]),
        "scope": "core",
        "variant": 0,
    }
    if _profile == "1":
        result.update(
            native_schema="aten::_adaptive_avg_pool3d_backward.default",
            grid_contract="_adaptive_avg_pool3d_backward.kernelgen-dev.core/v1",
            input_contract="adaptive-avg-pool3d-backward-randn-gradient/v1",
            input_builder_sha256=CURRENT_BUILDER_SHA256,
            producer_selector={_SELECTOR: "1"},
            random_policy="source_unseeded_global_generator_no_reset",
            input_recipe="randn(self); randn(grad_output); no forward or seed reset",
        )
    return result
