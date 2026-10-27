# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


"""ldexp_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_ldexp.py:63 (test_ldexp_out)
Native expression: torch.ops.aten.ldexp.out
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def ldexp_out_input_fn(shape, dtype, device):
    self = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    out = torch.empty(shape, dtype=dtype, device=device)
    yield (self, other, {"out": out})


def _resolve_native():
    return torch.ops.aten.ldexp.out


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = ldexp_out_input_fn

globals().update(export_contract("ldexp_out", DTYPES, _resolve_native, _inputs))

# Current FlagGems core recipes at ac9a1f703757537a3d54cc1ac1febeca410316c0.
# Keep every legacy binding above and append only previously absent shapes.
_CURRENT_CORE_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
_EXTRA_SHAPES = [(1073741824,), (64, 64), (64, 512, 512), (1024, 1024, 1024)]
_LEGACY = {
    key: globals()[key] for key in ("grid", "build_inputs", "key_shape", "config")
}


def grid():
    return _LEGACY["grid"]() + [
        {"shape": list(shape), "variant": 0} for shape in _EXTRA_SHAPES
    ]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    if shape not in _EXTRA_SHAPES:
        return _LEGACY["build_inputs"](binding, dtype, device)
    # Match the current FlagGems builder's explicit seed and value recipes.
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
    kwargs = (
        {"out": torch.empty(shape, dtype=dtype, device=device)}
        if OP_NAME == "ldexp_out"
        else {}
    )
    return (inp, exponent), kwargs


def config(binding, dtype):
    shape = tuple(binding["shape"])
    if shape not in _EXTRA_SHAPES:
        return _LEGACY["config"](binding, dtype)

    def describe(dtype):
        stride = []
        size = 1
        for dim in reversed(shape):
            stride.append(size)
            size *= dim
        return {
            "shape": list(shape),
            "dtype": str(dtype),
            "stride": list(reversed(stride)),
            "requires_grad": False,
        }

    kwargs = {"out": describe(dtype)} if OP_NAME == "ldexp_out" else {}
    return {
        "inputs": {"args": [describe(dtype), describe(torch.int32)], "kwargs": kwargs},
        "source_commit": _CURRENT_CORE_SOURCE,
        "benchmark": "benchmark/test_ldexp.py",
        "descriptor": list(shape),
        "scope": "core",
        "variant": 0,
        "input_recipe": {
            "seed": 0,
            "input": "torch.randn",
            "exponent": "torch.randint(-8,9,int32)",
        },
    }
