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


"""_aminmax: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_aminmax.py:91 (test__aminmax)
Native expression: torch._aminmax
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryReductionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import os as _os

import torch

from _kernelgen import generate_tensor_input, export_contract


def _aminmax_input_fn(shape, cur_dtype, device):
    inp = generate_tensor_input(shape, cur_dtype, device)
    yield (inp,)


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        yield from _aminmax_input_fn(shape, dtype, device)


def _resolve_native():
    return torch._aminmax


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(export_contract("_aminmax", DTYPES, _resolve_native, _inputs))


# Explicit profiles for the two current original FlagGems benchmark assets.
# The default export above is retained verbatim, including legacy metadata.

_SELECT = "OPS_BASELINE_AMINMAX_CORE_PROFILE"
_selected_profile = _os.environ.get(_SELECT)
if _selected_profile not in (None, "legacy", "shared", "direct"):
    raise ValueError(f"{_SELECT} must be legacy, shared or direct")
PROFILE = _selected_profile or "legacy"
_CURRENT_SOURCE = "2ff0931a646b97ec98ebd50642b5889619125a6b"
_PROFILE_SHAPES = {
    "shared": (
        (1048576,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 512),
    ),
    "direct": (
        (1073741824,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 1024),
    ),
}
_SOURCE_BINDINGS = {
    "direct": {
        "_case_fn": "bd835636353cb2703100478d9d32de51b0983fabe987d999284db5f34dbe6657",
        "_build_inputs_fn": "bd6712aeee3823008708ab3f0bf363021a0d7a84596a899ab4511190c666fbc5",
        "generator": "24e59f7572e9db79adb13460155847b72069bbd315db3093b2a10b4b1b9779d3",
    },
    "shared": {
        "get_case_iter": "8e7fcff16f63bbfaf19961471d4610ea25cb2c135a498e30684a49482a6cee1a",
        "build_inputs": "3edecbb308be85061151717090daf9f026ac51444853bd134558546bd678771f",
        "generator": "24e59f7572e9db79adb13460155847b72069bbd315db3093b2a10b4b1b9779d3",
    },
}

if PROFILE != "legacy":

    def _current_shape(binding):
        if set(binding) != {"shape", "variant"}:
            raise ValueError("Unexpected _aminmax profile binding fields")
        if type(binding["variant"]) is not int or binding["variant"] != 0:
            raise ValueError("Only the original single core variant is supported")
        shape = tuple(binding["shape"])
        if shape not in _PROFILE_SHAPES[PROFILE]:
            raise ValueError("Unrecognized original _aminmax core descriptor")
        return shape

    def native():
        # Preserve the original private native API and tuple-return contract.
        return torch._aminmax

    def grid():
        return [
            {"shape": list(shape), "variant": 0} for shape in _PROFILE_SHAPES[PROFILE]
        ]

    def key_shape(binding):
        return list(_current_shape(binding))

    def build_inputs(binding, dtype, device):
        shape = _current_shape(binding)
        if dtype not in DTYPES:
            raise ValueError("Only the original three floating dtypes are supported")
        if PROFILE == "direct":
            torch.manual_seed(0)
        return (generate_tensor_input(shape, dtype, device),), {}

    def config(binding, dtype):
        shape = _current_shape(binding)
        if dtype not in DTYPES:
            raise ValueError("Only the original three floating dtypes are supported")
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        inputs = {
            "args": [
                {
                    "shape": list(shape),
                    "dtype": str(dtype),
                    "stride": stride,
                    "requires_grad": False,
                }
            ],
            "kwargs": {},
        }
        scalar = {
            "shape": [],
            "dtype": str(dtype),
            "stride": [],
            "requires_grad": False,
        }
        return {
            "inputs": inputs,
            "native_timer_contract": {
                "kind": "native_private_all_reduce",
                "native_schema": "aten::_aminmax.default",
                "argument_order": ["self"],
                "kwargs": {},
                "returns": {
                    "container": "tuple",
                    "items": [dict(scalar), dict(scalar)],
                },
                "setup_timed": False,
            },
            "random_policy": (
                "source_manual_seed0_then_randn_self"
                if PROFILE == "direct"
                else "source_unseeded_global_generator_no_reset"
            ),
            "input_recipe": (
                "manual_seed(0); generate(self)"
                if PROFILE == "direct"
                else "generate(self)"
            ),
            "benchmark": (
                "benchmark/test__aminmax.py"
                if PROFILE == "direct"
                else "benchmark/test_aminmax.py"
            ),
            "source_commit": _CURRENT_SOURCE,
            "source_test": "test__aminmax",
            "descriptor": list(shape),
            "scope": "core",
            "variant": 0,
            "grid_contract": f"_aminmax.{PROFILE}.core/v1",
            "producer_selector": {_SELECT: PROFILE},
            "source_bindings": dict(_SOURCE_BINDINGS[PROFILE]),
        }
