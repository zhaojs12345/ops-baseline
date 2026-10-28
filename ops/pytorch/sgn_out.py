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


"""sgn_out: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_sgn.py:44 (test_sgn_out)
Native expression: torch.sgn
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml Benchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import standard_inputs, export_contract


def _resolve_native():
    return torch.sgn


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: standard_inputs(
    "unary_out", shape, dtype, device
)

globals().update(export_contract("sgn_out", DTYPES, _resolve_native, _inputs))
# Current operator-local profiles are explicit opt-ins; legacy exports above stay intact.
import os as _os
from _kernelgen import generate_tensor_input as _current_input

_PROFILE_DATA = {
    "current-shared": {
        "source_file": "benchmark/test_sgn.py",
        "benchmark_node": "benchmark/test_sgn.py::test_sgn_out",
        "shapes": [
            [536870912],
            [64, 64],
            [4096, 4096],
            [64, 512, 512],
            [1024, 1024, 512],
        ],
        "original_catalog_sha256": "1b94eaf06aeb416d9c2c4102939ba7aa062e801144f575586310b71b13ff7bad",
        "original_source_sha256": "943623da48c152b78ecbbbbe86a1468e477e3dd9e316de5e8b1c17d946425edb",
        "native_api": "torch.sgn(input,out=out)",
        "fresh_inputs": False,
    },
    "current-fresh": {
        "source_file": "benchmark/test_sgn_out.py",
        "benchmark_node": "benchmark/test_sgn_out.py::test_sgn_out",
        "shapes": [
            [1073741824],
            [64, 64],
            [4096, 4096],
            [64, 512, 512],
            [1024, 1024, 1024],
        ],
        "original_catalog_sha256": "1b94eaf06aeb416d9c2c4102939ba7aa062e801144f575586310b71b13ff7bad",
        "original_source_sha256": "bd6e99a5b4eb510bd66eb8c9ce3bc223cd2d866ec4d60beeb1b7259f73cbc9e9",
        "native_api": "torch.ops.aten.sgn.out(input,out=out)",
        "fresh_inputs": True,
    },
}
_SOURCE_BINDINGS = {
    "current-shared": {
        "_out_case_fn": "3f65cd08d0921dfc673a999e425f51e6346e807f1fb2722d5d3f4927e7ffa895",
        "_out_build_inputs_fn": "6a78fbb0faab3bfc6e1736df2219e3487ff6eb7bcd67ea4be8cef37aac520d4c",
        "source_file_sha256": "943623da48c152b78ecbbbbe86a1468e477e3dd9e316de5e8b1c17d946425edb",
    },
    "current-fresh": {
        "_case_fn": "88414178e65eec0d2b4866562c6febf729f23af1ac6d253e287fbfcbb6858448",
        "_build_inputs_fn": "c1f400adff2960eda451a89cc840e9cb17b179d8c41221f5eac906a6d5fd9fbf",
        "source_file_sha256": "bd6e99a5b4eb510bd66eb8c9ce3bc223cd2d866ec4d60beeb1b7259f73cbc9e9",
    },
}
_RECIPE_SOURCE = "2ff0931a646b97ec98ebd50642b5889619125a6b"
_CLONE_SOURCE_SHA256 = (
    "ca5f7cad847f6c0b9e1e9d8984bb987f634699cedbaea14d8937b7c022160e10"
)
_selected = _os.environ.get("OPS_BASELINE_SGN_OUT_PROFILE", "legacy")
if _selected not in ("legacy", "current-shared", "current-fresh"):
    raise ValueError(
        "OPS_BASELINE_SGN_OUT_PROFILE must be legacy, current-shared or current-fresh"
    )
# Explicit None also avoids carrying an earlier opt-in hook through module reload.
prepare_sample = None
if _selected != "legacy":
    IS_INPLACE = True
    _profile = _PROFILE_DATA[_selected]
    _shapes = [tuple(shape) for shape in _profile["shapes"]]

    def _current_shape(binding, dtype):
        if (
            set(binding) != {"shape"}
            or tuple(binding["shape"]) not in _shapes
            or dtype not in DTYPES
        ):
            raise ValueError("Unknown original sgn_out current profile cell")
        return tuple(binding["shape"])

    def native():
        if _selected == "current-shared":
            return torch.sgn
        return torch.ops.aten.sgn.out

    def grid():
        return [{"shape": list(shape)} for shape in _shapes]

    def key_shape(binding):
        shape = _current_shape(binding, DTYPES[0])
        return _selected + ":" + "x".join(map(str, shape))

    def build_inputs(binding, dtype, device):
        shape = _current_shape(binding, dtype)
        torch.manual_seed(0)
        inp = _current_input(shape, dtype, device)
        out = (
            torch.empty_like(inp)
            if _selected == "current-fresh"
            else torch.empty(shape, dtype=dtype, device=device)
        )
        return (inp,), {"out": out}

    def config(binding, dtype):
        shape = _current_shape(binding, dtype)
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size

        def tensor():
            return dict(
                shape=list(shape),
                dtype=str(dtype),
                stride=list(stride),
                requires_grad=False,
            )

        fresh = _selected == "current-fresh"
        return dict(
            profile=_selected,
            grid_contract="sgn_out.suiyuan." + _selected + ".core/v1",
            operator="sgn_out",
            benchmark_node=_profile["benchmark_node"],
            original_case_ordinal=_shapes.index(shape),
            original_case_params={},
            flag_gems_recipe_source=_RECIPE_SOURCE,
            original_target_catalog_sha256=_profile["original_catalog_sha256"],
            source_bindings=dict(_SOURCE_BINDINGS[_selected]),
            clone_source_sha256=_CLONE_SOURCE_SHA256 if fresh else None,
            native_api=_profile["native_api"],
            seed_policy="manual_seed(0)",
            fresh_inputs=fresh,
            out_initial_values_undefined=True,
            input_recipe=(
                "generate_tensor_input;empty_like(out)"
                if fresh
                else "generate_tensor_input;empty(shape,dtype,device)"
            ),
            native_timer_contract=dict(
                native_schema="aten::sgn.out",
                returns_sample_out=True,
                mutates=["out"],
                fresh_inputs=fresh,
                setup_timed=False,
                preparation=(
                    "clone_inputs_then_device_sync_before_start_event"
                    if fresh
                    else "none_reused_buffers"
                ),
                clock=(
                    "collector_CUDA_event_arithmetic_mean"
                    if fresh
                    else "unchanged_triton_do_bench"
                ),
                warmup_GPU_ms=25,
                measured_GPU_ms=100,
                NCU_supported=not fresh,
            ),
            producer_selector={"OPS_BASELINE_SGN_OUT_PROFILE": _selected},
            inputs=dict(args=[tensor()], kwargs={"out": tensor()}),
        )


if _selected == "current-fresh":

    def _clone_current_inputs(value):
        if isinstance(value, torch.Tensor):
            return value.detach().clone().requires_grad_(value.requires_grad)
        if isinstance(value, (tuple, list)):
            return type(value)(_clone_current_inputs(x) for x in value)
        if isinstance(value, dict):
            return {k: _clone_current_inputs(v) for k, v in value.items()}
        return value

    def prepare_sample(args, kwargs):
        # The collector synchronizes these real clones before its start event.
        return _clone_current_inputs((args, kwargs))
