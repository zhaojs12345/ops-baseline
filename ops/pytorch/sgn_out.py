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

# UNAPPLIED operator-local current profile; full legacy source above is intact.
import os as _os
from _kernelgen import generate_tensor_input as _current_input

_selected = _os.environ.get("OPS_BASELINE_SGN_OUT_PROFILE", "legacy")
if _selected not in ["legacy", "current-shared"]:
    raise ValueError("Unknown explicit sgn_out current profile selector")
if _selected == "current-shared":
    OP_NAME = "sgn_out"
    IS_INPLACE = True
    DTYPES = [torch.float16, torch.float32, torch.bfloat16]
    SHAPES = [(536870912,), (64, 64), (4096, 4096), (64, 512, 512), (1024, 1024, 512)]

    def native():
        return torch.sgn

    def grid():
        return [{"shape": list(s)} for s in SHAPES]

    def key_shape(binding):
        return "current-shared" + ":" + "x".join(map(str, binding["shape"]))

    def _current_shape(binding, dtype):
        if (
            set(binding) != {"shape"}
            or tuple(binding["shape"]) not in SHAPES
            or dtype not in DTYPES
        ):
            raise ValueError("Unknown original current profile shape/dtype")
        return tuple(binding["shape"])

    def build_inputs(binding, dtype, device):
        shape = _current_shape(binding, dtype)
        torch.manual_seed(0)
        inp = _current_input(shape, dtype, device)
        return (inp,), {"out": torch.empty(shape, dtype=dtype, device=device)}

    def config(binding, dtype):
        shape = _current_shape(binding, dtype)
        stride = []
        extent = 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size
        tensor = dict(
            shape=list(shape), dtype=str(dtype), stride=stride, requires_grad=False
        )
        result = {
            "profile": "current-shared",
            "operator": "sgn_out",
            "benchmark_node": "benchmark/test_sgn.py::test_sgn_out",
            "grid_contract": "sgn_out.suiyuan.current-shared.core/v1",
            "native_api": "torch.sgn(input,out=out)",
            "seed_policy": "manual_seed(0)",
            "fresh_inputs": False,
            "latency_boundary": "native_reused_buffers",
            "producer_selector": {"OPS_BASELINE_SGN_OUT_PROFILE": "current-shared"},
            "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
            "source_bindings": {
                "builder": "7832876e0f939715b2db757d73588782fe1e246417239fd54a4ee13ada1c9f95",
                "case_fn": "62de112a82930bc067be2781fc41c01eb2e11d19735ce2cb9cd5cd50e9879f98",
                "input_generator": "e991af3667ae315b99d8508a74c52528e478f5bc00ddd96eab06dad01b219495",
                "generated_framework_sha256": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
                "base_sha256": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
                "backend_core_shapes_sha256": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
            },
        }
        result["inputs"] = {"args": [tensor], "kwargs": {"out": tensor}}
        return result
