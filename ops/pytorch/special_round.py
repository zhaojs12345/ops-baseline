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


"""special_round: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_special_round.py:9 (test_special_round)
Native expression: torch.special.round
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml UnaryPointwiseBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import standard_inputs, export_contract


def _resolve_native():
    return torch.special.round


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = lambda shape, dtype, device: standard_inputs("unary", shape, dtype, device)

globals().update(export_contract("special_round", DTYPES, _resolve_native, _inputs))

# UNAPPLIED operator-local current profile; full legacy source above is intact.
import os as _os
from _kernelgen import generate_tensor_input as _current_input

_selected = _os.environ.get("OPS_BASELINE_SPECIAL_ROUND_CURRENT_CORE", "0")
if _selected not in ["0", "1"]:
    raise ValueError("Unknown explicit special_round current profile selector")
if _selected == "1":
    OP_NAME = "special_round"
    IS_INPLACE = False
    DTYPES = [torch.float16, torch.float32, torch.bfloat16]
    SHAPES = [(536870912,), (64, 64), (4096, 4096), (64, 512, 512), (1024, 1024, 512)]

    def native():
        return torch.special.round

    def grid():
        return [{"shape": list(s)} for s in SHAPES]

    def key_shape(binding):
        return "current-decimals0" + ":" + "x".join(map(str, binding["shape"]))

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
        # Original builder is unseeded: do not reset process or device RNG.
        return (_current_input(shape, dtype, device),), {"decimals": 0}

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
            "profile": "current-decimals0",
            "operator": "special_round",
            "benchmark_node": "benchmark/test_special_round.py::test_special_round",
            "grid_contract": "special_round.suiyuan.current-decimals0.core/v1",
            "native_api": "torch.special.round(input,decimals=0)",
            "seed_policy": "unseeded_original_builder_no_reset",
            "fresh_inputs": False,
            "latency_boundary": "native_decimals_overload",
            "producer_selector": {"OPS_BASELINE_SPECIAL_ROUND_CURRENT_CORE": "1"},
            "flag_gems_recipe_source": "2ff0931a646b97ec98ebd50642b5889619125a6b",
            "source_bindings": {
                "builder": "b42c345aaba1f33f5fe541195a792536dccfeab29be18c57c69eda8a9bb9b641",
                "case_fn": "f18d265600090a634059e6a0aaa426aa947f984f2665bef5aa322eeb15c1c0bc",
                "input_generator": "e991af3667ae315b99d8508a74c52528e478f5bc00ddd96eab06dad01b219495",
                "generated_framework_sha256": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
                "base_sha256": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
                "backend_core_shapes_sha256": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
            },
        }
        result["inputs"] = {"args": [tensor], "kwargs": {"decimals": 0}}
        return result
