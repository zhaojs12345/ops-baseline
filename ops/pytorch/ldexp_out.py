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

# Explicit current seeded core profile. Legacy default above remains intact.
import os as _os

_CURRENT_SELECTED = _os.environ.get("OPS_BASELINE_LDEXP_OUT_CURRENT_CORE", "0")
if _CURRENT_SELECTED not in ("0", "1"):
    raise ValueError("OPS_BASELINE_LDEXP_OUT_CURRENT_CORE must be 0 or 1")
if _CURRENT_SELECTED == "1":
    IS_INPLACE = True
    _CURRENT_SHAPES = [
        (1073741824,),
        (64, 64),
        (4096, 4096),
        (64, 512, 512),
        (1024, 1024, 1024),
    ]
    _SOURCE_BINDINGS = {
        "_case_fn": "76d3b145ad3ba1a2dfb50d7d39e1c2c0eb2985fb58099689e578bbb1403d26af",
        "_build_inputs_fn": "1b3e7d963b530b11549967ce595fbd2e32c79375536b1ca53e6e12937b713f22",
        "_out_build_inputs_fn": "69bf0ff0d5a5d81bbde435157880bffe1fdd504b8bbb66adbe4494b2d25738e3",
        "benchmark/generated_operator_utils.py": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
        "benchmark/base.py": "daa8c109e43a06b03f34a14333135cf8cdc6efdbedab082eedb3bc4d7a3e3f5d",
        "benchmark/core_shapes.yaml": "720ad7776ce6e172e6c6eb78ee94b84b243ec7ee84a295543ac76fddd63b664d",
        "src/flag_gems/runtime/backend/_enflame/core_shapes.yaml": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
    }

    def _current_shape(binding, dtype):
        if (
            set(binding) != {"shape"}
            or tuple(binding["shape"]) not in _CURRENT_SHAPES
            or dtype not in DTYPES
        ):
            raise ValueError("Unknown original ldexp_out core shape/dtype")
        return tuple(binding["shape"])

    def native():
        return torch.ops.aten.ldexp.out

    def grid():
        return [{"shape": list(shape)} for shape in _CURRENT_SHAPES]

    def key_shape(binding):
        return "current-seed0-out:" + "x".join(
            map(str, _current_shape(binding, DTYPES[0]))
        )

    def build_inputs(binding, dtype, device):
        shape = _current_shape(binding, dtype)
        torch.manual_seed(0)
        inp = torch.randn(shape, dtype=dtype, device=device)
        exponent = torch.randint(-8, 9, shape, dtype=torch.int32, device=device)
        return (inp, exponent), {"out": torch.empty_like(inp)}

    def config(binding, dtype):
        shape = _current_shape(binding, dtype)
        stride, extent = [], 1
        for size in reversed(shape):
            stride.insert(0, extent)
            extent *= size

        def tensor(dt):
            return dict(
                shape=list(shape),
                dtype=str(dt),
                stride=list(stride),
                requires_grad=False,
            )

        return dict(
            profile="current-seed0-out",
            grid_contract="ldexp_out.suiyuan.current-seed0.core/v1",
            operator="ldexp_out",
            benchmark_node="benchmark/test_ldexp.py::test_ldexp_out",
            original_case_ordinal=_CURRENT_SHAPES.index(shape),
            original_case_params={"exponent_dtype": "int32"},
            flag_gems_recipe_source="2ff0931a646b97ec98ebd50642b5889619125a6b",
            producer_selector={"OPS_BASELINE_LDEXP_OUT_CURRENT_CORE": "1"},
            source_bindings=dict(_SOURCE_BINDINGS),
            inputs=dict(
                args=[tensor(dtype), tensor(torch.int32)], kwargs={"out": tensor(dtype)}
            ),
            input_recipe="manual_seed(0);randn(self);randint(-8,9,int32);empty_like(out)",
            out_initial_values_undefined=True,
            native_timer_contract=dict(
                native_schema="aten::ldexp.out",
                argument_order=["self", "other"],
                keyword_tensor="out",
                returns_original_out=True,
                mutates=["out"],
                fresh_inputs=False,
                setup_timed=False,
                original_is_inplace_flag=True,
            ),
        )
