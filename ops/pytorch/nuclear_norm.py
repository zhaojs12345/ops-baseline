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


"""Native nuclear_norm baseline, retaining the historical default catalog.

Legacy source: 9d57a23b4386bd669152a935fe88a96db907da7c; default grid,
input construction and metadata remain the original _kernelgen contract.
Explicit OPS_BASELINE_NUCLEAR_NORM_CURRENT_CORE=1 selects the exact six float32
core cases from FlagGems ac9a1f703757537a3d54cc1ac1febeca410316c0 instead.
Use a fresh output (not --resume across grid modes); preserve this selector in
collection launch/provenance. No shared catalog or collector change is needed.
"""

import os

import torch

from _kernelgen import export_contract


def nuclear_norm_input_fn(shape, cur_dtype, device):
    m, n = shape
    inp = torch.randn([m, n], dtype=torch.float32, device=device)
    yield (inp,)


def _resolve_legacy_native():
    return torch.nuclear_norm


OP_NAME = "nuclear_norm"
DTYPES = [torch.float32]
IS_INPLACE = False
SHAPES = [(16, 16), (32, 32), (64, 64), (128, 128), (256, 256), (512, 512)]


def native():
    return torch.nuclear_norm


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    if dtype != torch.float32 or tuple(binding["shape"]) not in SHAPES:
        raise ValueError("Expected an original float32 nuclear_norm core case")
    torch.manual_seed(0)
    inp = torch.randn(tuple(binding["shape"]), dtype=dtype, device=device)
    return (inp,), {"keepdim": False}


def config(binding, dtype):
    shape = list(binding["shape"])
    if dtype != torch.float32 or tuple(shape) not in SHAPES:
        raise ValueError("Expected an original float32 nuclear_norm core case")
    return {
        "inputs": {
            "args": [
                {
                    "shape": shape,
                    "dtype": str(dtype),
                    "stride": [shape[1], 1],
                    "requires_grad": False,
                }
            ],
            "kwargs": {"keepdim": False},
        },
        "grid_contract": "flaggems-ac9-core6",
        "reference_implementation": "torch.nuclear_norm",
        "input_recipe": "torch.manual_seed(0);torch.randn(shape,dtype=torch.float32,device=device)",
        "output_aliases": [],
    }


_grid_mode = os.environ.get("OPS_BASELINE_NUCLEAR_NORM_CURRENT_CORE", "0")
if _grid_mode == "0":
    # The default retains exact old shapes, kwargs and pinned source metadata.
    globals().update(
        export_contract(
            "nuclear_norm", DTYPES, _resolve_legacy_native, nuclear_norm_input_fn
        )
    )
elif _grid_mode != "1":
    raise ValueError("OPS_BASELINE_NUCLEAR_NORM_CURRENT_CORE must be 0 or 1")
