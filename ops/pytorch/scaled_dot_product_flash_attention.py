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


"""Nine-output SDPA native core recipes from both original benchmark modules.

Shared recipes keep their old shape keys and randn construction. Dedicated
recipes add distinct causal/scale/dropout/debug configurations and uniform
inputs; they never reuse records from the five-output flash-forward API.
"""
import copy
import json
import math
from pathlib import Path

import torch

OP_NAME = "scaled_dot_product_flash_attention"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
_CATALOG = json.loads(Path(__file__).with_name("_inspect216_sdpa_flash_cases.json").read_text())


def native():
    return torch.ops.aten._scaled_dot_product_flash_attention.default


def grid():
    return [{"recipe": index} for index in range(len(_CATALOG["recipes"]))]


def _recipe(binding):
    return _CATALOG["recipes"][binding["recipe"]]


def key_shape(binding):
    recipe = _recipe(binding)
    return recipe["shape"]["query"] if recipe["key"].startswith("shared_") else recipe["key"]


def build_inputs(binding, dtype, device):
    inp = _recipe(binding)["inputs"]
    if inp["seed"] is not None:
        torch.manual_seed(inp["seed"])
    tensors = []
    for descriptor in inp["args"][:3]:
        if descriptor["factory"] == "torch.randn":
            tensor = torch.randn(descriptor["shape"], dtype=dtype, device=device)
        elif descriptor["factory"] == "torch.empty.uniform_(-0.05,0.05)":
            tensor = torch.empty(descriptor["shape"], dtype=dtype, device=device).uniform_(-.05, .05)
        else:
            raise ValueError("Unknown frozen SDPA input factory")
        tensors.append(tensor)
    return (*tensors, *inp["args"][3:]), dict(inp["kwargs"])


def config(binding, dtype):
    recipe = _recipe(binding)
    inputs = copy.deepcopy(recipe["inputs"])
    for index, descriptor in enumerate(inputs["args"][:3]):
        shape = descriptor["shape"]
        inputs["args"][index] = dict(shape=shape, dtype=str(dtype), requires_grad=False,
                                     stride=[math.prod(shape[i + 1:]) for i in range(len(shape))])
    inputs.pop("seed")
    return dict(inputs=inputs, source_commit=_CATALOG["source_commit"],
                benchmark=recipe["benchmark"], scope="core", variant=recipe["ordinal"],
                case_id=recipe["case_ids"][str(dtype)], case_shape=copy.deepcopy(recipe["shape"]),
                case_params=copy.deepcopy(recipe["params"]),
                reference_implementation="pytorch_native_operator")
