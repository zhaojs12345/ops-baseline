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


"""scaled_dot_product_flash_attention: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_scaled_dot_product_attention.py:39 (test_scaled_dot_product_flash_attention)
Native expression: torch.ops.aten._scaled_dot_product_flash_attention.default
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml AttentionBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def scaled_dot_product_flash_attention_input_fn(shape, dtype, device):
    query = torch.randn(shape, device=device, dtype=dtype)
    key = torch.randn(shape, device=device, dtype=dtype)
    value = torch.randn(shape, device=device, dtype=dtype)
    yield (query, key, value, 0.0, False, False)


def _resolve_native():
    return torch.ops.aten._scaled_dot_product_flash_attention.default


DTYPES = [torch.float16, torch.bfloat16]

_inputs = scaled_dot_product_flash_attention_input_fn

globals().update(
    export_contract(
        "scaled_dot_product_flash_attention", DTYPES, _resolve_native, _inputs
    )
)
