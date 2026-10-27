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


"""embedding_bag_dense_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_embedding_bag_dense_backward.py:69 (test_embedding_bag_dense_backward)
Native expression: torch.ops.aten._embedding_bag_dense_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for num_bags, embedding_dim, num_weights, samples_per_bag in [shape]:
        num_samples = num_bags * samples_per_bag
        weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(
            0, num_weights, (num_samples,), dtype=torch.long, device=device
        )
        offsets = torch.arange(
            0, num_samples + 1, samples_per_bag, dtype=torch.long, device=device
        )[:num_bags]
        output, offset2bag, bag_size, maximum_indices = torch.ops.aten._embedding_bag(
            weight, indices, offsets, False, 0, False, None, False, -1
        )
        grad = torch.randn_like(output)
        yield (
            grad,
            indices,
            offset2bag,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            0,
            None,
            -1,
        )


def _resolve_native():
    return torch.ops.aten._embedding_bag_dense_backward


DTYPES = [torch.float16, torch.float32, torch.bfloat16]

_inputs = _input_iter

globals().update(
    export_contract("embedding_bag_dense_backward", DTYPES, _resolve_native, _inputs)
)

# Retain the pinned legacy catalog unless this target-metadata recipe is explicit.
import os as _os

_target_core = _os.environ.get("OPS_BASELINE_EMBEDDING_BAG_TARGET_CORE")
if _target_core not in (None, "0", "1"):
    raise ValueError("OPS_BASELINE_EMBEDDING_BAG_TARGET_CORE must be 0 or 1")
if _target_core == "1":
    OP_NAME = "embedding_bag_dense_backward"
    DTYPES = [torch.float16, torch.float32, torch.bfloat16]
    IS_INPLACE = False
    FLAG_GEMS_OPERATOR = "_embedding_bag_dense_backward"
    FLAG_GEMS_SOURCE_COMMIT = "1b96c4e3e6af089e5c1af7ca691712f95428ecea"
    INPUT_BUILDER_SHA256 = (
        "b98b44bdedf641bb73e5e8567bebbc6148984d23e5be7429b91abb37f87c45e4"
    )
    INPUT_CONTRACT = "_embedding_bag_dense_backward.sum_target_metadata/v1"
    GRID_CONTRACT = "_embedding_bag_dense_backward.kernelgen-dev.core/v1"
    SHAPES = [
        (8, 16, 50, 4),
        (16, 32, 100, 4),
        (32, 64, 100, 4),
        (64, 128, 200, 4),
        (128, 256, 500, 4),
        (256, 128, 500, 8),
        (512, 256, 1000, 8),
    ]

    def native():
        return torch.ops.aten._embedding_bag_dense_backward.default

    def grid():
        return [{"shape": list(shape), "variant": 0} for shape in SHAPES]

    def key_shape(binding):
        return list(binding["shape"])

    def _shape_and_dtype(binding, dtype):
        shape = tuple(binding["shape"])
        if shape not in SHAPES or dtype not in DTYPES or binding.get("variant", 0) != 0:
            raise ValueError("Unrecognized current embedding-bag core case")
        return shape

    def build_inputs(binding, dtype, device):
        num_bags, embedding_dim, num_weights, samples_per_bag = _shape_and_dtype(
            binding, dtype
        )
        torch.manual_seed(42)
        samples = num_bags * samples_per_bag
        weight = torch.randn(num_weights, embedding_dim, dtype=dtype, device=device)
        indices = torch.randint(0, num_weights, (samples,), device=device)
        offsets = torch.arange(0, samples, samples_per_bag, device=device)
        output, _, _, _ = torch.ops.aten._embedding_bag.default(
            weight, indices, offsets, False, 0, False, None, False, -1
        )
        # Match the target's existing optimized-sum metadata, without changing
        # any target input: only the missing mapping is repaired in FlagGems.
        mapping = (
            torch.arange(num_bags, dtype=torch.long, device="cpu")
            .repeat_interleave(samples_per_bag)
            .to(device)
        )
        bag_size = torch.zeros(num_bags, dtype=torch.long, device=device)
        maximum_indices = torch.zeros(num_bags, dtype=torch.long, device=device)
        grad = torch.randn_like(output)
        return (
            grad,
            indices,
            mapping,
            bag_size,
            maximum_indices,
            num_weights,
            False,
            0,
            None,
            -1,
        ), {}

    def config(binding, dtype):
        bags, width, weights, samples_per_bag = _shape_and_dtype(binding, dtype)
        samples = bags * samples_per_bag

        def tensor(shape, value_dtype, stride):
            return {
                "shape": shape,
                "dtype": str(value_dtype),
                "stride": stride,
                "requires_grad": False,
            }

        return {
            "inputs": {
                "args": [
                    tensor([bags, width], dtype, [width, 1]),
                    tensor([samples], torch.int64, [1]),
                    tensor([samples], torch.int64, [1]),
                    tensor([bags], torch.int64, [1]),
                    tensor([bags], torch.int64, [1]),
                    weights,
                    False,
                    0,
                    None,
                    -1,
                ],
                "kwargs": {},
            },
            "metadata_values": {
                "offset2bag": [
                    bag for bag in range(bags) for _ in range(samples_per_bag)
                ],
                "bag_size": [0] * bags,
                "maximum_indices": [0] * bags,
            },
            "tensor_alias_groups": [0, 1, 2, 3, 4],
            "seed": 42,
            "descriptor": [bags, width, weights, samples_per_bag],
            "variant": 0,
            "benchmark": "benchmark/test_embedding_bag_dense_backward.py",
            "flag_gems_operator": FLAG_GEMS_OPERATOR,
            "native_schema": "aten::_embedding_bag_dense_backward.default",
            "scope": "core",
            "input_contract": INPUT_CONTRACT,
            "grid_contract": GRID_CONTRACT,
            "producer_selector": {"OPS_BASELINE_EMBEDDING_BAG_TARGET_CORE": "1"},
            "input_builder_sha256": INPUT_BUILDER_SHA256,
            "flag_gems_source_commit": FLAG_GEMS_SOURCE_COMMIT,
        }
