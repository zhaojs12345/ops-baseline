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

"""Exact repaired FlagGems core training cuDNN backward, NVIDIA-native only.

BF16 has no CuDNN implementation for the captured H800 environment; keep that
capability evidence separately. Never substitute another API/CPU latency.
"""

import torch

OP_NAME = "cudnn_batch_norm_backward"
DTYPES = [torch.float16, torch.float32]
IS_INPLACE = False
INPUT_CONTRACT = "cudnn-batch-norm-backward-training-invstd/v1"
FLAG_GEMS_SOURCE_COMMIT = "1aedcf1a3f659fbf8d2fa8f98d0b76c3869b82ae"
INPUT_BUILDER_SHA256 = (
    "acce93b32b5d669f49f136a364e0ee906cfb4e3032da48b6ae315c8bd278fa81"
)
SHAPES = [(16, 16, 64), (32, 64, 56, 56)]


def native():
    return torch.ops.aten.cudnn_batch_norm_backward


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    if dtype not in DTYPES:
        raise ValueError("No native H800 cuDNN backward baseline for this dtype")
    shape = tuple(binding["shape"])
    if shape not in SHAPES:
        raise ValueError("Unrecognized FlagGems core shape")
    torch.manual_seed(0)
    channels = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    grad = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.ones(channels, dtype=torch.float32, device=device)
    save_mean = torch.randn(channels, dtype=torch.float32, device=device)
    # Public save_var holds saved inverse standard deviation, not variance.
    save_var = torch.rand(channels, dtype=torch.float32, device=device) + 0.1
    reserve = torch.empty((0,), dtype=torch.uint8, device=device)
    return (inp, grad, weight, None, None, save_mean, save_var, 1e-5, reserve), {}


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")

    def encode(value):
        if not isinstance(value, torch.Tensor):
            return value
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }

    return {
        "inputs": {"args": [encode(value) for value in args], "kwargs": kwargs},
        "benchmark": "benchmark/test_cudnn_batch_norm_backward.py",
        "scope": "core",
        "input_contract": INPUT_CONTRACT,
        "input_builder_sha256": INPUT_BUILDER_SHA256,
        "flag_gems_source_commit": FLAG_GEMS_SOURCE_COMMIT,
    }
