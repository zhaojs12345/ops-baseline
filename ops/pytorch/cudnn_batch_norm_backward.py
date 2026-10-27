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


"""cudnn_batch_norm_backward: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cudnn_batch_norm_backward.py:95 (test_cudnn_batch_norm_backward)
Native expression: torch.ops.aten.cudnn_batch_norm_backward
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml NormBenchmark or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def batchnorm_input_fn(shape, dtype, device):
    C = shape[1]
    inp = torch.randn(shape, dtype=dtype, device=device)
    weight = torch.randn((C,), dtype=dtype, device=device)
    bias = torch.randn((C,), dtype=dtype, device=device)
    running_mean = None
    running_var = None
    training = True
    momentum = 0.1
    eps = 1e-05
    cudnn_enabled = True
    yield (
        inp,
        weight,
        bias,
        running_mean,
        running_var,
        training,
        momentum,
        eps,
        cudnn_enabled,
    )


def cudnn_batch_norm_backward_input_fn(shape, dtype, device):
    for forward_args in batchnorm_input_fn(shape, dtype, device):
        inp, weight, bias, running_mean, running_var, training, _, eps, _ = forward_args
        grad_output = torch.randn_like(inp)
        channels = weight.shape[0] if weight is not None else inp.shape[1]
        if weight is None:
            weight = torch.ones(channels, dtype=dtype, device=device)
        if bias is None:
            bias = torch.ones(channels, dtype=dtype, device=device)
        inp_f32 = inp.to(torch.float32)
        weight_f32 = weight.to(torch.float32)
        bias_f32 = bias.to(torch.float32)
        out, save_mean, save_var, reserve = torch.ops.aten.cudnn_batch_norm(
            inp_f32, weight_f32, bias_f32, None, None, training, eps, False
        )
        save_mean = save_mean.to(dtype)
        save_var = save_var.to(dtype)
        reserve = reserve.to(dtype)
        yield (
            inp,
            grad_output,
            weight,
            running_mean,
            running_var,
            save_mean,
            save_var,
            eps,
            reserve,
        )


def _resolve_native():
    return torch.ops.aten.cudnn_batch_norm_backward


DTYPES = [torch.float32]

_inputs = cudnn_batch_norm_backward_input_fn

globals().update(
    export_contract("cudnn_batch_norm_backward", DTYPES, _resolve_native, _inputs)
)

# Preserve the pinned 9d57 catalog as the default. The repaired current FlagGems
# core is explicit and local to this operator; its full config/source is recorded.
import os as _os

_current_core = _os.environ.get("OPS_BASELINE_CUDNN_BN_CURRENT_CORE")
if _current_core not in (None, "0", "1"):
    raise ValueError("OPS_BASELINE_CUDNN_BN_CURRENT_CORE must be 0 or 1")
if _current_core == "1":
    OP_NAME = "cudnn_batch_norm_backward"
    DTYPES = [torch.float16, torch.float32]
    IS_INPLACE = False
    INPUT_CONTRACT = "cudnn-batch-norm-backward-training-invstd/v1"
    FLAG_GEMS_SOURCE_COMMIT = "6b9c9afd971e50a9e99ae31d73d816ca8b79560d"
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
            "grid_contract": "cudnn_batch_norm_backward.kernelgen-dev.core/v1",
            "producer_selector": {"OPS_BASELINE_CUDNN_BN_CURRENT_CORE": "1"},
            "input_builder_sha256": INPUT_BUILDER_SHA256,
            "flag_gems_source_commit": FLAG_GEMS_SOURCE_COMMIT,
        }
