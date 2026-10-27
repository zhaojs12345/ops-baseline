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


"""cholesky_inverse: PyTorch native baseline from FlagGems kernelgen-dev.
Source commit: 9d57a23b4386bd669152a935fe88a96db907da7c
Benchmark: benchmark/test_cholesky_inverse.py:38 (test_cholesky_inverse)
Native expression: torch.cholesky_inverse
Input construction: upstream benchmark helpers/base class, CORE only.
Shape source: benchmark/core_shapes.yaml benchmark class/default descriptors or explicit class descriptors.
Comprehensive-only shapes are not imported.
"""

import torch

from _kernelgen import export_contract


def _input_iter(shape, dtype, device):
    for shape in [shape]:
        n = shape[-1]
        B = torch.randn(shape, dtype=dtype, device=device)
        A = B @ B.transpose(-2, -1) + torch.eye(n, dtype=dtype, device=device) * n
        L = torch.linalg.cholesky(A)
        yield (L,)


def _resolve_native():
    return torch.cholesky_inverse


DTYPES = [torch.float32, torch.float64]

_inputs = _input_iter

globals().update(export_contract("cholesky_inverse", DTYPES, _resolve_native, _inputs))

# Legacy defaults above remain the pinned 9d57 export contract. This explicit
# profile times native H800 inverse on the original Ascend core factor layout;
# it does not change the target benchmark to fit NVIDIA's default factor stride.
import os as _os

_ascend_core = _os.environ.get("OPS_BASELINE_CHOLESKY_ASCEND_CORE")
if _ascend_core not in (None, "0", "1"):
    raise ValueError("OPS_BASELINE_CHOLESKY_ASCEND_CORE must be 0 or 1")
if _ascend_core == "1":
    OP_NAME = "cholesky_inverse"
    DTYPES = [torch.float32]
    IS_INPLACE = False
    FLAG_GEMS_SOURCE_COMMIT = "0b3c96ae2624ed17f85c1499f4ecf64899ec3dcb"
    INPUT_BUILDER_SHA256 = (
        "65ab582417c249d275c752645cf920ab432cd64eff1ef18a55022659dc6f2242"
    )
    SHAPES = [(n, n) for n in (2, 4, 8, 16, 32, 64, 128, 256)]

    def native():
        return torch.cholesky_inverse

    def grid():
        return [{"shape": list(shape)} for shape in SHAPES]

    def key_shape(binding):
        return list(binding["shape"])

    def build_inputs(binding, dtype, device):
        if dtype != torch.float32:
            raise ValueError("Ascend current-core profile supports float32 only")
        shape = tuple(binding["shape"])
        if shape not in SHAPES:
            raise ValueError("Unrecognized cholesky_inverse current-core shape")
        torch.manual_seed(0)
        n = shape[-1]
        matrix = torch.randn(shape, dtype=dtype, device=device)
        positive_definite = (
            matrix @ matrix.transpose(-2, -1)
            + torch.eye(n, dtype=dtype, device=device) * n
        )
        # Original Ascend linalg.cholesky returns [n,1], unlike CUDA's [1,n].
        # The H800 operation is actually measured on this row-major factor.
        factor = torch.linalg.cholesky(positive_definite).contiguous()
        return (factor,), {"upper": False}

    def config(binding, dtype):
        args, kwargs = build_inputs(binding, dtype, "meta")
        factor = args[0]
        return {
            "inputs": {
                "args": [
                    {
                        "shape": list(factor.shape),
                        "dtype": str(factor.dtype),
                        "stride": list(factor.stride()),
                        "requires_grad": factor.requires_grad,
                    }
                ],
                "kwargs": kwargs,
            },
            "benchmark": "benchmark/test_cholesky_inverse.py",
            "scope": "core",
            "input_contract": "cholesky-inverse-lower-spd-factor/v1",
            "grid_contract": "cholesky_inverse.kernelgen-dev.core/v1",
            "layout_profile": "ascend-native-factor-row-major/v1",
            "producer_selector": {"OPS_BASELINE_CHOLESKY_ASCEND_CORE": "1"},
            "input_recipe": "manual_seed(0); matrix@matrix.T+n*eye; cholesky; row-major factor; upper=False",
            "input_builder_hash_method": "ast.get_source_segment(build_inputs)",
            "input_builder_sha256": INPUT_BUILDER_SHA256,
            "flag_gems_source_commit": FLAG_GEMS_SOURCE_COMMIT,
        }
