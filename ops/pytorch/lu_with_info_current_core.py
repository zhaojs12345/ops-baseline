# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native _lu_with_info for the five original FP32 target core cases."""
import torch

OP_NAME = "lu_with_info_current_core"
DTYPES = [torch.float32]
IS_INPLACE = False
SHAPES = [(64, 64), (256, 256), (512, 512), (1024, 1024), (8, 128, 128)]
SOURCE_COMMIT = "ac9a1f703757537a3d54cc1ac1febeca410316c0"


def native():
    return torch._lu_with_info


def grid():
    return SHAPES


def key_shape(shape):
    return "x".join(map(str, shape))


def build_inputs(shape, dtype, device):
    torch.manual_seed(0)
    return (torch.randn(shape, dtype=dtype, device=device),), {
        "pivot": True, "check_errors": False
    }


def config(shape, dtype):
    stride = []
    product = 1
    for size in reversed(shape):
        stride.insert(0, product)
        product *= size
    return {
        "operator": "_lu_with_info",
        "native_api": "torch._lu_with_info",
        "source_commit": SOURCE_COMMIT,
        "benchmark": "benchmark/test_lu_with_info.py",
        "scope": "core",
        "shape": list(shape),
        "dtype": str(dtype),
        "inputs": {
            "args": [{"kind": "tensor", "shape": list(shape), "stride": stride,
                      "dtype": str(dtype), "requires_grad": False}],
            "kwargs": {"pivot": True, "check_errors": False},
        },
        "input_recipe": {"manual_seed": 0, "factory": "torch.randn",
                         "cross_device_rng": "same distribution, not byte-identical"},
        "effects": {"mutates": [], "return_aliases_input": False},
        "reference_implementation": "pytorch_native_operator",
    }
