# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native H800 sparse softmax; canonical inputs for the original six core rows.

Shape/dtype source is FlagGems38fd6d1b benchmark/test__sparse_softmax.py. CPU code
prepares inputs only. The timed callable is native CUDA ATen, never a CPU oracle.
This new deterministic fixture does not reproduce old random tensors bit-for-bit.
"""

import torch
from _sparse_softmax_inputs import (
    RECIPE,
    forward_cpu,
    input_identity,
    seed_hex,
    to_device,
)

OP_NAME = "_sparse_softmax"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
CASE_SOURCE = "38fd6d1ba0374f40f9c60b6291d951e24e52a4c0"
BENCHMARK_SHA256 = "f44a7b4388513d8e3279183f9b2c6b92f8fc5b76321887ad3786b43aa4d10a88"
SHAPES = (
    ((1024, 1024), 0, 1),
    ((1024, 1024), 262144, 1),
    ((1024, 1024), 1048576, 1),
    ((4096, 4096), 1048576, 1),
    ((2048, 2048, 64), 1048576, 2),
    ((256, 256, 256), 1048576, 2),
)


def native():
    try:
        return torch.ops.aten._sparse_softmax.default
    except (AttributeError, RuntimeError):
        return None


def grid():
    return [
        {"shape": list(shape), "nnz": nnz, "dim": dim, "ordinal": i}
        for i, (shape, nnz, dim) in enumerate(SHAPES)
    ]


def _checked(binding, dtype):
    if dtype not in DTYPES:
        raise TypeError("Unsupported original sparse softmax dtype")
    if not isinstance(binding, dict) or type(binding.get("ordinal")) is not int:
        raise ValueError("Expected an original descriptor and ordinal")
    i = binding["ordinal"]
    if not 0 <= i < len(SHAPES) or binding != grid()[i]:
        raise ValueError("Binding does not match the original core descriptor")
    return SHAPES[i]


def _cpu_inputs(binding, dtype):
    shape, nnz, dim = _checked(binding, dtype)
    return {"input": forward_cpu(shape, nnz, dim, dtype)}


def build_inputs(binding, dtype, device):
    _checked(binding, dtype)
    moved = to_device(_cpu_inputs(binding, dtype), device)
    return (moved["input"],), {"dim": binding["dim"], "half_to_float": False}


def key_shape(binding):
    shape, nnz, dim = _checked(binding, DTYPES[0])
    return "coo_" + "x".join(map(str, shape)) + f"_nnz_{nnz}_sd_{len(shape)}_dim_{dim}"


def config(binding, dtype):
    shape, nnz, dim = _checked(binding, dtype)
    return {
        "native_expression": "torch.ops.aten._sparse_softmax.default",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_softmax.py",
        "benchmark_sha256": BENCHMARK_SHA256,
        "scope": "core",
        "case_id": (
            "benchmark/test__sparse_softmax.py"
            "::test__sparse_softmax::core::"
            f"{str(dtype).removeprefix('torch.')}::{binding['ordinal']}"
        ),
        "recipe": RECIPE,
        "seed_hex": seed_hex("forward", shape, nnz, len(shape), dim, dtype),
        "requested_nnz": nnz,
        "sparse_dim": len(shape),
        "dim": dim,
        "half_to_float": False,
        "input_device_role": "target",
        "inputs": input_identity(_cpu_inputs(binding, dtype)),
    }
