# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch backward GPU baseline; canonical input preparation only.

All original core descriptors and F32/F64 widths retained. No CPU oracle is timed.
Sparse support follows the original source; PCG64 normal VALUES are a new recipe.
"""

import torch
from _sparse_log_softmax_inputs import (
    RECIPE,
    seed_hex,
    backward_cpu,
    input_identity,
    to_device,
)

OP_NAME = "_sparse_log_softmax_backward_data"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
CASE_SOURCE = "f30a563479449016490ef674b3670ede6c101988"
BENCHMARK_SHA256 = "b825cab775806d3912d0fbe9c1976d3fabdb919165546d5e0ccbfb216daeccc0"
SHAPES = (
    ((1048576,), 1048576, 1, 0),
    ((1024, 1024), 1048576, 2, 1),
    ((2048, 2048), 1048576, 2, 1),
    ((4096, 4096), 1048576, 2, 1),
    ((256, 256, 256), 1048576, 3, 2),
    ((1024, 64), 1024, 1, 1),
    ((1024, 64, 16), 8192, 3, 1),
    ((4096,), 8192, 1, 0),
    ((), 1, 0, -1),
    ((), 0, 0, -1),
    ((1048576,), 0, 1, 0),
)


def native():
    try:
        return getattr(torch.ops.aten, OP_NAME).default
    except (AttributeError, RuntimeError):
        return None


def grid():
    return [
        {"shape": list(shape), "nnz": nnz, "sparse_dim": sd, "dim": dim, "ordinal": i}
        for i, (shape, nnz, sd, dim) in enumerate(SHAPES)
    ]


def _checked(binding, dtype):
    if dtype not in DTYPES:
        raise TypeError("Unsupported original dtype")
    if not isinstance(binding, dict) or type(binding.get("ordinal")) is not int:
        raise ValueError("Expected original descriptor and ordinal")
    i = binding["ordinal"]
    if not 0 <= i < len(SHAPES) or binding != grid()[i]:
        raise ValueError("Not an exact original core descriptor")
    return SHAPES[i]


def _cpu_inputs(binding, dtype):
    shape, nnz, sd, dim = _checked(binding, dtype)
    grad, output, self_tensor = backward_cpu(shape, nnz, sd, dim, dtype)
    return {"grad": grad, "output": output, "self": self_tensor}


def build_inputs(binding, dtype, device):
    _checked(binding, dtype)
    moved = to_device(_cpu_inputs(binding, dtype), device)
    return (moved["grad"], moved["output"], binding["dim"], moved["self"]), {}


def key_shape(binding):
    shape, nnz, sd, dim = _checked(binding, DTYPES[0])
    return (
        "coo_"
        + ("x".join(map(str, shape)) or "scalar")
        + f"_nnz_{nnz}_sd_{sd}_dim_{dim}"
    )


def config(binding, dtype):
    shape, nnz, sd, dim = _checked(binding, dtype)
    return {
        "native_expression": "torch.ops.aten._sparse_log_softmax_backward_data.default",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_log_softmax_backward_data.py",
        "benchmark_sha256": BENCHMARK_SHA256,
        "scope": "core",
        "case_id": "benchmark/test__sparse_log_softmax_backward_data.py::test__sparse_log_softmax_backward_data_benchmark::core::"
        + f"{str(dtype).removeprefix('torch.')}::{binding['ordinal']}",
        "recipe": RECIPE,
        "seed_hex": seed_hex("backward", shape, nnz, sd, dim, dtype),
        "requested_nnz": nnz,
        "sparse_dim": sd,
        "dim": dim,
        "input_device_role": "target",
        "inputs": input_identity(_cpu_inputs(binding, dtype)),
    }
