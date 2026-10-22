# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native H800 sparse softmax backward; original six core descriptors/dtypes.

Shape source is FlagGems38fd6d1b benchmark/test__sparse_softmax_backward_data.py.
Only the input recipe becomes canonical; native ATen arguments and the original
independent random grad/output inputs (not probabilities) are preserved. Old12
published random-input records are not this recipe and must not be relabeled.
"""

import torch
from _sparse_softmax_inputs import (
    RECIPE,
    backward_cpu,
    input_identity,
    seed_hex,
    to_device,
)

OP_NAME = "_sparse_softmax_backward_data"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
CASE_SOURCE = "38fd6d1ba0374f40f9c60b6291d951e24e52a4c0"
BENCHMARK_SHA256 = "277a4abf680476fc90612fdd696698d81c72dd87a5d2d8b196caf372b80bfd4f"
SHAPES = (
    ((1024, 1024), 65536, 2, 0),
    ((2048, 2048), 1048576, 2, 1),
    ((256, 256, 256), 262144, 3, 0),
    ((4096, 4096, 64), 65536, 2, 2),
    ((1048576, 128), 65536, 1, 1),
    ((16, 7, 57, 32, 29), 6, 5, 4),
)


def native():
    try:
        return torch.ops.aten._sparse_softmax_backward_data.default
    except (AttributeError, RuntimeError):
        return None


def grid():
    return [
        {"shape": list(shape), "nnz": nnz, "sparse_dim": sd, "dim": dim, "ordinal": i}
        for i, (shape, nnz, sd, dim) in enumerate(SHAPES)
    ]


def _checked(binding, dtype):
    if dtype not in DTYPES:
        raise TypeError("Unsupported original sparse backward dtype")
    if not isinstance(binding, dict) or type(binding.get("ordinal")) is not int:
        raise ValueError("Expected an original descriptor and ordinal")
    i = binding["ordinal"]
    if not 0 <= i < len(SHAPES) or binding != grid()[i]:
        raise ValueError("Binding does not match the original core descriptor")
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
    return "coo_" + "x".join(map(str, shape)) + f"_nnz_{nnz}_sd_{sd}_dim_{dim}"


def config(binding, dtype):
    shape, nnz, sd, dim = _checked(binding, dtype)
    return {
        "native_expression": "torch.ops.aten._sparse_softmax_backward_data.default",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_softmax_backward_data.py",
        "benchmark_sha256": BENCHMARK_SHA256,
        "scope": "core",
        "case_id": (
            "benchmark/test__sparse_softmax_backward_data.py"
            "::test__sparse_softmax_backward_data::core::"
            f"{str(dtype).removeprefix('torch.')}::{binding['ordinal']}"
        ),
        "recipe": RECIPE,
        "seed_hex": seed_hex("backward", shape, nnz, sd, dim, dtype),
        "requested_nnz": nnz,
        "sparse_dim": sd,
        "dim": dim,
        "input_device_role": "target",
        "inputs": input_identity(_cpu_inputs(binding, dtype)),
    }
