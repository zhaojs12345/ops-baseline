# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native CUDA GRU baseline; original FlagGems core grid and tensor recipes.
FlagGems source: b956dae66b57e94759cdd457b4f0d047221e35b9.
Each out buffer is passed by its actual keyword-only native schema.
"""

import torch

DTYPES = [torch.float16, torch.float32, torch.bfloat16, torch.float64]
CORE = [
    (1, 32, False),
    (8, 64, True),
    (32, 128, False),
    (64, 256, True),
    (128, 512, False),
    (256, 1024, True),
]


def _tensor(shape, dtype):
    stride = []
    size = 1
    for dim in reversed(shape):
        stride.insert(0, size)
        size *= dim
    return dict(shape=list(shape), dtype=str(dtype), stride=stride, requires_grad=False)


def grid():
    return [dict(batch=b, hidden=h, with_bias=bias) for b, h, bias in CORE]


def key_shape(binding):
    return f"B{binding['batch']}_H{binding['hidden']}_bias{int(binding['with_bias'])}"


OP_NAME = "_thnn_fused_gru_cell"
IS_INPLACE = False


def native():
    return torch.ops.aten._thnn_fused_gru_cell.default


def build_inputs(binding, dtype, device):
    b, h = binding["batch"], binding["hidden"]
    ig = torch.randn(b, 3 * h, dtype=dtype, device=device)
    hg = torch.randn_like(ig)
    hx = torch.randn(b, h, dtype=dtype, device=device)
    if binding["with_bias"]:
        ib = torch.randn(3 * h, dtype=dtype, device=device)
        hb = torch.randn_like(ib)
    else:
        ib = hb = None
    return (ig, hg, hx, ib, hb), {}


def config(binding, dtype):
    b, h = binding["batch"], binding["hidden"]
    t = lambda shape: _tensor(shape, dtype)
    args = [
        t((b, 3 * h)),
        t((b, 3 * h)),
        t((b, h)),
        t((3 * h,)) if binding["with_bias"] else None,
        t((3 * h,)) if binding["with_bias"] else None,
    ]
    kwargs = {}
    ordinal = next(
        i for i, (batch, hidden, _) in enumerate(CORE) if (batch, hidden) == (b, h)
    )
    return dict(
        inputs=dict(args=args, kwargs=kwargs),
        native_api="torch.ops.aten._thnn_fused_gru_cell.default",
        source_flaggems_commit="b956dae66b57e94759cdd457b4f0d047221e35b9",
        source_benchmark="benchmark/test_thnn_fused_gru_cell.py",
        case_id=f"benchmark/test_thnn_fused_gru_cell.py::test_thnn_fused_gru_cell::core::{str(dtype).removeprefix('torch.')}::{ordinal}",
        workspace_recipe="native CUDA forward; optional bias follows the original with_bias field",
        input_recipe="original dense contiguous torch.randn/empty recipe; no requires_grad",
    )
