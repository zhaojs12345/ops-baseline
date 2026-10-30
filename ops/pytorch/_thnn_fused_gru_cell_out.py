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
    return [dict(batch=b, hidden=h, with_bias=True) for b, h, bias in CORE]


def key_shape(binding):
    return f"B{binding['batch']}_H{binding['hidden']}_bias{int(binding['with_bias'])}"


OP_NAME = "_thnn_fused_gru_cell_out"
IS_INPLACE = True


def native():
    return torch.ops.aten._thnn_fused_gru_cell.out


def build_inputs(binding, dtype, device):
    b, h = binding["batch"], binding["hidden"]
    ig = torch.randn(b, 3 * h, dtype=dtype, device=device)
    hg = torch.randn_like(ig)
    hx = torch.randn(b, h, dtype=dtype, device=device)
    ib = torch.randn(3 * h, dtype=dtype, device=device)
    hb = torch.randn_like(ib)
    return (ig, hg, hx, ib, hb), dict(
        out0=torch.empty_like(hx),
        out1=torch.empty(b, 5 * h, dtype=dtype, device=device),
    )


def config(binding, dtype):
    b, h = binding["batch"], binding["hidden"]
    t = lambda shape: _tensor(shape, dtype)
    args = [t((b, 3 * h)), t((b, 3 * h)), t((b, h)), t((3 * h,)), t((3 * h,))]
    kwargs = dict(out0=t((b, h)), out1=t((b, 5 * h)))
    ordinal = next(
        i for i, (batch, hidden, _) in enumerate(CORE) if (batch, hidden) == (b, h)
    )
    return dict(
        inputs=dict(args=args, kwargs=kwargs),
        native_api="torch.ops.aten._thnn_fused_gru_cell.out",
        source_flaggems_commit="b956dae66b57e94759cdd457b4f0d047221e35b9",
        source_benchmark="benchmark/test_thnn_fused_gru_cell.py",
        case_id=f"benchmark/test_thnn_fused_gru_cell.py::test_thnn_fused_gru_cell_out::core::{str(dtype).removeprefix('torch.')}::{ordinal}",
        workspace_recipe="native CUDA forward with independent random gates/hx and both biases",
        input_recipe="original dense contiguous torch.randn/empty recipe; no requires_grad",
    )
