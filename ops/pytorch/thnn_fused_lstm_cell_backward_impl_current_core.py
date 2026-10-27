# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact original dense fused-cell core recipe, FlagGems ac9a1f7037.
Native CUDA forward prepares workspace outside the backward timer. This module
collects only the named ATen backward ABI and does not time a portable CPU oracle.
"""

import torch

OP_NAME = "thnn_fused_lstm_cell_backward_impl_current_core"
_ATEN_NAME = "_thnn_fused_lstm_cell_backward_impl"
IS_INPLACE = False
DTYPES = [torch.float32]
SHAPES = [(1, 4, True), (4, 16, True), (16, 64, True)]
SOURCE_COMMIT = "ac9a1f703757537a3d54cc1ac1febeca410316c0"


def native():
    def run(*args):
        with torch.no_grad():
            return getattr(torch.ops.aten, _ATEN_NAME)(*args)

    return run


def grid():
    return [dict(shape=list(shape), variant=0) for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    batch, hidden, has_bias = binding["shape"]
    torch.manual_seed(0)
    input_gates = torch.randn(batch, 4 * hidden, dtype=dtype, device=device)
    hidden_gates = torch.randn_like(input_gates)
    state = torch.randn(batch, hidden, dtype=dtype, device=device)
    input_bias = torch.zeros(4 * hidden, dtype=dtype, device=device)
    hidden_bias = torch.randn(4 * hidden, dtype=dtype, device=device)
    with torch.no_grad():
        forward = torch.ops.aten._thnn_fused_lstm_cell(
            input_gates, hidden_gates, state, input_bias, hidden_bias
        )
    return (
        torch.randn_like(forward[0]),
        torch.randn_like(forward[1]),
        state,
        forward[1],
        forward[2],
        has_bias,
    ), {}


def config(binding, dtype):
    batch, hidden, has_bias = binding["shape"]

    def tensor(rows, cols):
        return dict(
            shape=[rows, cols], dtype=str(dtype), stride=[cols, 1], requires_grad=False
        )

    args = [tensor(batch, hidden) for _ in range(4)] + [
        tensor(batch, 4 * hidden),
        has_bias,
    ]
    return dict(
        inputs=dict(args=args, kwargs={}),
        source_commit=SOURCE_COMMIT,
        benchmark="benchmark/test_thnn_fused_lstm_cell_backward_impl.py",
        descriptor=list(binding["shape"]),
        scope="core",
        variant=0,
        native_api="torch.ops.aten." + _ATEN_NAME,
        input_recipe={
            "rng": "manual_seed(0)",
            "input_gates": "randn",
            "hidden_gates": "randn_like(input_gates)",
            "state": "randn",
            "input_bias": "zeros",
            "hidden_bias": "randn",
            "gradients": "randn_like forward outputs",
        },
        workspace_producer="torch.ops.aten._thnn_fused_lstm_cell native CUDA outside timer",
        autograd=False,
    )
