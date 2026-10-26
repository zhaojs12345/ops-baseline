# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native GPU packing with the original FlagGems core input recipes.

Only aten._pack_padded_sequence is timed. Input construction and transfers are
outside the timer; its batch_sizes output retains the native CPU ABI.
"""
import math

import torch

OP_NAME = "pack_padded_sequence"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
SOURCE_COMMIT = "00cd80f89b12864b2e6cf711db283d3d95b2187a"
SOURCE_BENCHMARK = "benchmark/test_pack_padded_sequence.py"
SHAPES = [(16, 8, 64), (32, 16, 128), (64, 32, 256), (128, 64, 256)]


def native():
    return torch.ops.aten._pack_padded_sequence.default


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def _lengths(shape):
    time_steps, batch, _ = shape
    lengths = torch.linspace(time_steps, 1, steps=batch, dtype=torch.int64).clamp(min=1)
    return torch.sort(lengths, descending=True).values


def build_inputs(binding, dtype, device):
    torch.manual_seed(0)
    shape = binding["shape"]
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp, _lengths(shape), False), {}


def _descriptor(shape, dtype):
    return dict(shape=list(shape), dtype=str(dtype),
                stride=[math.prod(shape[i + 1:]) for i in range(len(shape))],
                requires_grad=False)


def config(binding, dtype):
    shape = binding["shape"]
    lengths = _lengths(shape)
    lengths_description = _descriptor(lengths.shape, lengths.dtype)
    lengths_description.update(device="cpu", values=lengths.tolist())
    return dict(inputs=dict(args=[_descriptor(shape, dtype), lengths_description, False], kwargs={}),
                source_commit=SOURCE_COMMIT, benchmark=SOURCE_BENCHMARK,
                descriptor=list(shape), scope="core", variant=0,
                reference_implementation="pytorch_native_operator")
