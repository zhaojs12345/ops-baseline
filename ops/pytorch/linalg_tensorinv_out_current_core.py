# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native tensorinv(out=) with the unchanged FlagGems core SPD recipe.

The source builder consumes the caller's RNG stream without resetting a seed.
Keep that policy, record the actual producer RNG checkpoint and input digest,
and never claim random payload identity across different backends.
"""

import hashlib

import torch

OP_NAME = "linalg_tensorinv_out_current_core"
DTYPES = [torch.float32]
IS_INPLACE = True
BENCH_RETURN_MODE = "mean"
SOURCE_COMMIT = "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
CASES = [(2, 2), (4, 4), (8, 8), (16, 16), (24, 24)]
_samples = {}


def native():
    return torch.linalg.tensorinv


def grid():
    return [{"shape": list(shape)} for shape in CASES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    rng = (
        torch.cuda.get_rng_state(device)
        if torch.device(device).type == "cuda"
        else torch.get_rng_state()
    )
    matrix = torch.randn(shape, dtype=dtype, device=device)
    n = shape[0]
    matrix = matrix @ matrix.T + torch.eye(n, dtype=dtype, device=device) * 0.1
    out = torch.empty_like(matrix)
    payload = (
        matrix.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )
    _samples[(shape, str(dtype))] = {
        "rng_device_type": torch.device(device).type,
        "rng_state_before_hex": bytes(rng.cpu().tolist()).hex(),
        "input_sha256": hashlib.sha256(payload).hexdigest(),
        "input_bytes": len(payload),
    }
    return (matrix, 1), {"out": out}


def config(binding, dtype):
    shape = list(binding["shape"])
    stride = [shape[1], 1]
    tensor = dict(shape=shape, dtype=str(dtype), stride=stride, requires_grad=False)
    return {
        "inputs": {"args": [tensor, 1], "kwargs": {"out": dict(tensor)}},
        "source_commit": SOURCE_COMMIT,
        "benchmark": "benchmark/test_linalg_tensorinv.py",
        "scope": "core",
        "descriptor": shape,
        "variant": 0,
        "input_recipe": {
            "matrix": "torch.randn(shape,dtype,device); matrix@matrix.T+0.1*eye(n)",
            "rng_policy": "consume caller RNG stream; no per-case seed reset",
            "out": "torch.empty_like(matrix)",
            "native": "torch.linalg.tensorinv(matrix,1,out=out)",
        },
        "producer_sample": _samples[(tuple(shape), str(dtype))],
    }
