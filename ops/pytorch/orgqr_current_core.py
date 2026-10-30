# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native orgqr with the unchanged FlagGems core Householder recipe.

Factorization prepares the input outside the timed orgqr call. The caller RNG
stream is consumed without resetting a seed; producer checkpoints qualify only
this collection, never cross-device random-byte identity.
"""
import hashlib

import torch

OP_NAME = "orgqr_current_core"
DTYPES = [torch.float32, torch.float64]
IS_INPLACE = False
BENCH_RETURN_MODE = "mean"
SOURCE_COMMIT = "9c280800abb30b0fa3e1c073d564fc4b6d97d1cd"
SHAPES = [(16, 8), (32, 16), (64, 32), (128, 64), (128, 128)]
_samples = {}


def native():
    return torch.orgqr


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    rng = (torch.cuda.get_rng_state(device)
           if torch.device(device).type == "cuda" else torch.get_rng_state())
    matrix = torch.randn(shape, dtype=dtype, device=device)
    inp, tau = torch.geqrf(matrix)
    payloads = [value.detach().contiguous().reshape(-1).view(torch.uint8)
                .cpu().numpy().tobytes() for value in (inp, tau)]
    _samples[(shape, str(dtype))] = {
        "rng_device_type": torch.device(device).type,
        "rng_state_before_hex": bytes(rng.cpu().tolist()).hex(),
        "input_sha256": [hashlib.sha256(value).hexdigest() for value in payloads],
        "input_bytes": [len(value) for value in payloads],
    }
    return (inp, tau), {}


def config(binding, dtype):
    m, n = binding["shape"]
    return {
        "inputs": {"args": [
            dict(shape=[m, n], dtype=str(dtype), stride=[1, m], requires_grad=False),
            dict(shape=[n], dtype=str(dtype), stride=[1], requires_grad=False),
        ], "kwargs": {}},
        "source_commit": SOURCE_COMMIT,
        "benchmark": "benchmark/test_orgqr.py",
        "scope": "core",
        "descriptor": [m, n],
        "variant": 0,
        "input_recipe": {
            "matrix": "torch.randn(shape,dtype,device)",
            "factorization": "torch.geqrf(matrix) outside timed call",
            "rng_policy": "consume caller RNG stream; no per-case seed reset",
            "native": "torch.orgqr(input,tau)",
        },
        "producer_sample": _samples[((m, n), str(dtype))],
    }
