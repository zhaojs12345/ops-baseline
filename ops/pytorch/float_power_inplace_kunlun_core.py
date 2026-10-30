# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original KU float_power inplace core, with independent inputs before timing.

Depends on the reviewed prepare_sample collector contract. The callable itself
only executes the native inplace API. Original seed0 policy is retained without
claiming identical CPU/CUDA random bytes.
"""

import hashlib

import torch

OP_NAME = "float_power_inplace_kunlun_core"
DTYPES = [torch.float64]
IS_INPLACE = True
SHAPES = [(64, 64), (4096, 4096), (64, 512, 512)]
SCALARS = (0.5, -1.25, 2.0)
_samples = {}


def native():
    return torch.Tensor.float_power_


def grid():
    return [
        dict(shape=list(shape), overload=overload, exponent=exponent)
        for overload in ("Tensor", "Scalar")
        for shape in SHAPES
        for exponent in ((None,) if overload == "Tensor" else SCALARS)
    ]


def key_shape(binding):
    return [binding["shape"], binding["overload"], binding["exponent"]]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    torch.manual_seed(0)
    inp = torch.empty(shape, dtype=dtype, device=device).uniform_(0.25, 2.0)
    exponent = (
        torch.empty(shape, dtype=dtype, device=device).uniform_(-2.0, 3.0)
        if binding["overload"] == "Tensor"
        else binding["exponent"]
    )
    hashes = []
    for value in (inp, exponent):
        if isinstance(value, torch.Tensor):
            raw = value.detach().view(torch.uint8).cpu().numpy().tobytes()
            hashes.append(dict(sha256=hashlib.sha256(raw).hexdigest(), bytes=len(raw)))
    _samples[str(key_shape(binding))] = hashes
    return (inp, exponent), {}


def _clone_inputs(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone().requires_grad_(value.requires_grad)
    if isinstance(value, (tuple, list)):
        return type(value)(_clone_inputs(item) for item in value)
    if isinstance(value, dict):
        return {name: _clone_inputs(item) for name, item in value.items()}
    return value


def prepare_sample(args, kwargs):
    return _clone_inputs((args, kwargs))


def config(binding, dtype):
    shape = binding["shape"]
    stride = []
    running = 1
    for size in reversed(shape):
        stride.insert(0, running)
        running *= size
    meta = dict(shape=shape, dtype=str(dtype), stride=stride, requires_grad=False)
    tensor = binding["overload"] == "Tensor"
    return dict(
        native_expression="torch.Tensor.float_power_",
        canonical_operator=(
            "float_power_tensor_tensor_" if tensor else "float_power_tensor_scalar_"
        ),
        inputs=dict(
            args=[meta, meta.copy() if tensor else binding["exponent"]], kwargs={}
        ),
        params=dict(
            overload=binding["overload"],
            **({} if tensor else dict(exponent=binding["exponent"]))
        ),
        source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        benchmark="benchmark/test_float_power_.py",
        scope="original_kunlun_core",
        descriptor=shape,
        input_recipe=dict(
            self_uniform=[0.25, 2.0],
            tensor_exponent_uniform=[-2.0, 3.0] if tensor else None,
            seed=0,
        ),
        timing=dict(
            fresh_inputs=True,
            preparation="original detach.clone.requires_grad recursive recipe then synchronize before start event",
            clock="collector CUDA events",
            statistic="mean",
            warmup_ms=25,
            repetition_ms=100,
        ),
        producer_sample=_samples[str(key_shape(binding))],
    )
