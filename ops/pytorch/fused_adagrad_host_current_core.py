# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original eight Adagrad states; CPU-native with measured target transfers."""

import hashlib
import json
import time
from types import SimpleNamespace
import torch

OP_NAME = "fused_adagrad_host_current_core"
SOURCE_COMMIT = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
SHAPES = [(256, 256), (1024, 256), (2048, 512), (65536,)]
DTYPES = [torch.float32]
EXECUTION_KIND = "native_cpu_with_device_transfers"


def _adagrad_dense(tensor):
    if tensor.numel() <= 1:
        return True
    expected = 1
    for stride, size in sorted(
        (stride, size)
        for size, stride in zip(tensor.shape, tensor.stride())
        if size != 1
    ):
        if stride != expected:
            return False
        expected *= size
    return True


def _adagrad_validate(
    params, grads, state_sums, state_steps, grad_scale=None, found_inf=None
):
    n = len(params)
    for name, values in (
        ("grads", grads),
        ("state_sums", state_sums),
        ("state_steps", state_steps),
    ):
        if len(values) != n:
            raise RuntimeError(
                f"_fused_adagrad_: {name} must have the same length as params"
            )
    if not n:
        return
    dtype, device = params[0].dtype, params[0].device
    if dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
        raise RuntimeError("_fused_adagrad_ requires floating point parameters")
    for index, (param, grad, state_sum) in enumerate(zip(params, grads, state_sums)):
        for tensor in (param, grad, state_sum):
            if (
                tensor.dtype != dtype
                or tensor.device != device
                or tensor.layout != torch.strided
            ):
                raise RuntimeError(
                    "params, grads, and state_sums must have same dtype, device, and layout"
                )
            if not _adagrad_dense(tensor):
                raise RuntimeError(
                    "_fused_adagrad_ requires non-overlapping-and-dense tensors"
                )
        for other in (grad, state_sum):
            if other.shape != param.shape:
                raise RuntimeError(f"entry {index} does not match param shape")
            if any(
                size != 1 and left != right
                for size, left, right in zip(
                    param.shape, param.stride(), other.stride()
                )
            ):
                raise RuntimeError(f"entry {index} strides do not match param strides")
    for step in state_steps:
        if step.dtype != torch.float32:
            raise RuntimeError("state_steps must contain float32 scalar tensors")
        if step.numel() != 1:
            raise RuntimeError("state_steps must contain 1-element tensors")
        if step.device != device:
            raise RuntimeError("state_steps device is different from other tensors")
    for name, option in (("grad_scale", grad_scale), ("found_inf", found_inf)):
        if option is not None and (
            option.dtype != torch.float32
            or option.numel() != 1
            or option.device != device
        ):
            raise RuntimeError(
                f"{name} must be a float32 scalar on the original parameter device"
            )


def _adagrad_cpu_copy(tensor):
    # Independent CPU storage with the original dense strides; no dtype promotion.
    result = torch.empty_strided(
        tuple(tensor.shape), tuple(tensor.stride()), dtype=tensor.dtype, device="cpu"
    )
    result.copy_(tensor.detach().cpu())
    return result


def _adagrad_cpu_native(params, grads, state_sums, state_steps, **kwargs):
    # Validate every original target entry before any all-to-CPU conversion.
    _adagrad_validate(
        params,
        grads,
        state_sums,
        state_steps,
        kwargs.get("grad_scale"),
        kwargs.get("found_inf"),
    )
    p, g, s, st = (
        [_adagrad_cpu_copy(tensor) for tensor in values]
        for values in (params, grads, state_sums, state_steps)
    )
    options = {
        name: _adagrad_cpu_copy(value) if isinstance(value, torch.Tensor) else value
        for name, value in kwargs.items()
    }
    result = torch._fused_adagrad_(p, g, s, st, **options)
    assert result is None
    for values, originals in ((p, params), (s, state_sums)):
        for value, original in zip(values, originals):
            assert value.device.type == "cpu" and value.dtype == original.dtype
            assert value.shape == original.shape and value.stride() == original.stride()
    return p, s


def _adagrad_restore(value, original):
    result = torch.empty_strided(
        tuple(original.shape),
        tuple(original.stride()),
        dtype=original.dtype,
        device=original.device,
    )
    result.copy_(value)
    assert result.device == original.device and result.dtype == original.dtype
    assert result.shape == original.shape and result.stride() == original.stride()
    return result


def _adagrad_cpu_apply(params, grads, state_sums, state_steps, **kwargs):
    groups = (params, grads, state_sums, state_steps)
    frozen = [
        [
            (
                tensor,
                tensor.data_ptr(),
                tensor.device,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            )
            for tensor in group
        ]
        for group in groups
    ]
    p, s = _adagrad_cpu_native(params, grads, state_sums, state_steps, **kwargs)
    # Validation and the complete CPU operation succeed before any target write.
    for originals, values in ((params, p), (state_sums, s)):
        for original, value in zip(originals, values):
            original.copy_(value)
    for group, originals in zip(groups, frozen):
        assert len(group) == len(originals)
        for tensor, (original, pointer, device, dtype, shape, stride, offset) in zip(
            group, originals
        ):
            assert tensor is original and tensor.data_ptr() == pointer
            assert (
                tensor.device,
                tensor.dtype,
                tuple(tensor.shape),
                tuple(tensor.stride()),
                tensor.storage_offset(),
            ) == (device, dtype, shape, stride, offset)
    return None


_ADAGRAD_HOST_POLICY = dict(
    warmup_samples=1,
    minimum_samples=5,
    minimum_total_ms=100.0,
    maximum_samples=1000,
    cpu_threads=1,
    input_threads=2,
    statistic="median",
    clock="perf_counter_ns",
    fresh_state="full target clone before every single call; cloning/sync/hash untimed",
)


def _adagrad_tree_clone(value):
    if isinstance(value, torch.Tensor):
        return value.detach().clone(memory_format=torch.preserve_format)
    if isinstance(value, list):
        return [_adagrad_tree_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_adagrad_tree_clone(item) for item in value)
    if isinstance(value, dict):
        return {key: _adagrad_tree_clone(item) for key, item in value.items()}
    return value


def _adagrad_fingerprint(value):
    if isinstance(value, torch.Tensor):
        raw = (
            value.detach()
            .cpu()
            .contiguous()
            .reshape(-1)
            .view(torch.uint8)
            .numpy()
            .tobytes()
        )
        return dict(
            type="tensor",
            device=str(value.device),
            dtype=str(value.dtype),
            shape=list(value.shape),
            stride=list(value.stride()),
            layout=str(value.layout),
            values_sha256=hashlib.sha256(raw).hexdigest(),
        )
    if isinstance(value, (list, tuple)):
        return dict(
            type=type(value).__name__,
            items=[_adagrad_fingerprint(item) for item in value],
        )
    if isinstance(value, dict):
        return dict(
            type="dict",
            items={key: _adagrad_fingerprint(item) for key, item in value.items()},
        )
    return dict(
        type=type(value).__name__,
        value=value.hex() if isinstance(value, float) else value,
    )


def _adagrad_host_samples(fn, args, kwargs, synchronize):
    policy = dict(_ADAGRAD_HOST_POLICY)
    previous_threads = torch.get_num_threads()
    template = _adagrad_fingerprint((args, kwargs))
    samples, warmups = [], []
    try:
        while (
            len(warmups) < policy["warmup_samples"]
            or len(samples) < policy["minimum_samples"]
            or sum(item["elapsed_ns"] for item in samples) / 1e6
            < policy["minimum_total_ms"]
        ):
            if len(samples) >= policy["maximum_samples"]:
                raise RuntimeError("Adagrad host sample budget not reached")
            torch.set_num_threads(policy["input_threads"])
            fresh_args, fresh_kwargs = _adagrad_tree_clone((args, kwargs))
            synchronize()
            before = _adagrad_fingerprint((fresh_args, fresh_kwargs))
            if before != template:
                raise RuntimeError(
                    "Adagrad sample did not start from the complete original state"
                )
            lists = fresh_args[:4]
            frozen = [
                (
                    group,
                    tuple(
                        (
                            tensor,
                            tensor.data_ptr(),
                            tensor.device,
                            tensor.dtype,
                            tuple(tensor.shape),
                            tuple(tensor.stride()),
                            tensor.storage_offset(),
                        )
                        for tensor in group
                    ),
                )
                for group in lists
            ]
            torch.set_num_threads(policy["cpu_threads"])
            synchronize()
            started = time.perf_counter_ns()
            result = fn(*fresh_args, **fresh_kwargs)
            synchronize()
            elapsed = time.perf_counter_ns() - started
            if result is not None or elapsed <= 0:
                raise RuntimeError("Adagrad sample must execute one void-return call")
            for group, originals in frozen:
                if len(group) != len(originals):
                    raise RuntimeError("Adagrad candidate replaced an input list")
                for tensor, (
                    original,
                    pointer,
                    device,
                    dtype,
                    shape,
                    stride,
                    offset,
                ) in zip(group, originals):
                    if (
                        tensor is not original
                        or tensor.data_ptr() != pointer
                        or (
                            tensor.device,
                            tensor.dtype,
                            tuple(tensor.shape),
                            tuple(tensor.stride()),
                            tensor.storage_offset(),
                        )
                        != (device, dtype, shape, stride, offset)
                    ):
                        raise RuntimeError(
                            "Adagrad candidate changed original target ABI/alias/layout"
                        )
            after = _adagrad_fingerprint((fresh_args, fresh_kwargs))
            # Only parameter and state_sum values may change.
            for index in (1, 3):
                if (
                    before["items"][0]["items"][index]
                    != after["items"][0]["items"][index]
                ):
                    raise RuntimeError(
                        "Adagrad candidate mutated readonly gradients/steps"
                    )
            if before["items"][1] != after["items"][1]:
                raise RuntimeError("Adagrad candidate mutated options")
            row = dict(
                elapsed_ns=elapsed,
                input_state_sha256=hashlib.sha256(
                    json.dumps(before, sort_keys=True).encode()
                ).hexdigest(),
                single_call=True,
                original_target_abi_preserved=True,
                readonly_preserved=True,
            )
            (warmups if len(warmups) < policy["warmup_samples"] else samples).append(
                row
            )
        if _adagrad_fingerprint((args, kwargs)) != template:
            raise RuntimeError("Adagrad timing mutated the original template")
    finally:
        torch.set_num_threads(previous_threads)
    values = sorted(row["elapsed_ns"] for row in samples)
    count = len(values)
    median_ns = (values[(count - 1) // 2] + values[count // 2]) / 2
    return dict(
        latency_ms=median_ns / 1e6,
        measurement_method="synchronized_host_wall_single_call_fresh_state",
        policy=policy,
        samples=samples,
        warmups=warmups,
        original_template_unchanged=True,
        actual_input_fingerprint=template,
    )


def _build_inputs_fn(plan, dtype, device):
    shape, lr_decay, weight_decay, maximize = plan.builder_args
    torch.manual_seed(0)
    params = [torch.randn(shape, dtype=dtype, device=device)]
    grads = [torch.randn(shape, dtype=dtype, device=device)]
    state_sums = [torch.rand(shape, dtype=dtype, device=device) + 0.25]
    state_steps = [torch.tensor([3.0], dtype=torch.float32, device=device)]
    kwargs = {
        "lr": 0.01,
        "lr_decay": lr_decay,
        "weight_decay": weight_decay,
        "eps": 1e-10,
        "maximize": maximize,
        "grad_scale": None,
        "found_inf": None,
    }
    return params, grads, state_sums, state_steps, kwargs


def cases():
    for shape in SHAPES:
        for mode, decay, weight, maximize in [
            ("plain", 0.0, 0.0, False),
            ("options", 0.2, 0.01, True),
        ]:
            yield dict(
                shape=shape,
                mode=mode,
                lr_decay=decay,
                weight_decay=weight,
                maximize=maximize,
            )


def build_inputs(case, device):
    plan = SimpleNamespace(
        builder_args=(
            case["shape"],
            case["lr_decay"],
            case["weight_decay"],
            case["maximize"],
        )
    )
    values = _build_inputs_fn(plan, torch.float32, device)
    return tuple(values[:4]), values[4]


def native():
    # Standard GPU collectors receive only the real ATen callable. Its CUDA
    # dispatch is absent in this stack, so it cannot time a CPU roundtrip as GPU.
    return torch._fused_adagrad_


def host_native():
    """Explicit dedicated host entry; CPU-native with target transfers."""
    return cpu_native_transfer


def cpu_native_transfer(params, grads, state_sums, state_steps, **kwargs):
    return _adagrad_cpu_apply(params, grads, state_sums, state_steps, **kwargs)


def config(case):
    shape = list(case["shape"])
    stride = []
    product = 1
    for size in reversed(shape):
        stride.insert(0, product)
        product *= size
    tensor = dict(
        dtype="torch.float32",
        shape=shape,
        stride=stride,
        layout="torch.strided",
        requires_grad=False,
        device="target",
        storage_offset=0,
    )
    return dict(
        operator="_fused_adagrad_",
        source_commit=SOURCE_COMMIT,
        benchmark="benchmark/test_fused_adagrad_.py",
        scope="core",
        descriptor=shape,
        mode=case["mode"],
        dtype="torch.float32",
        native_api="torch._fused_adagrad_",
        execution_kind=EXECUTION_KIND,
        inputs=dict(
            params=[tensor],
            grads=[tensor],
            state_sums=[tensor],
            state_steps=[
                dict(
                    dtype="torch.float32",
                    shape=[1],
                    stride=[1],
                    layout="torch.strided",
                    requires_grad=False,
                    device="target",
                    storage_offset=0,
                    scalar_value=3.0,
                )
            ],
        ),
        options=dict(
            lr=0.01,
            lr_decay=case["lr_decay"],
            weight_decay=case["weight_decay"],
            eps=1e-10,
            maximize=case["maximize"],
            grad_scale=None,
            found_inf=None,
        ),
        input_recipe=dict(
            seed=0,
            params="target randn",
            grads="target randn",
            state_sums="target rand + 0.25",
            state_steps="target float32 [3.0]",
            cross_chip_values="same recipe; GPU random values not claimed equal",
        ),
        effects=dict(
            mutates=["params", "state_sums"],
            readonly=["grads", "state_steps", "grad_scale", "found_inf"],
            returns=None,
        ),
        measurement_policy=_ADAGRAD_HOST_POLICY,
    )
