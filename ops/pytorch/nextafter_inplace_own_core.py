# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact original nextafter_ profile; clone only mutable self before each event."""

import math
import torch

OP_NAME = "nextafter_inplace_own_core"
DTYPES = [torch.float16, torch.bfloat16, torch.float32, torch.float64]
IS_INPLACE = True
FG_SOURCE = "c0c8740c0d312b153c92e30511b739f49988f226"
BENCHMARK = "benchmark/test_nextafter_.py"
ORIGIN = "own"
BUILDER_AST_SHA256 = {
    "_case_fn": "4d1e07d4be240475f1f4c1de7155ec73e431cb0a58242188a3f4fc90df789701",
    "_build_inputs_fn": "a99e2b5f5789019bd6bfbf7a1e4144d2f07d13f098b8c3b80bf18c428219e067",
}
SHAPES = [(1073741824,), (64, 64), (4096, 4096), (64, 512, 512), (1024, 1024, 1024)]


def native():
    return torch.Tensor.nextafter_


def grid():
    return [dict(shape=list(shape), ordinal=i) for i, shape in enumerate(SHAPES)]


def key_shape(binding):
    return binding["shape"]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    if torch.device(device).type == "cuda":
        # Preflight probe and previous-loop arguments remain live in the
        # existing collector. Check current free memory before this allocation.
        free, total = torch.cuda.mem_get_info()
        stats = torch.cuda.memory_stats()
        inactive = (
            stats["reserved_bytes.all.current"] - stats["active_bytes.all.current"]
        )
        reclaimable = max(0, inactive - stats["inactive_split_bytes.all.current"])
        available = free + reclaimable
        required = 3 * math.prod(shape) * torch.empty((), dtype=dtype).element_size()
        assert available >= required + 8 * 1024**3, (
            "Insufficient free memory before original allocation",
            free,
            reclaimable,
            required,
            shape,
            dtype,
        )
    if ORIGIN == "own":
        torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    other = torch.randn(shape, dtype=dtype, device=device)
    return (inp, other), {}


def prepare_sample(args, kwargs):
    assert len(args) == 2 and not kwargs
    inp, other = args
    assert inp.untyped_storage()._cdata != other.untyped_storage()._cdata
    fresh = inp.clone(memory_format=torch.preserve_format)
    assert fresh.untyped_storage()._cdata != inp.untyped_storage()._cdata
    assert (
        fresh.shape == inp.shape
        and fresh.dtype == inp.dtype
        and fresh.device == inp.device
    )
    return (fresh, other), kwargs


def config(binding, dtype):
    shape = binding["shape"]
    args, kwargs = build_inputs(binding, dtype, "meta")
    describe = lambda t: dict(
        shape=list(t.shape),
        dtype=str(t.dtype),
        stride=list(t.stride()),
        storage_offset=t.storage_offset(),
        requires_grad=t.requires_grad,
    )
    original_shape = (
        {"inputs": [shape, shape]}
        if ORIGIN == "legacy"
        else {"self": shape, "other": shape}
    )
    return dict(
        canonical_operator="nextafter_",
        native_expression="torch.Tensor.nextafter_",
        source_commit=FG_SOURCE,
        benchmark=BENCHMARK,
        original_case_id=BENCHMARK
        + "::test_nextafter_::core::"
        + str(dtype).split(".")[-1]
        + "::"
        + str(binding["ordinal"]),
        ordinal=binding["ordinal"],
        origin=ORIGIN,
        original_shape=original_shape,
        original_params={},
        descriptor=shape,
        scope="core",
        builder_AST_SHA256=BUILDER_AST_SHA256,
        inputs=dict(args=[describe(t) for t in args], kwargs=kwargs),
        input_recipe=dict(
            generator="torch.randn",
            seed=0 if ORIGIN == "own" else None,
            dtype=str(dtype),
            allocation_order=["self", "other"],
        ),
        mutation=dict(mutates=["self"], readonly=["other"], result_alias="self"),
        timing_protocol=dict(
            prepare_sample="clone self only; keep original other object",
            prepare_before_CUDA_start_event=True,
            native_callable="Tensor.nextafter_",
            warmup_ms=25,
            rep_ms=100,
            return_mode="mean",
            ncu=False,
        ),
    )
