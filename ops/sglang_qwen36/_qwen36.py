# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Pinned source adapters shared by Qwen3.6 native recipes."""
import hashlib
import importlib.util
import os
from pathlib import Path

import torch


def common():
    source = os.environ.get("FLAGGEMS_SGLANG_SOURCE")
    if not source:
        raise RuntimeError("Set FLAGGEMS_SGLANG_SOURCE to the pinned workload checkout")
    path = Path(source) / "verification/qwen36_common.py"
    spec = importlib.util.spec_from_file_location("_baseline_qwen36", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def native(operator):
    return common().native_module().get_native(operator)


def grid(operator):
    return common().cases_module().workloads(operator)


def inputs(operator, row, dtype, device):
    helper = common()
    values = helper.cases_module().build_inputs(operator, row, dtype, device)
    if operator == "select_experts" and str(device) != "meta":
        values = helper.native_module().prepare_inputs(operator, values)
    return (), values


def key_shape(operator, row):
    helper = common()
    cases = helper.cases_module()
    _, values = inputs(operator, row, torch.bfloat16, "meta")
    return cases.baseline_key(operator, values)


def config(operator, row, dtype):
    helper = common()
    cases = helper.cases_module()
    _, values = inputs(operator, row, dtype, "meta")

    def describe(value):
        if isinstance(value, torch.Tensor):
            return dict(
                shape=list(value.shape),
                stride=list(value.stride()),
                dtype=str(value.dtype),
                storage_offset=value.storage_offset(),
            )
        from dataclasses import asdict, is_dataclass

        return dict(scalar=asdict(value) if is_dataclass(value) else value)

    path = helper.ROOT / "src/flaggems_sglang/testing/qwen36_workloads.py"
    return dict(
        inputs={name: describe(value) for name, value in values.items()},
        parameters=cases.parameters(operator, row),
        source=dict(
            upstream="sglang v0.5.11",
            commit="612785ffdcaf35552f1ed433a981d596ca9fe900",
            workload_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        ),
    )


def reset_module():
    return common().load(
        "src/flaggems_sglang/testing/reset_timing.py", "_baseline_reset"
    )


def make_input_reset(operator, inputs):
    return reset_module().capture_reset(operator, inputs)


def time_with_reset(fn, reset):
    return reset_module().measure_with_reset(fn, reset, torch.cuda)


def timing_contract():
    return dict(reset_module().TIMING)
