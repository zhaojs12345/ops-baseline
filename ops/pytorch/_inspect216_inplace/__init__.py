# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exact inplace BLAS recipes and snapshot preparation outside CUDA events."""
import copy
import json
import math
from pathlib import Path

import torch

CATALOG = json.loads(Path(__file__).with_name("cases.json").read_text())
TIMING_PROTOCOL = dict(kind="cuda_events_fresh_inputs", warmup_ms=25, rep_ms=100,
                       preparation="detach+clone every input preserving dense strides; synchronize",
                       reset_in_measured_interval=False, cache_flush_bytes=256_000_000)


def grid(operator):
    return [{"recipe": index} for index in range(len(CATALOG["operators"][operator]))]


def key_shape(operator, binding):
    row = CATALOG["operators"][operator][binding["recipe"]]
    return "x".join(map(str, row["descriptor"])) + "_cm" + str(int(row["params"]["column_major"]))


def build_inputs(operator, binding, dtype, device):
    row = CATALOG["operators"][operator][binding["recipe"]]
    batch, m, n, k = row["descriptor"]
    column_major = row["params"]["column_major"]
    torch.manual_seed(0)
    if operator == "addmm_":
        mat1 = torch.randn((m, k), dtype=dtype, device=device)
        mat2 = torch.randn((n, k) if column_major else (k, n), dtype=dtype, device=device)
        mat2 = mat2.t() if column_major else mat2
        inp = torch.randn((m, n), dtype=dtype, device=device)
    else:
        mat1 = torch.randn((batch, m, k), dtype=dtype, device=device)
        mat2 = torch.randn((batch, n, k) if column_major else (batch, k, n), dtype=dtype, device=device)
        mat2 = mat2.transpose(1, 2) if column_major else mat2
        inp = torch.randn((batch, m, n), dtype=dtype, device=device)
    return (inp, mat1, mat2), {}


def config(operator, binding, dtype):
    row = copy.deepcopy(CATALOG["operators"][operator][binding["recipe"]])
    for item in row["inputs"]["args"]:
        item["dtype"] = str(dtype)
    return dict(inputs=row["inputs"], source_commit=CATALOG["source_commit"],
                benchmark=row["source_benchmark"], descriptor=row["descriptor"], scope="core",
                variant=int(row["params"]["column_major"]), params=row["params"],
                binding_defaults=row["binding_defaults"], effects=row["effects"], fresh_inputs=True,
                reference_implementation="pytorch_native_operator",
                latency_measurement=dict(TIMING_PROTOCOL))


def event_bench(operation, prepare, *, warmup_ms=25, rep_ms=100):
    """Time operation only; prepare, flush, allocation and synchronization precede start."""
    cache = torch.empty(256_000_000, dtype=torch.int8, device="cuda")
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    def sample():
        args, kwargs = prepare()
        cache.zero_()
        torch.cuda.synchronize()
        start.record()
        operation(*args, **kwargs)
        end.record()
        end.synchronize()
        return start.elapsed_time(end)

    # Untimed first call also covers native library initialization.
    args, kwargs = prepare()
    operation(*args, **kwargs)
    torch.cuda.synchronize()
    del args, kwargs
    estimate = sum(sample() for _ in range(5)) / 5
    if not math.isfinite(estimate) or estimate <= 0:
        raise RuntimeError("Invalid native event calibration")
    for _ in range(max(1, math.ceil(warmup_ms / estimate))):
        sample()
    values = [sample() for _ in range(max(5, math.ceil(rep_ms / estimate)))]
    if not all(math.isfinite(value) and value > 0 for value in values):
        raise RuntimeError("Invalid native event latency")
    return sum(values) / len(values)


def measure_latency(operation, args, kwargs):
    # The original initial state is immutable, including beta=1 self/bias.
    snapshots = tuple(value.detach().clone(memory_format=torch.preserve_format) for value in args)

    def prepare():
        fresh = tuple(value.detach().clone(memory_format=torch.preserve_format) for value in snapshots)
        return fresh, dict(kwargs)

    return event_bench(operation, prepare)
