#!/usr/bin/env python3
"""Collect CUDA-dispatch _lazy_clone host metadata latency, without materializing.

The native API creates copy-on-write metadata. CUDA events do not measure its
host execution. Reuse ops/pytorch/_lazy_clone.py's exact target catalog with a
perf_counter host Timer instead. No payload is read, no copy is forced, and no
device synchronization is included in the timed call.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import time

from collection_environment import collect_environment

ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "ops/pytorch/_lazy_clone.py"


def source_state():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()

    return {
        "commit": git("rev-parse", "HEAD"),
        "dirty": bool(git("status", "--porcelain")),
    }


def load_recipe():
    spec = importlib.util.spec_from_file_location("lazy_clone_recipe", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metadata(tensor):
    return {
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "dtype": str(tensor.dtype),
        "device": str(tensor.device),
        "requires_grad": tensor.requires_grad,
        "is_conj": tensor.is_conj(),
        "is_neg": tensor.is_neg(),
    }


def collect_records(recipe, timer_class, *, threads, minimum, device):
    """Yield original case IDs and actual host measurements; no GPU timer path."""
    import torch

    native = recipe.native()
    if native is not torch.ops.aten._lazy_clone:
        raise ValueError("Only the original native aten::_lazy_clone is accepted")
    seen = set()
    for binding in recipe.grid():
        for dtype in recipe.DTYPES:
            config = recipe.config(binding, dtype)
            case_id = config["case_id"]
            if case_id in seen:
                raise ValueError(f"Duplicate native case ID: {case_id}")
            seen.add(case_id)
            args, kwargs = recipe.build_inputs(binding, dtype, device)
            inp = args[0]
            original = metadata(inp)
            expected = {
                "shape": config["input_shape"],
                "stride": config["input_stride"],
                "storage_offset": config["input_storage_offset"],
                "dtype": config["dtype"],
                "device": str(torch.device(device)),
                "requires_grad": config["input_requires_grad"],
                "is_conj": config["input_is_conj"],
                "is_neg": config["input_is_neg"],
            }
            if original != expected:
                raise ValueError(f"Recipe/input geometry mismatch for {case_id}")
            # Metadata inspection does not access data_ptr, storage bytes or
            # values: those observations could materialize COW and change cost.
            result = native(*args, **kwargs)
            if (
                result is inp
                or metadata(result) != original
                or metadata(inp) != original
            ):
                raise ValueError(f"Native lazy-clone metadata mismatch for {case_id}")
            del result

            def call():
                return native(*args, **kwargs)

            measured = timer_class(
                stmt="call()",
                globals={"call": call},
                num_threads=threads,
                timer=time.perf_counter,
            ).blocked_autorange(min_run_time=minimum)
            latency = measured.median * 1e6
            if not math.isfinite(latency) or latency <= 0:
                raise ValueError(f"Invalid native host latency for {case_id}")
            if metadata(inp) != original:
                raise ValueError(f"Native input metadata changed for {case_id}")
            yield {
                "case_id": case_id,
                "config": config,
                "inputs": [original],
                "T_host_us": latency,
                "host_measurement": {
                    "kind": "cuda_dispatch_host_metadata",
                    "timer": "torch.utils.benchmark.Timer/perf_counter",
                    "statistic": "median microseconds",
                    "threads": threads,
                    "min_run_time_seconds": minimum,
                    "number_per_run": measured.number_per_run,
                    "sample_count": len(measured.raw_times),
                    "synchronized_cuda": False,
                    "materializes_output": False,
                    "includes_output_destruction": True,
                },
            }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--min-run-time", type=float, default=0.05)
    args = parser.parse_args(argv)
    if (
        args.device < 0
        or args.threads < 1
        or not math.isfinite(args.min_run_time)
        or args.min_run_time <= 0
    ):
        parser.error(
            "device must be nonnegative; threads and timing budget must be positive"
        )
    output = args.output.resolve()
    if output == ROOT or ROOT in output.parents:
        parser.error("output must be outside the source checkout")
    state = source_state()
    if state["dirty"]:
        parser.error("collection requires a clean source checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    # Reserve the path before GPU work. Partial/failed runs are explicitly
    # incomplete and cannot masquerade as an accepted 60-case artifact.
    with output.open("x") as stream:
        json.dump({"_status": "initializing", "records": []}, stream)
    import torch
    from torch.utils.benchmark import Timer

    report = {
        "_format": "ops_pytorch_lazy_clone_host_v1",
        "_status": "incomplete",
        "_source": dict(state),
        "_measurement": {
            "kind": "cuda_dispatch_host_metadata",
            "not_a_gpu_kernel_baseline": True,
            "not_a_correctness_golden": True,
            "no_ncu": True,
        },
        "operator": "_lazy_clone",
        "records": [],
    }

    def save():
        temp = output.with_suffix(output.suffix + ".tmp")
        with temp.open("w") as stream:
            json.dump(report, stream, indent=2, allow_nan=False)
        os.replace(temp, output)

    try:
        if not torch.cuda.is_available():
            raise RuntimeError("Native H800 CUDA inputs required; no CPU fallback")
        torch.cuda.set_device(args.device)
        device = f"cuda:{torch.cuda.current_device()}"
        recipe = load_recipe()
        report["_source"].update(
            recipe_sha256=hashlib.sha256(MODULE.read_bytes()).hexdigest(),
            collector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        report["_environment"] = collect_environment(
            torch, vars(args) | {"output": str(output)}
        )
        for record in collect_records(
            recipe,
            Timer,
            threads=args.threads,
            minimum=args.min_run_time,
            device=device,
        ):
            report["records"].append(record)
            save()
        if source_state() != state:
            raise RuntimeError("Source changed during collection")
        if len(report["records"]) != 60:
            raise ValueError("Original target catalog must contain exactly 60 records")
        report["_status"] = "complete"
        save()
    except Exception as error:
        report["_error"] = f"{type(error).__name__}: {error}"
        save()
        raise
    print(f"Collected {len(report['records'])} native host metadata cases: {output}")


if __name__ == "__main__":
    main()
