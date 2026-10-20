#!/usr/bin/env python3
"""Collect native CPU/host latencies through the original FlagGems case catalog.

This deliberately produces a separate host artifact. It does not collect NCU
metrics or pretend that CPU dispatch and CUDA-transfer time are GPU kernel time.
"""

import argparse
import hashlib
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

OPERATORS = (
    "_to_sparse_semi_structured",
    "_use_cudnn_ctc_loss",
    "fbgemm_linear_fp16_weight_fp32_activation",
    "fbgemm_linear_quantize_weight",
    "q_per_channel_scales",
    "q_per_channel_zero_points",
)


def git_info(path):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=path, text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=path,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
        )
        return {"commit": commit, "dirty": dirty}
    except (subprocess.CalledProcessError, OSError):
        return {"commit": None, "dirty": None}


def validate_report(report, operator, expected_ids=None):
    """Accept a complete successful source report, never a partial timing dump."""
    records = report.get("records", [])
    if report.get("status") != "PASSED" or not records:
        raise ValueError(f"{operator}: reference report is incomplete")
    seen = set()
    for record in records:
        latency = record.get("latency_ms")
        key = record.get("case_id")
        if (
            record.get("operator") != operator
            or record.get("status") != "PASSED"
            or not key
            or key in seen
            or isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
            or not record.get("host_measurement")
        ):
            raise ValueError(f"{operator}: invalid or duplicate native host record")
        seen.add(key)
    if expected_ids is not None and seen != set(expected_ids):
        raise ValueError(f"{operator}: timing case IDs do not match its source catalog")
    return records


def validate_catalog(catalog, operator):
    benches = catalog.get("benchmarks", [])
    if not benches or any(b.get("op_name") != operator for b in benches):
        raise ValueError(f"{operator}: source catalog contains another operator")
    ids = [c["case_id"] for bench in benches for c in bench["cases"]]
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{operator}: source catalog is empty or ambiguous")
    return ids


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--flaggems", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operators", nargs="+", choices=OPERATORS, default=OPERATORS)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument(
        "--input-threads",
        type=int,
        default=4,
        help="CPU threads for untimed input preparation; --threads controls timing",
    )
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--min-run-time", type=float, default=0.05)
    parser.add_argument(
        "--device",
        type=int,
        default=0,
        help="CUDA source device for the two CUDA-dispatch host APIs",
    )
    parser.add_argument("--op-timeout", type=float, default=3600)
    args = parser.parse_args(argv)
    if (
        args.threads < 1
        or args.input_threads < 1
        or args.device < 0
        or not math.isfinite(args.min_run_time)
        or args.min_run_time <= 0
        or not math.isfinite(args.op_timeout)
        or args.op_timeout <= 0
    ):
        parser.error("threads/time budgets must be positive and device nonnegative")
    source = args.flaggems.resolve(strict=True)
    if not (source / "benchmark/base.py").is_file():
        parser.error("--flaggems must point to a FlagGems checkout")
    source_info = git_info(source)
    commit = source_info["commit"]
    if (
        source_info["dirty"] is not False
        or not isinstance(commit, str)
        or len(commit) != 40
        or any(c not in "0123456789abcdef" for c in commit)
    ):
        parser.error("--flaggems requires a clean Git checkout with a full commit ID")
    output = args.output.resolve()
    if source == output or source in output.parents:
        parser.error("--output must be outside the FlagGems checkout")
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        parser.error("output already exists; choose a fresh run path")
    work = output.with_name(output.name + ".work")
    work.mkdir(exist_ok=False)
    env = os.environ.copy()
    # An inherited replay plugin or selection filter must not turn a native
    # complete collection into imported latencies or a hidden case subset.
    for key in (
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "FLAGGEMS_HOST_BASELINE",
        "FLAGGEMS_NV_BASELINE",
    ):
        env.pop(key, None)
    env["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    env.update(
        DNN_VENDOR="nvidia",
        CUDA_VISIBLE_DEVICES=str(args.device),
        OMP_NUM_THREADS=str(args.input_threads),
        MKL_NUM_THREADS=str(args.input_threads),
        HOST_BASELINE_THREADS=str(args.threads),
        HOST_BASELINE_SEED=str(args.seed),
        HOST_BASELINE_MIN_TIME=str(args.min_run_time),
        PYTHONPATH=os.pathsep.join(
            (str(Path(__file__).parent), str(source / "src"), str(source))
        ),
    )

    result = {
        "_format": "flaggems_native_host_v1",
        "_tasks": {},
        "operators": {},
        "_source": {
            "flaggems": source_info,
            "collector": git_info(Path(__file__).parent),
            "collector_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "plugin_sha256": hashlib.sha256(
                Path(__file__).with_name("host_baseline_plugin.py").read_bytes()
            ).hexdigest(),
        },
        "_measurement": {
            "seed": args.seed,
            "threads": args.threads,
            "input_threads": args.input_threads,
            "min_run_time_seconds": args.min_run_time,
            "not_a_gpu_kernel_baseline": True,
            "not_a_correctness_golden": True,
        },
    }
    for operator in dict.fromkeys(args.operators):
        module = source / "benchmark" / f"test_{operator}.py"
        target = work / f"{operator}.json"
        env["HOST_BASELINE_OPERATOR"] = operator
        command = [
            sys.executable,
            "-m",
            "pytest",
            str(module),
            "-q",
            "--tb=short",
            "--level",
            "core",
            "--mode",
            "operator",
            "--reference-only",
            "--output",
            str(target),
            "-p",
            "host_baseline_plugin",
        ]
        started = time.monotonic()
        catalog_path = work / f"{operator}.catalog.json"
        catalog_command = [
            sys.executable,
            "-m",
            "pytest",
            str(module),
            "-q",
            "--tb=short",
            "--level",
            "core",
            "--list-cases",
            "--output",
            str(catalog_path),
        ]

        def execute(argv, suffix):
            if git_info(source) != source_info:
                raise ValueError("FlagGems source changed during collection")
            with (work / f"{operator}.{suffix}.log").open("w") as log:
                run = subprocess.Popen(
                    argv,
                    cwd=source,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                try:
                    code = run.wait(timeout=args.op_timeout)
                except subprocess.TimeoutExpired:
                    os.killpg(run.pid, signal.SIGTERM)
                    try:
                        run.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        os.killpg(run.pid, signal.SIGKILL)
                        run.wait()
                    code = 124
            if git_info(source) != source_info:
                raise ValueError("FlagGems source changed during collection")
            return code

        expected_ids = None
        catalog_error = None
        try:
            code = execute(catalog_command, "catalog")
            if code == 0:
                expected_ids = validate_catalog(
                    json.loads(catalog_path.read_text()), operator
                )
                code = execute(command, "reference")
        except (ValueError, OSError, KeyError) as error:
            code = 1
            catalog_error = str(error)
        task = {
            "returncode": code,
            "status": "failed",
            "seconds": time.monotonic() - started,
            "benchmark": str(module.relative_to(source)),
            "benchmark_sha256": hashlib.sha256(module.read_bytes()).hexdigest(),
        }
        if catalog_error:
            task["error"] = catalog_error
        if code == 0:
            try:
                result["operators"][operator] = validate_report(
                    json.loads(target.read_text()), operator, expected_ids
                )
                task["status"] = "completed"
            except (ValueError, OSError) as error:
                task["error"] = str(error)
        result["_tasks"][operator] = task
        temporary = output.with_name(output.name + ".tmp")
        temporary.write_text(json.dumps(result, indent=2, allow_nan=False))
        temporary.replace(output)
        print(f"{operator}: {task['status']}", flush=True)
    return (
        0 if all(t["status"] == "completed" for t in result["_tasks"].values()) else 1
    )


if __name__ == "__main__":
    raise SystemExit(main())
