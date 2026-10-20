"""Opt-in host-baseline replay for the six documented FlagGems operators.

Usage: PYTHONPATH=/path/to/ops-baseline/tools:$PYTHONPATH pytest ... \
    -p pytest_host_baseline --host-baseline /path/to/host.json --mode operator

Only operator-local benchmark classes are adapted, in this pytest process.
No source checkout or KernelGen code is modified.
"""

import hashlib
import json
import math
import os
import platform
import time
from dataclasses import asdict, dataclass
from functools import cached_property
from pathlib import Path

from collect_baseline_host import OPERATORS
from host_baseline_plugin import input_contract


def pytest_addoption(parser):
    parser.addoption(
        "--host-baseline",
        default=os.environ.get("FLAGGEMS_HOST_BASELINE"),
        help="Explicit native CPU/host artifact from collect_baseline_host.py",
    )


def _canonical(value):
    """Preserve CPU ABI; compare accelerator roles across CUDA and NPU devices."""
    if isinstance(value, dict):
        return {
            key: (
                "accelerator"
                if key == "device"
                and isinstance(item, str)
                and item.split(":")[0] in {"cuda", "npu"}
                else _canonical(item)
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_canonical(item) for item in value]
    return value


class HostArtifact:
    def __init__(self, path):
        self.path = Path(path)

    @cached_property
    def payload(self):
        def unique(pairs):
            result = {}
            for key, value in pairs:
                if key in result:
                    raise ValueError(f"Duplicate host baseline key: {key}")
                result[key] = value
            return result

        def reject(value):
            raise ValueError(f"Nonfinite host baseline JSON: {value}")

        raw = self.path.read_bytes()
        data = json.loads(raw, object_pairs_hook=unique, parse_constant=reject)
        source = data["_source"]["flaggems"]
        commit = source["commit"]
        if (
            data["_format"] != "flaggems_native_host_v1"
            or source["dirty"] is not False
            or not isinstance(commit, str)
            or len(commit) != 40
            or any(c not in "0123456789abcdef" for c in commit)
        ):
            raise ValueError("Host baseline requires a traceable clean source revision")
        return data, hashlib.sha256(raw).hexdigest()

    def lookup(self, operator, case_id, args, kwargs):
        data, digest = self.payload
        task = data["_tasks"][operator]
        if (
            task["status"] != "completed"
            or type(task["returncode"]) is not int
            or task["returncode"] != 0
        ):
            raise ValueError("Native host baseline task did not complete")
        matches = [r for r in data["operators"][operator] if r["case_id"] == case_id]
        if len(matches) != 1:
            raise ValueError("Expected exactly one matching native host case ID")
        record = matches[0]
        measurement = record["host_measurement"]
        expected_kind = (
            "cuda_dispatch_host_with_transfers"
            if operator == "_to_sparse_semi_structured"
            else (
                "cuda_dispatch_host_predicate"
                if operator == "_use_cudnn_ctc_loss"
                else "native_cpu"
            )
        )
        if (
            record["status"] != "PASSED"
            or record["operator"] != operator
            or measurement["kind"] != expected_kind
            or not measurement["environment"]["cpu_info"]
            or not measurement["environment"]["torch"]
            or measurement["environment"]["timer"] != "torch.utils.benchmark.Timer"
            or type(measurement["environment"]["threads"]) is not int
            or measurement["environment"]["threads"] < 1
            or type(measurement["sample_count"]) is not int
            or measurement["sample_count"] < 1
            or type(measurement["number_per_run"]) is not int
            or measurement["number_per_run"] < 1
            or measurement["native_api"]
            not in {
                f"aten.{operator}",
                f"aten.{operator}.default",
                f"aten.{operator}.out",
            }
        ):
            raise ValueError("Host record has no valid native source provenance")
        if measurement["synchronized_cuda"] is not (
            operator == "_to_sparse_semi_structured"
        ):
            raise ValueError("Host baseline synchronization scope is inconsistent")
        if operator in {"_to_sparse_semi_structured", "_use_cudnn_ctc_loss"}:
            gpu = measurement["environment"]["cuda_device"]
            if (
                not isinstance(gpu, str)
                or not gpu.startswith("NVIDIA ")
                or not measurement["environment"]["cuda"]
            ):
                raise ValueError(
                    "CUDA-dispatch host baseline lacks its actual NVIDIA environment"
                )
        actual = input_contract(operator, args, kwargs)
        if _canonical(actual) != _canonical(measurement["inputs"]):
            raise ValueError(
                "Host baseline input configuration differs from actual operands"
            )
        latency = record["latency_ms"]
        if (
            isinstance(latency, bool)
            or not isinstance(latency, (int, float))
            or not math.isfinite(latency)
            or latency <= 0
        ):
            raise ValueError("Host baseline latency must be finite and positive")
        return latency, {
            "kind": "external_host",
            "artifact": str(self.path.resolve()),
            "artifact_sha256": digest,
            "source": data["_source"],
            "source_task": task,
            "source_measurement": measurement,
            "ratio": "raw source host latency / target host latency; no GPU normalization",
        }


def pytest_collection_modifyitems(config, items):
    path = config.getoption("--host-baseline")
    if not path:
        return
    import flag_gems
    import torch
    from benchmark import base, consts
    from torch.utils.benchmark import Timer

    artifact = HostArtifact(path)  # Reading is deferred until a reference is timed.
    cpu_info = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]

    @dataclass
    class ExternalMetrics(base.BenchmarkMetrics):
        measurement_source: dict | None = None
        baseline_source: dict | None = None

    def adapt(cls, operator):
        original_measure = cls._measure_input
        original_reference = cls._run_reference_cases

        def target_source():
            return {
                "kind": "host_wall_time",
                "architecture": platform.machine(),
                "torch": str(torch.__version__),
                "threads": 1,
                "input_threads": torch.get_num_threads(),
                "cpu_info": cpu_info,
                "synchronized_accelerator": operator == "_to_sparse_semi_structured",
                "timer": "torch.utils.benchmark.Timer(perf_counter)",
                "target_vendor": flag_gems.vendor_name,
                "target_device": str(flag_gems.device),
            }

        def latency(self, op, *args, **kwargs):
            importing = base.Config.reference_only or getattr(
                self, "_host_native_pending", False
            )
            if importing:
                if base.Config.skip_native or op is not self.torch_op:
                    raise ValueError(
                        "Host import requires the original reference measurement"
                    )
                self._host_native_pending = False
                case_id = getattr(self, "_host_case_id", None)
                active = None
                if base.Config.reference_only:
                    active = [
                        r
                        for r in base.Config.reference_records
                        if r.get("operator") == operator
                        and r.get("nodeid") == base.Config.current_nodeid
                        and "failure" not in r
                        and r.get("status") == "FAILED"
                        and r.get("stage") == "benchmark_reference"
                    ]
                    if len(active) != 1:
                        raise RuntimeError(
                            "Cannot identify active external-host reference case"
                        )
                    case_id = active[0]["case_id"]
                result, self._host_imported = artifact.lookup(
                    operator, case_id, args, kwargs
                )
                self._host_imported["target_environment"] = target_source()
                if active is not None:
                    active[0]["baseline_source"] = self._host_imported
                return result

            def call():
                result = op(*args, **kwargs)
                if operator == "_to_sparse_semi_structured":
                    base.torch_device_fn.synchronize()
                return result

            if operator == "_to_sparse_semi_structured":
                base.torch_device_fn.synchronize()
            return (
                Timer(
                    stmt="call()",
                    globals={"call": call},
                    num_threads=1,
                    timer=time.perf_counter,
                )
                .blocked_autorange(min_run_time=0.05)
                .median
                * 1000
            )

        def measure(self, input, case_id=None):
            if (
                self.is_backward
                or base.Config.skip_native
                or base.Config.mode != consts.BenchMode.OPERATOR
                or any(m.startswith(("tflops", "gbps")) for m in self.to_bench_metrics)
            ):
                raise ValueError(
                    "External host baseline supports forward operator latency only"
                )
            self._host_case_id = case_id
            self._host_imported = None
            self._host_source = target_source()
            self._host_native_pending = "latency_base" in self.to_bench_metrics
            try:
                metric = original_measure(self, input, case_id)
                if metric.latency is not None and (
                    isinstance(metric.latency, bool)
                    or not math.isfinite(metric.latency)
                    or metric.latency <= 0
                ):
                    raise ValueError("Target host latency must be finite and positive")
                if metric.speedup is not None:
                    metric.speedup = metric.latency_base / metric.latency
                fields = asdict(metric)
                fields["measurement_source"] = target_source()
                fields["baseline_source"] = self._host_imported
                return ExternalMetrics(**fields)
            finally:
                self._host_native_pending = False

        def reference(self, case_ids):
            if base.Config.skip_native or self.is_backward:
                raise ValueError(
                    "External host reference needs a forward native baseline"
                )
            first = len(base.Config.reference_records)
            try:
                return original_reference(self, case_ids)
            finally:
                for record in base.Config.reference_records[first:]:
                    if record.get("status") == "PASSED" and record.get(
                        "baseline_source"
                    ):
                        record["stage"] = "external_host_baseline"

        cls.get_latency = latency
        cls._measure_input = measure
        cls._run_reference_cases = reference
        cls._external_host_adapted = True

    seen = set()
    for item in items:
        module = item.module
        operator = module.__name__.rsplit(".", 1)[-1].removeprefix("test_")
        if operator not in OPERATORS or not module.__name__.startswith("benchmark."):
            continue
        for cls in list(vars(module).values()):
            if (
                isinstance(cls, type)
                and cls.__module__ == module.__name__
                and issubclass(cls, base.Benchmark)
                and cls not in seen
            ):
                seen.add(cls)
                adapt(cls, operator)
