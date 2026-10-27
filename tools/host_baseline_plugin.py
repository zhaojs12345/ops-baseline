"""Source-only pytest instrumentation for collect_baseline_host.py.

Patches only the throwaway collector process; the FlagGems checkout is unmodified.
"""

import ast
import hashlib
import inspect
import json
import os
import platform
import sys
import time
from pathlib import Path

_HISTOGRAM_WRAPPER_AST_SHA256 = (
    "5f19fcb4a8386beeb92dd3847e8f40290354fe3614134efc29f72e703db8439b"
)
TRANSFER_OPERATORS = {"_to_sparse_semi_structured", "_histogramdd_from_bin_cts"}


def reviewed_histogram_wrapper(op):
    """Recognize only the original native CPU dispatch/transfer wrapper."""
    name = "benchmark.test_histogramdd_from_bin_cts"
    module = sys.modules.get(name)
    if (
        module is None
        or getattr(module, "_cpu_reference", None) is not op
        or getattr(op, "__module__", None) != name
    ):
        return False
    try:
        tree = ast.parse(inspect.getsource(op))
    except (OSError, TypeError, SyntaxError):
        return False
    if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
        return False
    # The JSON AST encoding is stable across Python's ast.dump formatting
    # changes. Ordinary-function type_params was added in Python 3.12.
    for node in ast.walk(tree.body[0]):
        if getattr(node, "type_params", []):
            return False

    def canonical(value):
        if isinstance(value, ast.AST):
            return [
                type(value).__name__,
                [
                    [key, canonical(item)]
                    for key, item in ast.iter_fields(value)
                    if key != "type_params"
                ],
            ]
        if isinstance(value, list):
            return [canonical(item) for item in value]
        return value

    try:
        source = json.dumps(canonical(tree.body[0]), separators=(",", ":")).encode()
    except (TypeError, ValueError):
        return False
    return hashlib.sha256(source).hexdigest() == _HISTOGRAM_WRAPPER_AST_SHA256


def describe_inputs(value):
    import torch

    if isinstance(value, torch.Tensor):
        result = {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "requires_grad": value.requires_grad,
        }
        if value.is_quantized:
            result.update(qscheme=str(value.qscheme()), axis=value.q_per_channel_axis())
            for key, tensor in [
                ("scales", value.q_per_channel_scales()),
                ("zero_points", value.q_per_channel_zero_points()),
            ]:
                result[key] = {
                    "dtype": str(tensor.dtype),
                    "shape": list(tensor.shape),
                    "sha256": hashlib.sha256(memoryview(tensor.numpy())).hexdigest(),
                }
        return result
    if isinstance(value, dict):
        return {key: describe_inputs(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [describe_inputs(item) for item in value]
    if isinstance(value, (torch.dtype, torch.device)):
        return str(value)
    return value


def input_contract(operator, args, kwargs):
    import torch

    config = describe_inputs({"args": args, "kwargs": kwargs})
    # Sparse packing depends on actual retained slots and values. CTC's tensor
    # lengths also affect eligibility; shape/dtype alone do not identify them.
    indices = (
        (0,)
        if operator == "_to_sparse_semi_structured"
        else (2, 3) if operator == "_use_cudnn_ctc_loss" else ()
    )
    for index in indices:
        tensor = args[index]
        if isinstance(tensor, torch.Tensor):
            payload = tensor.detach().cpu().contiguous().view(torch.uint8).numpy()
            config["args"][index]["payload_sha256"] = hashlib.sha256(
                memoryview(payload)
            ).hexdigest()
    return config


def pytest_collection_modifyitems(config, items):
    import torch
    from benchmark import base
    from torch.utils.benchmark import Timer

    operator = os.environ["HOST_BASELINE_OPERATOR"]
    torch.manual_seed(int(os.environ["HOST_BASELINE_SEED"]))
    threads = int(os.environ["HOST_BASELINE_THREADS"])
    minimum = float(os.environ["HOST_BASELINE_MIN_TIME"])
    cpu_info = Path("/proc/cpuinfo").read_text().split("\n\n", 1)[0]
    environment = {
        "architecture": platform.machine(),
        "cpu_info": cpu_info,
        "torch": str(torch.__version__),
        "threads": threads,
        "input_threads": torch.get_num_threads(),
        "python": platform.python_version(),
        "os": platform.platform(),
        "cuda": torch.version.cuda,
        "cuda_device": (
            torch.cuda.get_device_name() if torch.cuda.is_available() else None
        ),
        "timer": "torch.utils.benchmark.Timer",
        "min_run_time_seconds": minimum,
    }
    if operator == "_histogramdd_from_bin_cts":
        environment["cuda_device_uuid"] = str(torch.cuda.get_device_properties(0).uuid)
        environment["cuda_visible_devices"] = os.environ.get("CUDA_VISIBLE_DEVICES")

    def host_latency(self, op, *args, **kwargs):
        histogram_wrapper = (
            operator == "_histogramdd_from_bin_cts" and reviewed_histogram_wrapper(op)
        )
        if (
            not base.Config.reference_only
            or self.op_name != operator
            or op is not self.torch_op
            or not (
                isinstance(op, (torch._ops.OpOverloadPacket, torch._ops.OpOverload))
                or histogram_wrapper
            )
        ):
            raise ValueError(
                "Host collection only accepts the original native reference"
            )
        # Packing is host work reached through CUDA dispatch, including its
        # internal D2H/H2D transfers. The CTC predicate launches no computation:
        # a CUDA synchronization there would measure unrelated device work.
        synchronize = operator == "_to_sparse_semi_structured" or histogram_wrapper
        if synchronize:
            torch.cuda.synchronize()

        def call():
            result = op(*args, **kwargs)
            if synchronize:
                torch.cuda.synchronize()
            return result

        measurement = Timer(
            stmt="call()",
            globals={"call": call},
            num_threads=threads,
            timer=time.perf_counter,
        ).blocked_autorange(min_run_time=minimum)
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
            raise RuntimeError("Cannot identify active native host reference case")
        kind = (
            "cuda_dispatch_host_with_transfers"
            if synchronize
            else (
                "cuda_dispatch_host_predicate"
                if operator == "_use_cudnn_ctc_loss"
                else "native_cpu"
            )
        )
        active[0]["host_measurement"] = {
            "kind": kind,
            "environment": environment,
            "native_api": (
                "aten._histogramdd_from_bin_cts.default"
                if histogram_wrapper
                else str(op)
            ),
            "inputs": input_contract(operator, args, kwargs),
            "synchronized_cuda": synchronize,
            "number_per_run": measurement.number_per_run,
            "sample_count": len(measurement.raw_times),
            "latency_statistic": "median milliseconds",
            "opaque_packed_weight_replayable": (
                False
                if operator == "fbgemm_linear_fp16_weight_fp32_activation"
                else None
            ),
        }
        if histogram_wrapper:
            active[0]["host_measurement"]["reference_wrapper"] = {
                "source_ast_sha256": _HISTOGRAM_WRAPPER_AST_SHA256,
                "source_ast_scheme": "json_ast_without_empty_type_params_v1",
                "kind": "original native CPU histogram dispatch plus D2H/H2D transfers",
            }
        return measurement.median * 1000

    # Operator-local classes may override get_latency. Instrument those too,
    # without changing their builders, shapes, input ABI or native callable.
    for item in items:
        for value in list(vars(item.module).values()):
            if (
                isinstance(value, type)
                and value.__module__ == item.module.__name__
                and issubclass(value, base.Benchmark)
            ):
                value.get_latency = host_latency
