"""Host timing must call native metadata work, never a CUDA-event timer."""

import importlib.util
from pathlib import Path
import sys
import time
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

TOOLS = Path(__file__).resolve().parents[1] / "tools"
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location(
    "lazy_host", TOOLS / "collect_lazy_clone_host.py"
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


def test_original_catalog_uses_host_timer_without_gpu_observation():
    calls = []

    class Timer:
        def __init__(self, **kwargs):
            assert kwargs["stmt"] == "call()"
            assert kwargs["timer"] is time.perf_counter
            assert kwargs["num_threads"] == 1
            self.call = kwargs["globals"]["call"]

        def blocked_autorange(self, *, min_run_time):
            assert min_run_time == 0.05
            result = self.call()
            calls.append(host.metadata(result))
            return SimpleNamespace(median=2e-6, number_per_run=1, raw_times=[2e-6])

    recipe = host.load_recipe()
    with patch.object(
        torch.cuda, "synchronize", side_effect=AssertionError("unexpected CUDA sync")
    ):
        records = list(
            host.collect_records(recipe, Timer, threads=1, minimum=0.05, device="cpu")
        )
    assert len(records) == len(calls) == 60
    assert len({r["case_id"] for r in records}) == 60
    for record, result in zip(records, calls):
        assert record["inputs"] == [result]
        assert record["T_host_us"] == 2.0
        assert "T_us" not in record
        assert record["host_measurement"]["kind"] == "cuda_dispatch_host_metadata"
        assert record["host_measurement"]["synchronized_cuda"] is False
        assert record["host_measurement"]["materializes_output"] is False


@pytest.mark.parametrize("latency", [0, -1, float("nan"), float("inf")])
def test_rejects_invalid_host_measurement(latency):
    class Timer:
        def __init__(self, **kwargs):
            pass

        def blocked_autorange(self, **kwargs):
            return SimpleNamespace(
                median=latency, number_per_run=1, raw_times=[latency]
            )

    with pytest.raises(ValueError, match="Invalid native host latency"):
        next(
            host.collect_records(
                host.load_recipe(), Timer, threads=1, minimum=0.05, device="cpu"
            )
        )


def test_rejects_non_native_callable():
    recipe = host.load_recipe()
    recipe.native = lambda: lambda x: x
    with pytest.raises(ValueError, match="Only the original native"):
        next(host.collect_records(recipe, None, threads=1, minimum=0.05, device="cpu"))
