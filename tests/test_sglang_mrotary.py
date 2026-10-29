"""Source integration for the explicit native mRoPE recipe (no accelerator)."""

import importlib.util
import os
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


def test_six_distinct_source_workloads():
    if not os.environ.get("FLAGGEMS_SGLANG_SOURCE"):
        pytest.skip("set FLAGGEMS_SGLANG_SOURCE for source integration")
    spec = importlib.util.spec_from_file_location(
        "mrotary_recipe", ROOT / "ops/sglang/mrotary_embedding.py"
    )
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    rows = recipe.grid()
    assert len(rows) == 6
    assert len({recipe.key_shape(row) for row in rows}) == 6
    for row in rows:
        _, inputs = recipe.build_inputs(row, torch.bfloat16, "meta")
        cfg = recipe.config(row, torch.bfloat16)
        assert (
            list(inputs["positions"].stride()) == cfg["inputs"]["positions"]["stride"]
        )
        assert (
            cfg["source"]["upstream_blob"] == "0a8dc2c33c7bbb2acb7e02b10796d210c5c079fc"
        )


def test_missing_source_is_explicit(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_SGLANG_SOURCE", raising=False)
    spec = importlib.util.spec_from_file_location(
        "mrotary_recipe_missing", ROOT / "ops/sglang/mrotary_embedding.py"
    )
    recipe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(recipe)
    with pytest.raises(RuntimeError, match="FLAGGEMS_SGLANG_SOURCE"):
        recipe.native()


@pytest.mark.parametrize("mode", [None, "median"])
def test_collector_retains_default_and_records_opt_in_statistic(mode, monkeypatch):
    import sys
    from types import SimpleNamespace
    from statistics import mean, median

    sys.path.insert(0, str(ROOT / "tools"))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    import collect_baseline_nvidia as collector

    calls = []

    def bench(fn, **kwargs):
        fn()
        calls.append(kwargs)
        return (median if kwargs.get("return_mode") == "median" else mean)(
            [1.0, 1.0, 100.0]
        )

    monkeypatch.setattr(
        collector,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None)),
    )
    monkeypatch.setattr(
        collector, "triton", SimpleNamespace(testing=SimpleNamespace(do_bench=bench))
    )
    recipe = SimpleNamespace(
        OP_NAME="test_statistic",
        __name__="fixture",
        native=lambda: lambda: None,
        DTYPES=["test_dtype"],
        grid=lambda: [{}],
        build_inputs=lambda *a: ((), {}),
        key_shape=lambda b: "shape",
    )
    if mode:
        recipe.BENCH_RETURN_MODE = mode
    result = collector._collect_one_op_ops(recipe, False, None)
    record = result["shapes"]["shape"]["test_dtype"]
    assert record["latency_ms"] == (1.0 if mode else 34.0)
    if mode:
        assert record["timing"]["return_mode"] == "median"
    else:
        assert "return_mode" not in calls[0] and "timing" not in record
