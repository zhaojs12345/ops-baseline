"""Exact document layouts are retained without loading SGLang or a GPU."""

import os
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "ops/sglang_qwen36"))
import _qwen36  # noqa: E402


def test_documented_case_inventory_and_no_key_collisions():
    if not os.environ.get("FLAGGEMS_SGLANG_SOURCE"):
        pytest.skip("set FLAGGEMS_SGLANG_SOURCE for paired source validation")
    expected = {
        "gemma_rmsnorm": 12,
        "gemma_fused_add_rmsnorm": 4,
        "silu_and_mul_out": 4,
        "select_experts": 2,
        "_fused_moe_kernel_sequence": 2,
        "chunk_gated_delta_rule_fwd": 2,
        "fused_recurrent_gated_delta_rule_packed_decode": 2,
    }
    for op, count in expected.items():
        rows = _qwen36.grid(op)
        assert len(rows) == count
        assert len({_qwen36.key_shape(op, row) for row in rows}) == count
    for model, stride, offset in [("27b", 3584, 3072), ("35b", 2560, 2048)]:
        row = next(
            r
            for r in _qwen36.grid("gemma_rmsnorm")
            if r["model"] == model and r["role"] == "k" and r["tokens"] == 20
        )
        _, values = _qwen36.inputs("gemma_rmsnorm", row, torch.bfloat16, "meta")
        assert values["input"].stride() == (stride, 1)
        assert values["input"].storage_offset() == offset


def test_native_never_substitutes_a_torch_reference(monkeypatch):
    monkeypatch.delenv("FLAGGEMS_SGLANG_SOURCE", raising=False)
    with pytest.raises(RuntimeError, match="FLAGGEMS_SGLANG_SOURCE"):
        _qwen36.native("gemma_rmsnorm")


def test_profiles_are_discoverable_by_the_real_collector():
    import importlib.util

    pytest.importorskip("triton", reason="collector requires the Linux runtime")
    # Load the real host collector rather than reproducing its module contract.
    spec = importlib.util.spec_from_file_location(
        "qwen36_collector", ROOT / "tools/collect_baseline_nvidia.py"
    )
    collector = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(ROOT / "tools"))
    spec.loader.exec_module(collector)
    found = dict(collector._discover_ops(ROOT / "ops/sglang_qwen36"))
    assert len(found) == 7
    assert all(callable(module.native) for module in found.values())
    assert all(module.BENCH_RETURN_MODE == "median" for module in found.values())
