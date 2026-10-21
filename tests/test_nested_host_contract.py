"""Nested host identity must retain geometry unavailable from Tensor.shape."""

import sys
from pathlib import Path
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from host_baseline_plugin import input_contract
from pytest_host_baseline import _canonical


def test_nested_contract_preserves_ragged_layout_and_global_offsets():
    raw = torch.arange(20, dtype=torch.float32)
    nested = torch.ops.aten._nested_view_from_buffer(
        raw,
        torch.tensor([[2, 2], [3, 2]]),
        torch.tensor([[2, 1], [2, 1]]),
        torch.tensor([1, 8]),
    )
    contract = input_contract("_nested_tensor_strides", (nested,), {})["args"][0]
    assert contract["storage_elements"] == 20
    assert contract["components"] == [
        {"shape": [2, 2], "stride": [2, 1], "storage_offset": 1},
        {"shape": [3, 2], "stride": [2, 1], "storage_offset": 8},
    ]
    changed = torch.ops.aten._nested_view_from_buffer(
        raw,
        torch.tensor([[2, 2], [3, 2]]),
        torch.tensor([[1, 2], [2, 1]]),
        torch.tensor([1, 8]),
    )
    assert (
        input_contract("_nested_tensor_strides", (changed,), {})["args"][0] != contract
    )
    assert _canonical({"device": "cuda:0"}) == _canonical({"device": "npu:0"})
    assert _canonical({"device": "cpu"}) != _canonical({"device": "npu:0"})


def test_empty_nested_contract_has_no_dense_shape_access():
    nested = torch.nested.nested_tensor([], dtype=torch.float16)
    result = input_contract("_nested_tensor_storage_offsets", (nested,), {})["args"][0]
    assert result["components"] == []
    assert result["storage_elements"] == 0
    assert result["dtype"] == "torch.float16"
