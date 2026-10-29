# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Four current standalone native backward cases; legacy five remain separate."""
import torch

OP_NAME = "weight_norm_backward_current_extra_core"
DTYPES = [torch.float32]
IS_INPLACE = False
SOURCE_COMMIT = "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
CASES = [((4096, 256), 0), ((4096, 256), 1), ((4, 256, 3), 0), ((4, 256, 3), 2)]


def native():
    return torch.ops.aten._weight_norm_interface_backward.default


def grid():
    return [dict(shape=list(shape), dim=dim, ordinal=i) for i, (shape, dim) in enumerate(CASES)]


def key_shape(binding):
    return [*binding["shape"], "dim", binding["dim"]]


def build_inputs(binding, dtype, device):
    shape, dim = tuple(binding["shape"]), binding["dim"]
    torch.manual_seed(0)
    grad_w = torch.randn(shape, dtype=dtype, device=device)
    saved_v = torch.randn(shape, dtype=dtype, device=device)
    saved_g = torch.randn((shape[dim],), dtype=dtype, device=device)
    reduction_dims = tuple(range(1, len(shape))) if dim == 0 else tuple(range(len(shape) - 1))
    saved_norms = torch.sqrt(torch.sum(saved_v.float() * saved_v.float(), dim=reduction_dims))
    return (grad_w, saved_v, saved_g, saved_norms, dim), {}


def config(binding, dtype):
    args, kwargs = build_inputs(binding, dtype, "meta")
    shape, dim, ordinal = binding["shape"], binding["dim"], binding["ordinal"]
    inputs = [dict(shape=list(t.shape), dtype=str(t.dtype), stride=list(t.stride()), storage_offset=t.storage_offset(), requires_grad=t.requires_grad) for t in args[:4]]
    inputs.append(dim)
    return dict(
        source_commit=SOURCE_COMMIT,
        benchmark="benchmark/test_weight_norm_interface_backward.py",
        original_case_id="benchmark/test_weight_norm_interface_backward.py::test_weight_norm_interface_backward::core::float32::" + str(ordinal),
        original_shape=dict(grad_w=list(shape), saved_v=list(shape)),
        original_params=dict(dim=dim),
        descriptor=[list(shape), dim],
        ordinal=ordinal,
        scope="core",
        native_expression="torch.ops.aten._weight_norm_interface_backward.default",
        inputs=dict(args=inputs, kwargs=kwargs),
        input_recipe=dict(seed=0, allocation_order=["grad_w", "saved_v", "saved_g", "saved_norms"], saved_norms="sqrt(sum(saved_v.float()*saved_v.float(), original_reduction_dims))", reduction_dims=list(range(1, len(shape))) if dim == 0 else list(range(len(shape) - 1))),
        outputs=[dict(shape=list(args[1].shape), dtype=str(dtype)), dict(shape=list(args[2].shape), dtype=str(dtype))],
    )
