# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native H800 CSR product, matching the original FlagGems core case grid.

Case source: FlagGems 633f62d145b0c36ba374e4595dc3540a11044622,
benchmark/test__sparse_csr_prod.py. Six shapes x three dim forms x three dtypes.
The native callable is unchanged. The normal-input recipe is verified on the
task's Linux backend environments. NumPy float32 normals can differ on another
OS or CPU math stack: consumers must match the actual encoded CSR identity to
the collected config and reject differences, rather than infer it from shape.
No FlagGems or pytest dependency, cached tensors, or global RNG changes.
"""

import hashlib
import json

import numpy as np
import torch

OP_NAME = "_sparse_csr_prod"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
RECIPE = "csr-prod-pcg64-normal-f32-cast-compact-v2"
CASE_SOURCE = "633f62d145b0c36ba374e4595dc3540a11044622"
SHAPES = (
    (1024, 1024),
    (2048, 2048),
    (4096, 512),
    (512, 4096),
    (1, 65536),
    (65536, 1),
)
DIMS = ((0,), (1,), ())


def native():
    try:
        return torch.ops.aten._sparse_csr_prod.dim_dtype
    except (AttributeError, RuntimeError):
        return None


def _seed(shape, dtype):
    data = json.dumps(
        {"recipe": RECIPE, "shape": list(shape), "dtype": str(dtype)},
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return int.from_bytes(hashlib.sha256(data).digest()[:8], "little")


def _cpu_input(shape, dtype):
    if len(shape) != 2 or any(
        isinstance(d, bool) or not isinstance(d, int) or d < 0 for d in shape
    ):
        raise ValueError("Expected a nonnegative two-dimensional shape")
    if dtype not in DTYPES:
        raise TypeError("Unsupported CSR product benchmark dtype")
    rng = np.random.Generator(np.random.PCG64(_seed(shape, dtype)))
    dense = torch.from_numpy(rng.standard_normal(tuple(shape), dtype=np.float32))
    csr = dense.to(dtype).to_sparse_csr()
    # Native dense-to-CSR may share the index allocation and leave col at a
    # nonzero storage offset. Give each component its own compact allocation
    # so CPU configuration and target replay have the same physical geometry.
    return torch.sparse_csr_tensor(
        csr.crow_indices().clone(),
        csr.col_indices().clone(),
        csr.values().clone(),
        size=csr.shape,
        device="cpu",
    )


def _part_identity(part):
    # Even contiguous() may retain a singleton view's non-unit stride.
    logical = torch.empty(part.numel(), dtype=part.dtype, device="cpu")
    logical.copy_(part.detach().cpu().reshape(-1))
    raw = logical.view(torch.uint8)
    return {
        "dtype": str(part.dtype),
        "shape": list(part.shape),
        "stride": list(part.stride()),
        "storage_offset": part.storage_offset(),
        "storage_nbytes": part.untyped_storage().nbytes(),
        "is_conj": part.is_conj(),
        "is_neg": part.is_neg(),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def input_identity(inp):
    if inp.layout != torch.sparse_csr or inp.ndim != 2:
        raise ValueError("Expected a two-dimensional CSR input")
    shape = list(inp.shape)
    parts = (inp.crow_indices(), inp.col_indices(), inp.values())
    return {
        "recipe": RECIPE,
        "seed_hex": f"{_seed(shape, inp.dtype):016x}",
        "layout": str(inp.layout),
        "shape": shape,
        "dtype": str(inp.dtype),
        "nnz": inp._nnz(),
        "requires_grad": inp.requires_grad,
        "component_storage_aliases": [
            parts[i].untyped_storage()._cdata == parts[j].untyped_storage()._cdata
            for i, j in ((0, 1), (0, 2), (1, 2))
        ],
        "crow": _part_identity(inp.crow_indices()),
        "col": _part_identity(inp.col_indices()),
        "values": _part_identity(inp.values()),
    }


def grid():
    return [
        {"shape": list(shape), "dim": list(dim)} for shape in SHAPES for dim in DIMS
    ]


def build_inputs(binding, dtype, device):
    cpu = _cpu_input(binding["shape"], dtype)
    inp = torch.sparse_csr_tensor(
        cpu.crow_indices().to(device),
        cpu.col_indices().to(device),
        cpu.values().to(device),
        size=cpu.shape,
        device=device,
    )
    if inp.dtype != dtype or tuple(inp.shape) != tuple(binding["shape"]):
        raise ValueError("Input transfer changed the requested dtype or shape")
    if input_identity(inp) != input_identity(cpu):
        raise ValueError("Input transfer changed the CSR payload or storage geometry")
    return (inp,), {"dim": list(binding["dim"]), "keepdim": True}


def key_shape(binding):
    rows, cols = binding["shape"]
    dim = "_".join(map(str, binding["dim"])) or "all"
    return f"csr_{rows}x{cols}_dim_{dim}"


def config(binding, dtype):
    return {
        "native_expression": "torch.ops.aten._sparse_csr_prod.dim_dtype",
        "case_source_commit": CASE_SOURCE,
        "benchmark": "benchmark/test__sparse_csr_prod.py",
        "scope": "core",
        "dim": list(binding["dim"]),
        "keepdim": True,
        "input": input_identity(_cpu_input(binding["shape"], dtype)),
    }
