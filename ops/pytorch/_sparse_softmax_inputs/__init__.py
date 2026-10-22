# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical COO input codec; input preparation only, never the timed operator.

Copy this interface into each consuming FlagGems OWN benchmark, rather than
importing this repository at target runtime. Bit equality is established by the
recorded input identity, not by a blanket NumPy/platform portability claim.
"""

import hashlib
import json

import numpy as np
import torch

RECIPE = "coo-softmax-pcg64-native-width-compact-v1"
DTYPES = (torch.float32, torch.float64)


def _validate(shape, nnz, sparse_dim, dim, dtype):
    shape = tuple(shape)
    if not shape or any(type(n) is not int or n <= 0 for n in shape):
        raise ValueError("Expected a nonempty positive integral shape")
    if type(nnz) is not int or nnz < 0:
        raise ValueError("Expected nonnegative integral nnz")
    if type(sparse_dim) is not int or not 0 <= sparse_dim <= len(shape):
        raise ValueError("Invalid sparse_dim")
    if type(dim) is not int or not 0 <= dim < len(shape):
        raise ValueError("Invalid normalization dim")
    if dtype not in DTYPES:
        raise TypeError("Only original float32/float64 inputs are supported")
    return shape


def seed_hex(kind, shape, nnz, sparse_dim, dim, dtype):
    payload = json.dumps(
        {
            "recipe": RECIPE,
            "kind": kind,
            "shape": list(shape),
            "nnz": nnz,
            "sparse_dim": sparse_dim,
            "dim": dim,
            "dtype": str(dtype),
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    return hashlib.sha256(payload).digest()[:8].hex()


def _draw(kind, shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    seed = bytes.fromhex(seed_hex(kind, shape, nnz, sparse_dim, dim, dtype))
    rng = np.random.Generator(np.random.PCG64(int.from_bytes(seed, "little")))
    # Same per-axis replacement distribution/range and draw order as the source.
    indices = np.empty((sparse_dim, nnz), dtype=np.int64)
    for axis, extent in enumerate(shape[:sparse_dim]):
        indices[axis] = rng.integers(0, extent, size=nnz, dtype=np.int64)
    # Do not generate float64 through float32: retain genuine requested width.
    npdtype = np.float64 if dtype == torch.float64 else np.float32
    value_shape = (nnz,) + shape[sparse_dim:]
    return shape, rng, indices, value_shape, npdtype


def _compact_cpu(array, dtype):
    result = torch.empty(tuple(array.shape), dtype=dtype, device="cpu")
    result.copy_(torch.from_numpy(array))
    return result


def _canonical_coalesce(indices, values, shape):
    """Stable lexicographic support and typed accumulation of replacement draws.

    This is a NEW fixture recipe, not reproduction of old Torch RNG payloads.
    Stored zero sums are retained. No dense shape-sized allocation is made.
    """
    if indices.shape[1] == 0:
        return indices.copy(), values.copy()
    keys = np.ravel_multi_index(tuple(indices), shape)
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    starts = np.r_[True, ordered[1:] != ordered[:-1]]
    groups = np.cumsum(starts, dtype=np.int64) - 1
    result = np.zeros(int(groups[-1]) + 1, dtype=values.dtype)
    # Unbuffered repeated-index addition preserves original duplicate order.
    np.add.at(result, groups, values[order])
    return indices[:, order[starts]].copy(), result


def forward_cpu(shape, nnz, dim, dtype):
    shape, rng, indices, value_shape, npdtype = _draw(
        "forward", shape, nnz, len(shape), dim, dtype
    )
    values = rng.standard_normal(value_shape, dtype=npdtype)
    indices, values = _canonical_coalesce(indices, values, shape)
    return torch.sparse_coo_tensor(
        _compact_cpu(indices, torch.int64),
        _compact_cpu(values, dtype),
        size=shape,
        dtype=dtype,
        device="cpu",
        is_coalesced=True,
    )


def backward_cpu(shape, nnz, sparse_dim, dim, dtype):
    shape, rng, indices, value_shape, npdtype = _draw(
        "backward", shape, nnz, sparse_dim, dim, dtype
    )
    grad = torch.sparse_coo_tensor(
        _compact_cpu(indices, torch.int64),
        _compact_cpu(rng.standard_normal(value_shape, dtype=npdtype), dtype),
        size=shape,
        dtype=dtype,
        device="cpu",
    )
    # Independent indices allocation, and independent normal values, as upstream.
    output = torch.sparse_coo_tensor(
        _compact_cpu(indices.copy(), torch.int64),
        _compact_cpu(rng.standard_normal(value_shape, dtype=npdtype), dtype),
        size=shape,
        dtype=dtype,
        device="cpu",
    )
    self_tensor = _compact_cpu(rng.standard_normal((1,), dtype=npdtype), dtype)
    return grad, output, self_tensor


def _contiguous_stride(shape):
    stride, result = 1, []
    for extent in reversed(shape):
        result.append(stride)
        stride *= max(extent, 1)
    return tuple(reversed(result))


def _part_identity(part):
    if not isinstance(part, torch.Tensor) or part.layout != torch.strided:
        raise TypeError("Expected a dense component Tensor")
    extent = part.numel() * part.element_size()
    # The canonical contract is compact, so every hashed storage byte is defined.
    if (
        part.storage_offset() != 0
        or part.untyped_storage().nbytes() != extent
        or tuple(part.stride()) != _contiguous_stride(part.shape)
        or part.is_conj()
        or part.is_neg()
    ):
        raise ValueError("Noncanonical component storage geometry or lazy flags")
    raw = (
        torch.empty(0, dtype=torch.uint8, device=part.device)
        .set_(part.untyped_storage(), 0, (extent,), (1,))
        .cpu()
    )
    if raw.dtype != torch.uint8 or raw.device.type != "cpu":
        raise ValueError("Byte observation changed its ABI")
    return {
        "dtype": str(part.dtype),
        "shape": list(part.shape),
        "stride": list(part.stride()),
        "storage_offset": part.storage_offset(),
        "storage_nbytes": extent,
        "is_conj": part.is_conj(),
        "is_neg": part.is_neg(),
        "sha256": hashlib.sha256(raw.numpy().tobytes()).hexdigest(),
    }


def _tensor_identity(value):
    if not isinstance(value, torch.Tensor):
        raise TypeError("Expected an input Tensor")
    if value.layout != torch.sparse_coo:
        return {
            "layout": str(value.layout),
            "requires_grad": value.requires_grad,
            "part": _part_identity(value),
        }
    return {
        "layout": str(value.layout),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sparse_dim": value.sparse_dim(),
        "dense_dim": value.dense_dim(),
        "nnz": value._nnz(),
        "is_coalesced": value.is_coalesced(),
        "requires_grad": value.requires_grad,
        "is_conj": value.is_conj(),
        "is_neg": value.is_neg(),
        "indices": _part_identity(value._indices()),
        "values": _part_identity(value._values()),
    }


def input_identity(named_inputs):
    """Complete defined input content/geometry plus every component alias pair.

    Absolute pointers/devices are not cross-machine identities. build transfer
    checks the exact requested physical device separately; config uses target role.
    """
    components = []
    tensors = {}
    for name, value in named_inputs.items():
        tensors[name] = _tensor_identity(value)
        parts = (
            [value._indices(), value._values()]
            if value.layout == torch.sparse_coo
            else [value]
        )
        labels = (
            ["indices", "values"] if value.layout == torch.sparse_coo else ["tensor"]
        )
        components.extend(
            (name + "." + label, part) for label, part in zip(labels, parts)
        )
    aliases = {}
    for i, (left, first) in enumerate(components):
        for right, second in components[i + 1 :]:
            aliases[left + "|" + right] = (
                first.untyped_storage()._cdata == second.untyped_storage()._cdata
            )
    return {"tensors": tensors, "component_storage_aliases": aliases}


def to_device(named_inputs, device):
    """Byte-preserving input transport only; native callable is resolved elsewhere."""
    expected = torch.empty(0, dtype=torch.uint8, device=device).device
    before = input_identity(named_inputs)

    def part_on_target(part):
        # All components are compact and physically defined. Independent copies
        # avoid typed F64 conversion/narrowing and preserve storage independence.
        raw = (
            torch.empty(0, dtype=torch.uint8, device="cpu")
            .set_(
                part.untyped_storage(), 0, (part.numel() * part.element_size(),), (1,)
            )
            .to(device=expected, copy=True)
        )
        result = raw.view(part.dtype).reshape(part.shape)
        if (
            result.device != expected
            or result.dtype != part.dtype
            or result.shape != part.shape
        ):
            raise ValueError("Input transport changed requested device/dtype/shape")
        return result

    result = {}
    for name, value in named_inputs.items():
        if value.layout == torch.sparse_coo:
            output = torch.sparse_coo_tensor(
                part_on_target(value._indices()),
                part_on_target(value._values()),
                size=value.shape,
                dtype=value.dtype,
                device=expected,
                is_coalesced=value.is_coalesced(),
            )
        else:
            output = part_on_target(value)
        if (
            output.device != expected
            or output.dtype != value.dtype
            or output.shape != value.shape
        ):
            raise ValueError("Input construction changed required ABI")
        result[name] = output
    if input_identity(result) != before:
        raise ValueError("Input transport changed payload, geometry, flags or aliases")
    return result
