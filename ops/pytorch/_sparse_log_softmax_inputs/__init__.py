# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Canonical sparse log-softmax INPUT codec, never a timed oracle.

Independent PCG64 normal draws are a new payload recipe. Descriptor/support/normal
semantics match the pinned original benchmarks; old Torch RNG bits are not claimed.
Copy this interface into each own FG consumer, never import the Ops checkout there.
Strict actual full input identity, not a platform portability assertion, gates replay.
"""

import hashlib
import json
import math
import numpy as np
import torch

RECIPE = "coo-log-softmax-pcg64-native-width-compact-v1"
DTYPES = (torch.float32, torch.float64)


def _validate(shape, nnz, sparse_dim, dim, dtype):
    shape = tuple(shape)
    if any(type(n) is not int or n < 0 for n in shape):
        raise ValueError("Expected nonnegative integral extents")
    if type(nnz) is not int or nnz < 0:
        raise ValueError("Invalid stored count")
    if type(sparse_dim) is not int or not 0 <= sparse_dim <= len(shape):
        raise ValueError("Invalid sparse dimension count")
    if type(dim) is not int or (
        dim != -1 if not shape else not -len(shape) <= dim < len(shape)
    ):
        raise ValueError("Invalid normalization dimension")
    if nnz and any(n == 0 for n in shape[:sparse_dim]):
        raise ValueError("Nonempty support cannot occupy a zero sparse extent")
    if dtype not in DTYPES:
        raise TypeError("Only original float32/float64 dtypes")
    return shape


def seed_hex(kind, shape, nnz, sparse_dim, dim, dtype):
    fields = {
        "recipe": RECIPE,
        "kind": kind,
        "shape": list(shape),
        "nnz": nnz,
        "sparse_dim": sparse_dim,
        "dim": dim,
        "dtype": str(dtype),
    }
    return (
        hashlib.sha256(
            json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
        )
        .digest()[:8]
        .hex()
    )


def _rng(kind, shape, nnz, sparse_dim, dim, dtype):
    _validate(shape, nnz, sparse_dim, dim, dtype)
    return np.random.Generator(
        np.random.PCG64(
            int.from_bytes(
                bytes.fromhex(seed_hex(kind, shape, nnz, sparse_dim, dim, dtype)),
                "little",
            )
        )
    )


def _compact_cpu(array, dtype):
    result = torch.empty(tuple(array.shape), dtype=dtype, device="cpu")
    result.copy_(torch.from_numpy(array))
    return result


def forward_indices(shape, sparse_dim, nnz, dim):
    """Original O(nnz) grouped/rank-sorted unique forward support, no random draw."""
    if nnz == 0 or sparse_dim == 0:
        return np.empty((sparse_dim, nnz), dtype=np.int64)
    sizes = tuple(shape[:sparse_dim])
    axis = dim + len(shape) if dim < 0 else dim
    axis = axis if 0 <= axis < sparse_dim else sparse_dim - 1
    others = [i for i in range(sparse_dim) if i != axis]
    extent = math.prod(sizes[i] for i in others)
    group = min(sizes[axis], max(2, -(-nnz // extent)))
    order = np.arange(nnz, dtype=np.int64)
    base = order // group
    coords = [None] * sparse_dim
    coords[axis] = order % group
    for i in reversed(others):
        coords[i] = base % sizes[i]
        base = base // sizes[i]
    rank = np.ravel_multi_index(tuple(coords), sizes)
    rank.sort()
    return np.stack(np.unravel_index(rank, sizes)).astype(np.int64, copy=False)


def backward_indices(shape, sparse_dim, nnz, dim):
    """Original support groups, dense-axis spread, duplicates and original order."""
    if sparse_dim == 0 or nnz == 0:
        return np.empty((sparse_dim, nnz), dtype=np.int64)
    if dim < 0:
        dim += len(shape)
    index = np.arange(nnz, dtype=np.int64)
    if dim < sparse_dim:
        groups = tuple(shape[a] for a in range(sparse_dim) if a != dim)
        n_groups = math.prod(groups) if groups else 1
        cols = max(1, min(shape[dim], nnz))
        used = max(1, -(-nnz // cols))
        step = max(1, n_groups // used)
        group_id = ((index // cols) * step) % n_groups
        parts = iter(np.unravel_index(group_id, groups) if groups else ())
        coords = [index % cols if a == dim else next(parts) for a in range(sparse_dim)]
    else:
        spread = math.prod(shape[:sparse_dim])
        step = max(1, spread // nnz)
        coords = np.unravel_index((index * step) % spread, tuple(shape[:sparse_dim]))
    return np.stack(coords).astype(np.int64, copy=False)


def _coo(indices, values, shape, dtype, coalesced=None):
    kwargs = {} if coalesced is None else {"is_coalesced": coalesced}
    return torch.sparse_coo_tensor(
        _compact_cpu(indices, torch.int64),
        _compact_cpu(values, dtype),
        size=tuple(shape),
        dtype=dtype,
        device="cpu",
        **kwargs
    )


def forward_cpu(shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    if not shape:
        raise ValueError("Original forward core excludes scalar shape")
    if nnz > math.prod(shape[:sparse_dim]):
        raise ValueError("Forward support must stay unique")
    rng = _rng("forward", shape, nnz, sparse_dim, dim, dtype)
    values = rng.standard_normal(
        (nnz,) + shape[sparse_dim:],
        dtype=np.float64 if dtype == torch.float64 else np.float32,
    )
    return _coo(
        forward_indices(shape, sparse_dim, nnz, dim), values, shape, dtype, True
    )


def backward_cpu(shape, nnz, sparse_dim, dim, dtype):
    shape = _validate(shape, nnz, sparse_dim, dim, dtype)
    rng = _rng("backward", shape, nnz, sparse_dim, dim, dtype)
    indices = backward_indices(shape, sparse_dim, nnz, dim)
    # All THREE original operands are full sparse COO tensors, independent normal
    # values and storage, equal support CONTENT. self is never a [1] dense placeholder.
    return tuple(
        _coo(
            indices.copy(),
            rng.standard_normal(
                (nnz,) + shape[sparse_dim:],
                dtype=np.float64 if dtype == torch.float64 else np.float32,
            ),
            shape,
            dtype,
        )
        for _ in range(3)
    )


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
