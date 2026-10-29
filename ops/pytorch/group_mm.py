# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Real CUDA per-group mm + cat for the exact original group_mm core recipe.

Not the literal grouped kernel. Actual offsets D2H/Python/mm/cat are all inside
this timed callable. No padding, alignment workaround or captured fixed offsets.
"""

import torch

OP_NAME = "group_mm"
DTYPES = [torch.bfloat16]
IS_INPLACE = False
BENCH_RETURN_MODE = "median"
_SHAPES = [(16, 512, 2048), (16, 2560, 2048), (64, 2048, 128)]


def native_composite(mat_a, mat_b, offs):
    assert mat_a.ndim == 2 and mat_b.ndim == 3 and offs.ndim == 1
    assert mat_a.dtype == mat_b.dtype == torch.bfloat16 and offs.dtype == torch.int32
    assert mat_a.device == mat_b.device == offs.device and mat_a.device.type == "cuda"
    assert mat_a.is_contiguous() and mat_b.is_contiguous() and offs.is_contiguous()
    assert not any(t.requires_grad for t in (mat_a, mat_b, offs))
    groups, k_size, _ = mat_b.shape
    assert mat_a.shape[1] == k_size and offs.numel() == groups
    boundaries = [0] + offs.cpu().tolist()
    assert boundaries[-1] == mat_a.shape[0]
    assert all(left < right for left, right in zip(boundaries, boundaries[1:]))
    return torch.cat(
        [
            torch.mm(mat_a[boundaries[i] : boundaries[i + 1]], mat_b[i])
            for i in range(groups)
        ],
        dim=0,
    )


def native():
    return native_composite


def grid():
    return [{"shape": list(shape)} for shape in _SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def _sizes(binding):
    groups, _, _ = binding["shape"]
    return [16 + (idx * 13) % 48 for idx in range(groups)]


def build_inputs(binding, dtype, device):
    groups, n_size, k_size = binding["shape"]
    group_sizes = _sizes(binding)
    torch.manual_seed(0)
    mat_a = torch.rand((sum(group_sizes), k_size), dtype=dtype, device=device)
    mat_b = torch.rand((groups, k_size, n_size), dtype=dtype, device=device)
    offs = torch.tensor(
        [sum(group_sizes[: idx + 1]) for idx in range(groups)],
        dtype=torch.int32,
        device=device,
    )
    return (mat_a, mat_b, offs), {}


def config(binding, dtype):
    groups, n_size, k_size = binding["shape"]
    sizes = _sizes(binding)
    m_size = sum(sizes)

    def tensor(shape, tensor_dtype, stride):
        return dict(
            shape=shape,
            dtype=str(tensor_dtype),
            stride=stride,
            storage_offset=0,
            requires_grad=False,
        )

    return dict(
        native_expression="native_CUDA_per_group_mm_actual_offsets_D2H_cat",
        literal_grouped_kernel=False,
        timed_components=[
            "offs.cpu().tolist",
            "Python group loop",
            "torch.mm each original group",
            "torch.cat",
        ],
        source_commit="2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        source_benchmark_sha256="552ddc59212ff654de494630215c8958c83aebdd10bfb4bafdc411030c38a4dd",
        source_builder_ast_sha256="23cdc17663c803e4e87d8101cc68803571cb24f0a396c6939b1c0d9eceb6467d",
        benchmark="benchmark/test_grouped_mm.py",
        scope="core",
        descriptor=list(binding["shape"]),
        group_sizes=sizes,
        offsets=[sum(sizes[: i + 1]) for i in range(groups)],
        recipe=dict(
            A="torch.rand",
            B="torch.rand",
            offsets="torch.tensor cumulative endpoints",
            seed=0,
            original_dtype=str(dtype),
        ),
        inputs=dict(
            args=[
                tensor([m_size, k_size], dtype, [k_size, 1]),
                tensor([groups, k_size, n_size], dtype, [k_size * n_size, n_size, 1]),
                tensor([groups], torch.int32, [1]),
            ],
            kwargs={},
        ),
        output=tensor([m_size, n_size], dtype, [n_size, 1]),
        effects=dict(mutates=[], returns_alias_of={}),
    )
