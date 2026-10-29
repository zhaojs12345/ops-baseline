# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Literal native CUDA activation: complete original shared+standalone core.

Legacy bias is two-dimensional; standalone bias is one-dimensional and carries
its original GELU and matrix-layout options. Descriptor b is retained even
where the original builder does not read it. No padding or shape reduction.
"""

import torch

OP_NAME = "addmm_activation_out_current_core"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
BENCH_RETURN_MODE = "median"
_SHAPES = [
    (2, 384, 384, 384),
    (2, 4096, 4096, 4096),
    (16, 1024, 1024, 1024),
    (16, 2048, 2048, 2048),
    (16, 4096, 4096, 4096),
]
_SOURCE_COMMIT = "9c280800abb30b0fa3e1c073d564fc4b6d97d1cd"
_SOURCE = {
    "legacy": {
        "file": "benchmark/test__addmm_activation.py",
        "sha256": "4cde204cf9e3058a730b4f5d1edadb09e849e3c4ad5edc62e4f69dee8a2267e8",
        "builder_ast_sha256": "165968e3d60a01991b7cf422a01ca5e8bd8f5744a1c02915468e7c08d964028d",
    },
    "standalone": {
        "file": "benchmark/test_addmm_activation_out.py",
        "sha256": "850c59f768a851dda3d9f04cd7ab9de55e5a0e9de18b9cad82b4a27f892d9084",
        "builder_ast_sha256": "53ab49740d9a1745d3228c21bfc6fafb32f2b193b7165218fdf8e29e5a8e87ad",
    },
}


def native():
    return torch.ops.aten._addmm_activation.out


def grid():
    bindings = [
        dict(
            origin="legacy",
            descriptor=list(shape),
            ordinal=i,
            column_major=False,
            use_gelu=False,
        )
        for i, shape in enumerate(_SHAPES)
    ]
    for i, shape in enumerate(_SHAPES):
        for use_gelu in (False, True):
            for column_major in (False, True):
                bindings.append(
                    dict(
                        origin="standalone",
                        descriptor=list(shape),
                        ordinal=i * 4 + int(use_gelu) * 2 + int(column_major),
                        column_major=column_major,
                        use_gelu=use_gelu,
                    )
                )
    return bindings


def key_shape(binding):
    return [
        binding["origin"],
        *binding["descriptor"],
        int(binding["column_major"]),
        int(binding["use_gelu"]),
    ]


def build_inputs(binding, dtype, device):
    b, m, n, k = binding["descriptor"]
    del b
    if binding["origin"] == "legacy":
        mat1 = torch.randn((m, k), dtype=dtype, device=device)
        bias = torch.randn((m, n), dtype=dtype, device=device)
        out = torch.empty((m, n), dtype=dtype, device=device)
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
        return (bias, mat1, mat2), {"out": out}
    assert binding["origin"] == "standalone"
    torch.manual_seed(0)
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if binding["column_major"]:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    bias = torch.randn((n,), dtype=dtype, device=device)
    out = torch.empty((m, n), dtype=dtype, device=device)
    return (bias, mat1, mat2), {"use_gelu": binding["use_gelu"], "out": out}


def config(binding, dtype):
    legacy = binding["origin"] == "legacy"
    b, m, n, k = binding["descriptor"]
    source = _SOURCE["legacy" if legacy else "standalone"]
    file = source["file"]
    function = "test_addmm_activation_out"
    case_id = (
        file
        + "::"
        + function
        + "::core::"
        + str(dtype).split(".")[-1]
        + "::"
        + str(binding["ordinal"])
    )

    def tensor(shape, stride):
        return dict(
            shape=shape,
            dtype=str(dtype),
            stride=stride,
            storage_offset=0,
            requires_grad=False,
        )

    args = [
        tensor([m, n] if legacy else [n], [n, 1] if legacy else [1]),
        tensor([m, k], [k, 1]),
        tensor([k, n], [1, k] if binding["column_major"] else [n, 1]),
    ]
    kwargs = dict() if legacy else dict(use_gelu=binding["use_gelu"])
    kwargs["out"] = tensor([m, n], [n, 1])
    return dict(
        native_expression="torch.ops.aten._addmm_activation.out",
        literal_native_API=True,
        source_commit=_SOURCE_COMMIT,
        source_benchmark_sha256=source["sha256"],
        source_builder_ast_sha256=source["builder_ast_sha256"],
        original_case_id=case_id,
        benchmark=file,
        function=function,
        scope="core",
        origin=binding["origin"],
        descriptor=list(binding["descriptor"]),
        ordinal=binding["ordinal"],
        original_shape=(
            dict(b=b, m=m, n=n, k=k)
            if legacy
            else dict(self=[n], mat1=[m, k], mat2=[k, n], out=[m, n])
        ),
        original_params=(
            dict(b_column_major=False)
            if legacy
            else dict(
                column_major=binding["column_major"], use_gelu=binding["use_gelu"]
            )
        ),
        recipe=dict(
            generator="torch.randn",
            seed=None if legacy else 0,
            allocation_order=(
                ["mat1", "bias", "out", "mat2"]
                if legacy
                else ["mat1", "mat2", "bias", "out"]
            ),
            column_major=binding["column_major"],
            bias_rank=2 if legacy else 1,
            use_gelu_keyword_absent=legacy,
            original_dtype=str(dtype),
        ),
        inputs=dict(args=args, kwargs=kwargs),
        output=tensor([m, n], [n, 1]),
        effects=dict(mutates=["kwargs.out"], returns_alias_of={"return": "kwargs.out"}),
    )
