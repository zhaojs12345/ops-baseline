"""Exact hidden four-output ATen solve; PPU core dtype subset, not solve_ex.

Source: FlagGems 2a3a455fa9eef530dc11a6b96ff333ffb47235fd,
benchmark/test__linalg_solve_ex.py. Nine descriptors, two left modes, both
original PPU-supported dtypes. No RNG seed/cap/precision change is introduced.
"""

import torch

OP_NAME = "_linalg_solve_ex"
DTYPES = [torch.float32, torch.complex64]
IS_INPLACE = False
_SOURCE = "2a3a455fa9eef530dc11a6b96ff333ffb47235fd"
_DESCRIPTORS = [
    (64, 64),
    (256, 256),
    (1024, 1024),
    (4096, 512),
    (16, 256, 128),
    (8, 4, 128, 64),
    (0, 0),
    (64, 0),
    (2, 0, 0),
]


def native():
    return torch.ops.aten._linalg_solve_ex.default


def grid():
    return [
        {"descriptor": list(shape), "left": left}
        for shape in _DESCRIPTORS
        for left in (True, False)
    ]


def _operand_shapes(binding):
    shape = binding["descriptor"]
    batch, n, m = tuple(shape[:-2]), shape[-2], shape[-1]
    return batch + (n, n), (batch + (n, m) if binding["left"] else batch + (m, n))


def build_inputs(binding, dtype, device):
    a_shape, b_shape = _operand_shapes(binding)
    a = torch.randn(a_shape, dtype=dtype, device=device)
    a.diagonal(dim1=-2, dim2=-1).add_(a_shape[-1])
    b = torch.randn(b_shape, dtype=dtype, device=device)
    return (a, b), {"left": binding["left"]}


def key_shape(binding):
    return {"descriptor": list(binding["descriptor"]), "left": binding["left"]}


def _tensor(shape, dtype):
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(shape[i + 1], 1)
    return {
        "shape": list(shape),
        "stride": stride,
        "storage_offset": 0,
        "dtype": str(dtype),
        "requires_grad": False,
    }


def config(binding, dtype):
    a, b = _operand_shapes(binding)
    return {
        "native_expression": "torch.ops.aten._linalg_solve_ex.default",
        "native_schema": (
            "aten::_linalg_solve_ex(Tensor A, Tensor B, *, bool left=True, "
            "bool check_errors=False) -> (Tensor result, Tensor LU, Tensor pivots, Tensor info)"
        ),
        "source_commit": _SOURCE,
        "source_benchmark_sha256": "290a441e2eecd795d1d5212d74d5174746968e16a92b284f72c4c71f29233ce1",
        "source_builder_ast_sha256": "674a644e5239cca08107c1e588ab60f1777a3432c4e4a27581a563ca473b1118",
        "benchmark": "benchmark/test__linalg_solve_ex.py",
        "builder": "_build_inputs_fn",
        "scope": "core",
        "descriptor": list(binding["descriptor"]),
        "variant": 0 if binding["left"] else 1,
        "recipe": {
            "A": "torch.randn; diagonal add n",
            "B": "torch.randn",
            "on_requested_device": True,
            "seed": "original unseeded recipe",
        },
        "inputs": {
            "args": [_tensor(a, dtype), _tensor(b, dtype)],
            "kwargs": {"left": binding["left"]},
        },
        "outputs": [
            {"name": "result", "shape": list(b), "dtype": str(dtype)},
            {"name": "LU", "shape": list(a), "dtype": str(dtype)},
            {"name": "pivots", "shape": list(a[:-1]), "dtype": "torch.int32"},
            {"name": "info", "shape": list(a[:-2]), "dtype": "torch.int32"},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }
