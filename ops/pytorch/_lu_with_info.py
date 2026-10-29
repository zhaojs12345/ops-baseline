"""Original PPU deprecated three-output LU core, including exact pivot/check flags."""

import torch

OP_NAME = "_lu_with_info"
DTYPES = [torch.float32]
IS_INPLACE = False
_SHAPES = [(64, 64), (256, 256), (512, 512), (1024, 1024), (8, 128, 128)]


def native():
    return torch._lu_with_info


def grid():
    return _SHAPES


def build_inputs(shape, dtype, device):
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    return (inp,), {"pivot": True, "check_errors": False}


def key_shape(shape):
    return list(shape)


def config(shape, dtype):
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    return {
        "native_expression": "torch._lu_with_info",
        "native_schema": (
            "aten::_lu_with_info(Tensor self, bool pivot=True, bool check_errors=True) "
            "-> (Tensor LU, Tensor pivots, Tensor info)"
        ),
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "9d3c5f655d248422d67371653b93d0d8d64c08f3570ca91cce29f190f8f7be49",
        "source_builder_ast_sha256": "62e04bcce6bc7da3aa44e0fe9f7664402d68457fff63fca0bc2490f0dedae2d5",
        "benchmark": "benchmark/test_lu_with_info.py",
        "builder": "LuWithInfoBenchmark.build_inputs",
        "scope": "core",
        "descriptor": list(shape),
        "recipe": {"seed": 0, "input": "torch.randn on requested device"},
        "inputs": {
            "args": [
                {
                    "shape": list(shape),
                    "stride": stride,
                    "storage_offset": 0,
                    "dtype": str(dtype),
                    "requires_grad": False,
                }
            ],
            "kwargs": {"pivot": True, "check_errors": False},
        },
        "outputs": [
            {"name": "LU", "shape": list(shape), "dtype": str(dtype)},
            {
                "name": "pivots",
                "shape": list(shape[:-2]) + [min(shape[-2:])],
                "dtype": "torch.int32",
            },
            {"name": "info", "shape": list(shape[:-2]), "dtype": "torch.int32"},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }
