"""Original five float32 masked-scale core recipes; actual ATen CUDA native API.

False-masked NaN/Inf follows native multiplication, not a general where oracle.
The original finite core domain is retained with identical input/mask/scale ABI.
"""

import torch

OP_NAME = "_masked_scale"
DTYPES = [torch.float32]
IS_INPLACE = False
_SHAPES = [(1073741824,), (64, 64), (4096, 4096), (64, 512, 512), (1024, 1024, 1024)]


def native():
    return torch.ops.aten._masked_scale.default


def grid():
    return [{"shape": list(shape), "scale": 2.0} for shape in _SHAPES]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=device)
    return (inp, mask, binding["scale"]), {}


def key_shape(binding):
    return {"shape": list(binding["shape"]), "scale": binding["scale"]}


def config(binding, dtype):
    shape = list(binding["shape"])
    stride = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        stride[i] = stride[i + 1] * max(shape[i + 1], 1)

    def tensor(kind):
        return {
            "shape": shape,
            "stride": stride,
            "storage_offset": 0,
            "dtype": kind,
            "requires_grad": False,
        }

    return {
        "native_expression": "torch.ops.aten._masked_scale.default",
        "native_schema": "aten::_masked_scale(Tensor self, Tensor mask, float scale) -> Tensor",
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "b6227b7d29e0e871ec1d8a63b88413e92dbe42bd662bb5e80c6ee0c64eccba62",
        "source_builder_ast_sha256": "1e3dafe4b321481ab866cceac636bb52a0fcf0dc76d77d119fcb7b7395eaea2b",
        "source_tensor_factory_sha256": "2fa295e0f5d02e000d9e30967354cc663011de77366dd469feb8ddee7395a4a5",
        "benchmark": "benchmark/test_masked_scale.py",
        "builder": "MaskedScaleBenchmark.build_inputs",
        "scope": "core",
        "descriptor": shape,
        "variant": 0,
        "recipe": {
            "input": "torch.randn",
            "mask": "torch.randint(0, 2, dtype=torch.uint8)",
            "seed": 0,
            "on_requested_device": True,
        },
        "inputs": {
            "args": [tensor(str(dtype)), tensor("torch.uint8"), binding["scale"]],
            "kwargs": {},
        },
        "outputs": [{"name": "out", "shape": shape, "dtype": str(dtype)}],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }
