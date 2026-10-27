"""Native H800 baseline for the original ac9 FlagGems _masked_scale core recipe."""
import torch

OP_NAME = "masked_scale"
DTYPES = [torch.float32]
IS_INPLACE = False
_SOURCE = "ac9a1f703757537a3d54cc1ac1febeca410316c0"
_SHAPES = [(1073741824,), (64, 64), (4096, 4096), (64, 512, 512), (1024, 1024, 1024)]


def native():
    return torch.ops.aten._masked_scale.default


def grid():
    return [{"shape": list(shape), "variant": 0} for shape in _SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    torch.manual_seed(0)
    inp = torch.randn(shape, dtype=dtype, device=device)
    mask = torch.randint(0, 2, shape, dtype=torch.uint8, device=device)
    return (inp, mask, 2.0), {}


def _describe(shape, dtype):
    strides = []
    size = 1
    for dim in reversed(shape):
        strides.append(size)
        size *= dim
    return dict(shape=list(shape), dtype=str(dtype), stride=list(reversed(strides)), requires_grad=False)


def config(binding, dtype):
    shape = tuple(binding["shape"])
    return dict(
        inputs=dict(args=[_describe(shape, dtype), _describe(shape, torch.uint8), 2.0], kwargs={}),
        source_commit=_SOURCE,
        benchmark="benchmark/test_masked_scale.py",
        descriptor=list(shape), scope="core", variant=0,
        input_recipe=dict(seed=0, input="torch.randn", mask="torch.randint", mask_low=0, mask_high=2, mask_dtype="torch.uint8", draw_order=["input", "mask"], scale=2.0),
    )
