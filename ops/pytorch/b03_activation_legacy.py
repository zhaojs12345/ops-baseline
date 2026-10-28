"""Exact ac9 core recipe; benchmark module isolated by overload and generation."""
import torch

def _input_fn(b, m, n, k, dtype, device, b_column_major):
    inp1 = torch.randn([m, k], dtype=dtype, device=device)
    bias = torch.randn([m, n], dtype=dtype, device=device)
    if b_column_major:
        inp2 = torch.randn([n, k], dtype=dtype, device=device)
        yield bias, inp1, inp2.t(),
    else:
        inp2 = torch.randn([k, n], dtype=dtype, device=device)
        yield bias, inp1, inp2,

from types import SimpleNamespace
OP_NAME = 'b03_activation_legacy'
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
def native():
    return torch._addmm_activation
def grid():
    return [[2, 384, 384, 384], [2, 4096, 4096, 4096], [16, 1024, 1024, 1024], [16, 2048, 2048, 2048], [16, 4096, 4096, 4096]]
def build_inputs(binding, dtype, device):
    return tuple(next(_input_fn(*binding, dtype, device, False))), {}
def key_shape(binding):
    return binding
def config(binding, dtype):
    b, m, n, k = binding[:4]
    column = False
    gelu = False
    shapes = {"self": [m,n] if True else [n], "mat1": [m,k], "mat2": [k,n]}
    strides = {"self": [n,1] if True else [1], "mat1": [k,1], "mat2": ([1,k]) if column else ([n,1])}
    if False:
        shapes['out'] = [m,n]
        strides['out'] = [n,1]
    return {"binding": binding, "dtype": str(dtype), "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0", "source_file": 'benchmark/test__addmm_activation.py', "builder": '_input_fn', "native_callable": 'torch._addmm_activation', "seed_policy": 'ambient RNG; no reset; official collector preflight consumes RNG; historical random values not preserved', "column_major": column, "use_gelu": gelu, "alpha": 1, "beta": 1, "tensor_shapes": shapes, "tensor_strides": strides, "requires_grad": False, "out_reads_old_value": False}
