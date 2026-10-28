"""Exact ac9 core recipe; benchmark module isolated by overload and generation."""
import torch

def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major, use_gelu = plan.builder_args
    torch.manual_seed(0)
    mat1 = torch.randn((m, k), dtype=dtype, device=device)
    if column_major:
        mat2 = torch.randn((n, k), dtype=dtype, device=device).t()
    else:
        mat2 = torch.randn((k, n), dtype=dtype, device=device)
    self = torch.randn((n,) if True else (m, n), dtype=dtype, device=device)
    out = torch.empty((m, n), dtype=dtype, device=device)
    return self, mat1, mat2, {"use_gelu": use_gelu, "out": out}

from types import SimpleNamespace
OP_NAME = 'b03_activation_out'
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True
def native():
    return torch.ops.aten._addmm_activation.out
def grid():
    return [[2, 384, 384, 384, False, False], [2, 384, 384, 384, True, False], [2, 384, 384, 384, False, True], [2, 384, 384, 384, True, True], [2, 4096, 4096, 4096, False, False], [2, 4096, 4096, 4096, True, False], [2, 4096, 4096, 4096, False, True], [2, 4096, 4096, 4096, True, True], [16, 1024, 1024, 1024, False, False], [16, 1024, 1024, 1024, True, False], [16, 1024, 1024, 1024, False, True], [16, 1024, 1024, 1024, True, True], [16, 2048, 2048, 2048, False, False], [16, 2048, 2048, 2048, True, False], [16, 2048, 2048, 2048, False, True], [16, 2048, 2048, 2048, True, True], [16, 4096, 4096, 4096, False, False], [16, 4096, 4096, 4096, True, False], [16, 4096, 4096, 4096, False, True], [16, 4096, 4096, 4096, True, True]]
def build_inputs(binding, dtype, device):
    values = _build_inputs_fn(SimpleNamespace(builder_args=binding), dtype, device)
    return (tuple(values[:-1]), values[-1]) if isinstance(values[-1], dict) else (tuple(values), {})
def key_shape(binding):
    return binding
def config(binding, dtype):
    b, m, n, k = binding[:4]
    column = binding[4]
    gelu = binding[5]
    shapes = {"self": [m,n] if False else [n], "mat1": [m,k], "mat2": [k,n]}
    strides = {"self": [n,1] if False else [1], "mat1": [k,1], "mat2": ([1,k]) if column else ([n,1])}
    if True:
        shapes['out'] = [m,n]
        strides['out'] = [n,1]
    return {"binding": binding, "dtype": str(dtype), "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0", "source_file": 'benchmark/test_addmm_activation_out.py', "builder": '_build_inputs_fn', "native_callable": 'torch.ops.aten._addmm_activation.out', "seed_policy": 'original builder torch.manual_seed(0) each call', "column_major": column, "use_gelu": gelu, "alpha": 1, "beta": 1, "tensor_shapes": shapes, "tensor_strides": strides, "requires_grad": False, "out_reads_old_value": False}
