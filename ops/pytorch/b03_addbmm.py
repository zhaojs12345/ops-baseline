"""Exact ac9 core recipe; benchmark module isolated by overload and generation."""
import torch

def _build_inputs_fn(plan, dtype, device):
    b, m, n, k, column_major = plan.builder_args
    torch.manual_seed(0)
    batch1 = torch.randn((b, m, k), dtype=dtype, device=device)
    if column_major:
        batch2 = torch.randn((b, n, k), dtype=dtype, device=device).transpose(1, 2)
    else:
        batch2 = torch.randn((b, k, n), dtype=dtype, device=device)
    self = torch.randn((m, n), dtype=dtype, device=device)
    return self, batch1, batch2

from types import SimpleNamespace
OP_NAME = 'b03_addbmm'
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
def native():
    return torch.addbmm
def grid():
    return [[2, 384, 384, 384, False], [2, 384, 384, 384, True], [2, 4096, 4096, 4096, False], [2, 4096, 4096, 4096, True], [16, 1024, 1024, 1024, False], [16, 1024, 1024, 1024, True], [16, 2048, 2048, 2048, False], [16, 2048, 2048, 2048, True], [16, 4096, 4096, 4096, False], [16, 4096, 4096, 4096, True]]
def build_inputs(binding, dtype, device):
    values = _build_inputs_fn(SimpleNamespace(builder_args=binding), dtype, device)
    return (tuple(values[:-1]), values[-1]) if isinstance(values[-1], dict) else (tuple(values), {})
def key_shape(binding):
    return binding
def config(binding, dtype):
    b, m, n, k = binding[:4]
    column = binding[4]
    gelu = False
    shapes = {"self": [m,n] if True else [n], "batch1": [b,m,k], "batch2": [b,k,n]}
    strides = {"self": [n,1] if True else [1], "batch1": [m*k,k,1], "batch2": ([n*k,1,k]) if column else ([k*n,n,1])}
    if False:
        shapes['out'] = [m,n]
        strides['out'] = [n,1]
    return {"binding": binding, "dtype": str(dtype), "source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0", "source_file": 'benchmark/test_addbmm.py', "builder": '_build_inputs_fn', "native_callable": 'torch.addbmm', "seed_policy": 'original builder torch.manual_seed(0) each call', "column_major": column, "use_gelu": gelu, "alpha": 1, "beta": 1, "tensor_shapes": shapes, "tensor_strides": strides, "requires_grad": False, "out_reads_old_value": False}
