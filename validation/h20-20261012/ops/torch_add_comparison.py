"""Comparison-only real CUDA workload; not a vLLM operator."""
import torch
OP_NAME = "torch_add_comparison"
DTYPES = [torch.float32]
IS_INPLACE = False
def native():
    return torch.add
def grid():
    return [{"n": 1048576}]
def build_inputs(binding, dtype, device):
    torch.manual_seed(42)
    return (torch.randn(binding["n"], dtype=dtype, device=device), torch.randn(binding["n"], dtype=dtype, device=device)), {}
def key_shape(binding):
    return [binding["n"]]
