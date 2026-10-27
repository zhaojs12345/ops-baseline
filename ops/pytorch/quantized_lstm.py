# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native floating LSTM baseline for FlagGems' dequantized-weight contract.

Source: FlagGems ac9a1f703757537a3d54cc1ac1febeca410316c0,
benchmark/test_quantized_lstm.py. This benchmarks the original native reference,
torch._VF.lstm with dequantized packed int8 weights, not ATen dynamic quantized
LSTM (which has different arithmetic). Packed ScriptObjects are irrelevant to
this floating native reference and are excluded from its native argument list.
All three original shapes and all three FLOAT_DTYPES are retained.
"""
import torch

OP_NAME = "quantized_lstm"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
SHAPES = [(4, 32, 256, 256), (4, 64, 512, 512), (8, 64, 256, 256)]
RECIPE = {"source_commit": "ac9a1f703757537a3d54cc1ac1febeca410316c0",
          "source_file": "benchmark/test_quantized_lstm.py",
          "seed": 0, "quantized_engine": "fbgemm", "weight_distribution": "randn * 0.3",
          "bias_distribution": "randn * 0.1", "weight_qscheme": "per_tensor_affine",
          "weight_qdtype": "torch.qint8", "weight_zero_point": 0,
          "weight_scale": "max(abs(weight).max().item() / 127, 1e-8)",
          "rounding": "linear_prepack -> linear_unpack -> dequantize -> input dtype",
          "native_expression": "torch._VF.lstm(value, hx, deq, *flags)",
          "packed_params": "CPU ScriptObjects used by candidate only, excluded from original floating native reference"}


def _native(value, hx, deq, *flags):
    return torch._VF.lstm(value, hx, deq, *flags)


def native():
    return _native


def grid():
    return [{"shape": list(shape)} for shape in SHAPES]


def key_shape(binding):
    return list(binding["shape"])


def build_inputs(binding, dtype, device):
    seq, batch, inp, hidden = binding["shape"]
    torch.manual_seed(0)
    value = torch.randn(seq, batch, inp, dtype=dtype, device=device)
    hx = [torch.randn(1, batch, hidden, dtype=dtype, device=device),
          torch.randn(1, batch, hidden, dtype=dtype, device=device)]
    old = torch.backends.quantized.engine
    torch.backends.quantized.engine = "fbgemm"
    try:
        wi = torch.randn(4 * hidden, inp) * 0.3
        wh = torch.randn(4 * hidden, hidden) * 0.3
        bi = torch.randn(4 * hidden) * 0.1
        bh = torch.randn(4 * hidden) * 0.1
        qi = torch.quantize_per_tensor(wi, max(wi.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        qh = torch.quantize_per_tensor(wh, max(wh.abs().max().item() / 127, 1e-8), 0, torch.qint8)
        pi = torch.ops.quantized.linear_prepack(qi, bi)
        ph = torch.ops.quantized.linear_prepack(qh, bh)
        # Reproduce the source's packing operation even though native floating
        # LSTM consumes only the unpacked weights and original biases.
        torch.ops.quantized.make_quantized_cell_params_dynamic(pi, ph, bi, bh, True)
        deq = [torch.ops.quantized.linear_unpack(pi)[0].dequantize().to(device=device, dtype=dtype),
               torch.ops.quantized.linear_unpack(ph)[0].dequantize().to(device=device, dtype=dtype),
               bi.to(device=device, dtype=dtype), bh.to(device=device, dtype=dtype)]
        return (value, hx, deq, True, 1, 0.0, False, False, False), {}
    finally:
        torch.backends.quantized.engine = old


def _tensor(shape, dtype):
    x = torch.empty(shape, device="meta", dtype=dtype)
    return {"shape": list(x.shape), "dtype": str(dtype), "stride": list(x.stride()), "requires_grad": False}


def config(binding, dtype):
    seq, batch, inp, hidden = binding["shape"]
    ordinal = SHAPES.index(tuple(binding["shape"]))
    args = [_tensor((seq, batch, inp), dtype),
            [_tensor((1, batch, hidden), dtype)] * 2,
            [_tensor((4 * hidden, inp), dtype), _tensor((4 * hidden, hidden), dtype),
             _tensor((4 * hidden,), dtype), _tensor((4 * hidden,), dtype)],
            True, 1, 0.0, False, False, False]
    return {"inputs": {"args": args, "kwargs": {}}, "recipe": dict(RECIPE),
            "case_id": "benchmark/test_quantized_lstm.py::test_quantized_lstm::core::" + str(dtype).split('.')[-1] + "::" + str(ordinal)}
