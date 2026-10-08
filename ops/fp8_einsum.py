# Copyright 2026 FlagOS Contributors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""fp8_einsum baseline（方案 B）。

native：vllm.utils.deep_gemm.fp8_einsum(subscripts, (x_data, x_scale),
        (y_data, y_scale), out, recipe=...) -> out。deep_gemm 惰性绑定 wrapper，
        调用时 _lazy_init()；deep_gemm 未加载/该版本无 fp8_einsum 时抛错（采集器会在
        调用处捕获并跳过）。native() 只负责 import 并返回 wrapper。
        src: vllm/utils/deep_gemm.py#L467

输入构造复刻 FlagGems-vllm/benchmark/test_fp8_einsum.py 的
FP8EinsumBenchmark.get_input_iter + _make_fp8_einsum_inputs（einsum "bhr,hdr->bhd"，
block_shape=(block_n, block_k)=(128,128)）：
    x = randn(b, h, r) bf16  ->  x_data (b, h, r) fp8_e4m3,
                                 x_scale (b, h, ceil(r/block_k)) fp32  （per-token）
    y = randn(h, d, r) bf16  ->  y_data (h, d, r) fp8_e4m3,
                                 y_scale (h, ceil(d/block_n), ceil(r/block_k)) fp32 （per-block）
    out = empty(b, h, d) bf16
    调用 fp8_einsum("bhr,hdr->bhd", (x_data,x_scale), (y_data,y_scale), out,
                    recipe=(1,128,128))
注：benchmark 用 per_token_cast_to_fp8 / per_block_cast_to_fp8 算 UE8M0 scale（含
位运算），采集只需形状/dtype 正确的实参，故此处 scale 用同形状 fp32 随机占位
（真机若 deep_gemm 拒绝该布局，采集器会跳过该点）。

shape 来源：benchmark 的 self.shapes = core_shapes.yaml「fp8_einsum」键（b,h,r,d，
8 档）∪ set_more_shapes()（batches × {flash(8,4096,1024), pro(16,7168,1024)}），
后者为前者超集，合并去重后共 22 档。shape_desc = "b, h, r, d"。
"""

import importlib
import math

import torch

OP_NAME = "fp8_einsum"
DTYPES = [torch.bfloat16]  # 输出 out 的 dtype；operands 为 fp8（build_inputs 固定）
IS_INPLACE = True  # 结果写入 out

_FP8_DTYPE = getattr(torch, "float8_e4m3fn", None)
_BLOCK_SHAPE = (128, 128)  # (block_n, block_k)，benchmark DEFAULT_BLOCK_SHAPE
_RECIPE = (1, 128, 128)    # benchmark 固定 recipe

# benchmark set_more_shapes()：batches × hrd_groups（flash/pro）；(b, h, r, d)。
_BATCHES = (1, 4, 8, 16, 32, 64, 128, 4096, 8192, 16384, 32768)
_HRD_GROUPS = ((8, 4096, 1024), (16, 7168, 1024))  # flash, pro
_SHAPES = [(b, h, r, d) for (h, r, d) in _HRD_GROUPS for b in _BATCHES]


def native():
    """解析 vllm.utils.deep_gemm.fp8_einsum wrapper；解析不到返回 None。

    只 import + 取 wrapper；不触发 _lazy_init（那会在调用时发生）。
    """
    try:
        mod = importlib.import_module("vllm.utils.deep_gemm")
    except ImportError:
        return None
    op = getattr(mod, "fp8_einsum", None)
    return op if callable(op) else None


def grid():
    return [{"b": b, "h": h, "r": r, "d": d} for (b, h, r, d) in _SHAPES]


def _fp8(shape, device):
    t = torch.randn(shape, device=device, dtype=torch.float32)
    if hasattr(torch, "finfo"):
        finfo = torch.finfo(_FP8_DTYPE)
        t = t.clamp(min=finfo.min, max=finfo.max)
    return t.to(_FP8_DTYPE)


def build_inputs(binding, dtype, device):
    b, h, r, d = binding["b"], binding["h"], binding["r"], binding["d"]
    block_n, block_k = _BLOCK_SHAPE
    rk = math.ceil(r / block_k)
    dn = math.ceil(d / block_n)

    x_data = _fp8((b, h, r), device)
    x_scale = torch.randn(b, h, rk, dtype=torch.float32, device=device)
    y_data = _fp8((h, d, r), device)
    y_scale = torch.randn(h, dn, rk, dtype=torch.float32, device=device)
    out = torch.empty(b, h, d, dtype=torch.bfloat16, device=device)

    args = ("bhr,hdr->bhd", (x_data, x_scale), (y_data, y_scale), out)
    return args, {"recipe": _RECIPE}


def key_shape(binding):
    return [binding["b"], binding["h"], binding["r"], binding["d"]]


def config(binding, dtype):
    b, h, r, d = binding["b"], binding["h"], binding["r"], binding["d"]
    block_n, block_k = _BLOCK_SHAPE
    rk = math.ceil(r / block_k)
    dn = math.ceil(d / block_n)
    fp8 = str(_FP8_DTYPE)
    return {
        "inputs": {
            "subscripts": {"scalar": "bhr,hdr->bhd"},
            "x_data": {"shape": [b, h, r], "dtype": fp8},
            "x_scale": {"shape": [b, h, rk], "dtype": "torch.float32",
                        "note": "per-token block scale 占位"},
            "y_data": {"shape": [h, d, r], "dtype": fp8},
            "y_scale": {"shape": [h, dn, rk], "dtype": "torch.float32",
                        "note": "per-block scale 占位"},
            "out": {"shape": [b, h, d], "dtype": "torch.bfloat16", "note": "原地写"},
            "recipe": {"scalar": list(_RECIPE)},
        },
        "outputs": {
            "out": {"shape": [b, h, d], "dtype": "torch.bfloat16"},
        },
        "dims": {"b": b, "h": h, "r": r, "d": d,
                 "block_n": block_n, "block_k": block_k},
    }
