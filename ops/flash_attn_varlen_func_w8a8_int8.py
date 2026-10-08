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

"""flash_attn_varlen_func_w8a8_int8 baseline（方案 B）。

native：vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func
    （已回源码核对：/Users/zjs-office/all_code/vllm/vllm/vllm_flash_attn/
    flash_attn_interface.py#L176）。
    说明：w8a8_int8 的 **NV 原生基准**不是 int8 kernel——FlagGems-vllm 的
    test_flash_attn_varlen_func_w8a8_int8.py 里 NVIDIA 分支基准
    _varlen_bf16_baseline（#L163-164）就是对 bf16 变长 **paged** FlashAttention 的
    调用（flaggems_vllm.flash_attn_varlen_func 透传到 vllm 原生）；int8 量化 kernel
    是待验收的国产后端算子（gems 分支），非 NV 侧基准。故本模块复刻 bf16 paged 基准。

输入构造复刻基类 test_flash_attn_varlen_func.py 的 flash_attn_varlen_input_fn
    （paged 路径）+ int8 变体的 set_shapes 工作负载：
    固定 num_heads=32, num_heads_k=8, head_dim=128, block_size=16；
    18 组 standard_attention 工作负载，每组 (request_count, q_len, kv_len)。
    query      = randn(sum_q, num_heads, head_dim)
    key/value_cache = randn(num_blocks, block_size, num_heads_k, head_dim)
    cu_seqlens_q / seqused_k 由各序列长度得，block_table = arange(num_blocks)
      reshape(batch,-1)（int8 变体 flash_attn_varlen_input_fn 的覆写，#L77-85）。
    scale = head_dim**-0.5，window=(-1,-1)。
shape 来源：vllm standard_attention.yaml 的 18 组工作负载（core/comprehensive 同）。
"""

import importlib
from itertools import accumulate

import torch

OP_NAME = "flash_attn_varlen_func_w8a8_int8"
# 与 benchmark 一致：test_flash_attn_varlen_func_w8a8_int8.py#L196 dtypes=[bfloat16]。
DTYPES = [torch.bfloat16]
IS_INPLACE = False

_NUM_HEADS = 32
_NUM_HEADS_K = 8
_HEAD_DIM = 128
_BLOCK_SIZE = 16

# standard_attention.yaml 的 18 组工作负载：每组若干 (request_count, q_len, kv_len)
_WORKLOADS = [
    ((1, 512, 512),),
    ((1, 2048, 2048),),
    ((1, 4096, 4096),),
    ((1, 8192, 8192),),
    ((8, 1, 1024),),
    ((16, 1, 2048),),
    ((32, 1, 1024),),
    ((64, 1, 4096),),
    ((2, 2048, 2048), (8, 1, 1024)),
    ((4, 1024, 1024), (16, 1, 2048)),
    ((2, 4096, 4096), (32, 1, 1024)),
    ((16, 2, 1024),),
    ((16, 4, 1024),),
    ((16, 8, 1024),),
    ((32, 4, 2048),),
    ((8, 8, 4096),),
    ((1, 1024, 2048),),
    ((2, 1024, 4096),),
]


def native():
    """解析 flash_attn_varlen_func；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.vllm_flash_attn.flash_attn_interface"
        )
    except ImportError:
        return None
    op = getattr(mod, "flash_attn_varlen_func", None)
    return op if callable(op) else None


def _lengths(groups):
    """由一组 (count, q_len, kv_len) 展开出每序列的 q/kv 长度。"""
    qlens = tuple(q for count, q, kv in groups for _ in range(count))
    klens = tuple(kv for count, q, kv in groups for _ in range(count))
    return qlens, klens


def _num_blocks(klens):
    return len(klens) * ((max(klens) + _BLOCK_SIZE - 1) // _BLOCK_SIZE)


def grid():
    return [{"shape_idx": i} for i in range(len(_WORKLOADS))]


def build_inputs(binding, dtype, device):
    groups = _WORKLOADS[binding["shape_idx"]]
    qlens, klens = _lengths(groups)
    num_seqs = len(qlens)
    num_blocks = _num_blocks(klens)
    scale = _HEAD_DIM ** -0.5

    query = torch.randn(sum(qlens), _NUM_HEADS, _HEAD_DIM, dtype=dtype, device=device)
    out = torch.empty_like(query)
    key_cache = torch.randn(
        num_blocks, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM, dtype=dtype, device=device
    )
    value_cache = torch.randn_like(key_cache)
    cu_seqlens_q = torch.tensor((0, *accumulate(qlens)), dtype=torch.int32,
                                device=device)
    seqused_k = torch.tensor(klens, dtype=torch.int32, device=device)
    # int8 变体覆写：按 arange 连续分配 cache block（含短序列的未用槽位）。
    block_table = torch.arange(num_blocks, dtype=torch.int32, device=device).reshape(
        num_seqs, -1
    )

    args = (query, key_cache, value_cache, max(qlens), cu_seqlens_q, max(klens),
            None, seqused_k)
    kwargs = {
        "softmax_scale": scale,
        "causal": True,
        "window_size": (-1, -1),
        "block_table": block_table,
        "out": out,
        "fa_version": 2,
    }
    return args, kwargs


def key_shape(binding):
    groups = _WORKLOADS[binding["shape_idx"]]
    qlens, klens = _lengths(groups)
    return [sum(qlens), len(qlens), _NUM_HEADS, _HEAD_DIM, max(klens)]


def config(binding, dtype):
    groups = _WORKLOADS[binding["shape_idx"]]
    qlens, klens = _lengths(groups)
    num_seqs = len(qlens)
    num_blocks = _num_blocks(klens)
    max_kv = max(klens)
    blocks_per_seq = num_blocks // num_seqs
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [sum(qlens), _NUM_HEADS, _HEAD_DIM], "dtype": dt},
            "k": {"shape": [num_blocks, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "v": {"shape": [num_blocks, _BLOCK_SIZE, _NUM_HEADS_K, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "cu_seqlens_q": {"shape": [num_seqs + 1], "dtype": "torch.int32"},
            "seqused_k": {"shape": [num_seqs], "dtype": "torch.int32"},
            "block_table": {"shape": [num_seqs, blocks_per_seq],
                            "dtype": "torch.int32"},
            "max_seqlen_q": {"scalar": max(qlens)},
            "max_seqlen_k": {"scalar": max_kv},
            "causal": {"scalar": True},
        },
        "outputs": {
            "out": {"shape": [sum(qlens), _NUM_HEADS, _HEAD_DIM], "dtype": dt},
        },
        "dims": {
            "num_seqs": num_seqs,
            "num_heads": _NUM_HEADS,
            "num_heads_k": _NUM_HEADS_K,
            "head_dim": _HEAD_DIM,
            "block_size": _BLOCK_SIZE,
            "num_blocks": num_blocks,
            "max_kv_len": max_kv,
            "quant": "w8a8_int8",
            "note": "NV 原生基准为 bf16 FA2 paged varlen，非 int8 kernel；"
                    "int8 量化算子为待验收国产后端算子",
        },
    }
