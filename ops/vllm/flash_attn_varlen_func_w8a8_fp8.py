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

"""flash_attn_varlen_func_w8a8_fp8 baseline（方案 B）。

native：vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func
    （已回源码核对：/Users/zjs-office/all_code/vllm/vllm/vllm_flash_attn/
    flash_attn_interface.py#L176，签名含 q_descale/k_descale/v_descale/fa_version）。
    说明：w8a8_fp8 的 **NV 原生基准**不是 fp8 kernel——FlagGems-vllm 的
    test_flash_attn_varlen_func_w8a8_fp8.py 里 NVIDIA 分支
    baseline_flash_attn_varlen_func_w8a8_fp8（#L168-182）用的是对 fp8 反量化回的
    **bf16 变长 FlashAttention**（fa_version=2、非 paged、给 cu_seqlens_k）；
    fp8 量化 kernel 是待验收的国产后端算子（gems_* 分支），非 NV 侧基准。
    故本模块复刻该 bf16 基准路径，measured 的即 NV 原生 FA2 varlen。

输入构造复刻 test_flash_attn_varlen_func_w8a8_fp8.py 的
    flash_attn_varlen_func_w8a8_fp8_input_fn（非 paged 分支，config 为 5 元组）：
    config = (batch_or_qlens, seqlen_or_kvlens, num_heads, head_size, causal)
      - 首元为 int：_make_deterministic_ragged_lengths(batch, max_seq) 造 q/kv 变长；
      - 首元为 tuple：直接作为每序列 q/kv 长度。
    q/k/v = uniform(-0.05,0.05) 的 (sum_len, num_heads, head_size)；NV 基准侧用
    bf16（benchmark 把 fp8 反量化回 bf16 再喂 FA2），scale=1/sqrt(head_size)。
shape 取 benchmark 的 core_shapes（10 组采样档，COMPREHENSIVE 档才展开全量）。
"""

import importlib
import math

import torch

OP_NAME = "flash_attn_varlen_func_w8a8_fp8"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False  # 主输出返回；out 张量同时被写

# benchmark core_shapes（非 paged）：
#   (batch_or_qlens, seqlen_or_kvlens, num_heads, head_size, causal)
_RAGGED_Q = (32, 128, 512, 4096)
_RAGGED_KV = (1, 17, 129, 8192)
_DIM_Q = (17, 63, 129, 511)
_DIM_KV = (33, 1, 257, 513)
_CORE_SHAPES = [
    (1, 512, 16, 128, False),
    (1, 512, 32, 64, False),
    (2, 512, 16, 128, True),
    (1, 2048, 32, 64, False),
    (4, 4096, 32, 64, False),
    (8, 8192, 16, 128, True),
    (_RAGGED_Q, _RAGGED_KV, 32, 64, False),
    (1, 512, 8, 96, False),
    (_DIM_Q, _DIM_KV, 8, 192, True),
    (1, 4096, 8, 256, False),
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


def _make_deterministic_ragged_lengths(batch, max_seq_len):
    """复刻 benchmark 同名函数：确定性地造 batch 条 q/kv 变长。"""
    if batch == 1:
        return (max_seq_len,), (max_seq_len,)
    step = max(1, max_seq_len // (2 * batch))
    q_seq_lens = tuple(
        max(1, max_seq_len - i * step - i % 3) for i in range(batch)
    )
    kv_seq_lens = q_seq_lens[1:] + q_seq_lens[:1]
    return q_seq_lens, kv_seq_lens


def _resolve_lengths(config):
    batch_or_q, seq_or_kv, num_heads, head_size, causal = config
    if isinstance(batch_or_q, (list, tuple)):
        q_seq_lens = tuple(batch_or_q)
        kv_seq_lens = tuple(seq_or_kv)
    else:
        q_seq_lens, kv_seq_lens = _make_deterministic_ragged_lengths(
            batch_or_q, seq_or_kv
        )
    return q_seq_lens, kv_seq_lens, num_heads, head_size, causal


def grid():
    return [{"shape_idx": i} for i in range(len(_CORE_SHAPES))]


def _make_cu_seqlens(seq_lens, device):
    import itertools
    cu = (0,) + tuple(itertools.accumulate(seq_lens))
    return torch.tensor(cu, device=device, dtype=torch.int32)


def build_inputs(binding, dtype, device):
    config = _CORE_SHAPES[binding["shape_idx"]]
    q_seq_lens, kv_seq_lens, num_heads, head_size, causal = _resolve_lengths(config)

    # NV 原生基准走 bf16 反量化路径（benchmark #L168-182），非 fp8 kernel。
    q = torch.empty(
        (sum(q_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    k = torch.empty(
        (sum(kv_seq_lens), num_heads, head_size), device=device, dtype=dtype
    ).uniform_(-0.05, 0.05)
    v = torch.empty_like(k).uniform_(-0.05, 0.05)
    cu_seqlens_q = _make_cu_seqlens(q_seq_lens, device)
    cu_seqlens_k = _make_cu_seqlens(kv_seq_lens, device)
    out = torch.empty_like(q)
    scale = 1.0 / math.sqrt(head_size)

    # 对齐 benchmark NV baseline 的位置参数：
    #   flash_attn_varlen_func(q, k, v, max_seqlen_q, cu_seqlens_q,
    #       max_seqlen_k, cu_seqlens_k, softmax_scale=, causal=, out=, fa_version=2)
    args = (q, k, v, max(q_seq_lens), cu_seqlens_q, max(kv_seq_lens),
            cu_seqlens_k)
    kwargs = {
        "softmax_scale": scale,
        "causal": bool(causal),
        "out": out,
        "fa_version": 2,
    }
    return args, kwargs


def key_shape(binding):
    config = _CORE_SHAPES[binding["shape_idx"]]
    q_seq_lens, kv_seq_lens, num_heads, head_size, causal = _resolve_lengths(config)
    return [sum(q_seq_lens), len(q_seq_lens), num_heads, head_size,
            max(kv_seq_lens)]


def config(binding, dtype):
    cfg = _CORE_SHAPES[binding["shape_idx"]]
    q_seq_lens, kv_seq_lens, num_heads, head_size, causal = _resolve_lengths(cfg)
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [sum(q_seq_lens), num_heads, head_size], "dtype": dt},
            "k": {"shape": [sum(kv_seq_lens), num_heads, head_size], "dtype": dt},
            "v": {"shape": [sum(kv_seq_lens), num_heads, head_size], "dtype": dt},
            "cu_seqlens_q": {"shape": [len(q_seq_lens) + 1], "dtype": "torch.int32"},
            "cu_seqlens_k": {"shape": [len(kv_seq_lens) + 1], "dtype": "torch.int32"},
            "max_seqlen_q": {"scalar": max(q_seq_lens)},
            "max_seqlen_k": {"scalar": max(kv_seq_lens)},
            "causal": {"scalar": bool(causal)},
        },
        "outputs": {
            "out": {"shape": [sum(q_seq_lens), num_heads, head_size], "dtype": dt},
        },
        "dims": {
            "num_seqs": len(q_seq_lens),
            "num_heads": num_heads,
            "head_size": head_size,
            "total_q": sum(q_seq_lens),
            "total_kv": sum(kv_seq_lens),
            "quant": "w8a8_fp8",
            "note": "NV 原生基准为 bf16 FA2 varlen（fp8 反量化路径），"
                    "非 fp8 kernel；fp8 量化算子为待验收国产后端算子",
        },
    }
