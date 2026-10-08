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

"""flash_attn_varlen_func baseline（方案 B）。

native：vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func(
            q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k,
            cu_seqlens_k=None, seqused_k=None, q_v=None, dropout_p=0.0,
            softmax_scale=None, causal=False, window_size=None, softcap=0.0,
            alibi_slopes=None, deterministic=False, return_attn_probs=False,
            block_table=None, return_softmax_lse=False, out=None, ...)
    变长（varlen）paged FlashAttention 前向；这里走 paged 路径（k/v 为 KV cache
    分块，配合 block_table）。
    src: vllm/vllm_flash_attn/flash_attn_interface.py#L176

输入构造复刻 FlagGems-vllm/benchmark/test_flash_attn_varlen_func.py 的
flash_attn_varlen_input_fn（Qwen3.6-35B-A3B 采样档：6 组 TP1 + 6 组 TP4）：
    query      = randn(cu_query_lens[-1], num_query_heads, head_size)
    key_cache  = randn(num_blocks, block_size, num_kv_heads, head_size)
    value_cache= randn_like(key_cache)
    cu_query_lens = tensor(cu_seq_lens_q, int32)
    seqused_k     = tensor(seqused_k, int32)
    block_tables  = randint(0, num_blocks,
                            (num_seqs, ceil(max_kv_len/block_size)), int32)
    out           = empty_like(query)
    scale = head_size**-0.5, causal=True, window_size=(-1,-1)
调用位置参数顺序与 benchmark 完全一致（20 个位置参数 + 一组尾部 kwargs）。
每档的 (num_heads, num_heads_k)、block_size、num_blocks 随档位变化（见 _ALL_* 列表）；
head_dim 固定 256，alibi=False，soft_cap=None（→ softcap 0）。
shape 网格取 benchmark set_shapes 里的 12 组采样档（TP1 6 组 head=(16,2)/block=32/
blocks=73920；TP4 6 组 head=(4,1)/block=16，num_blocks 见 _ALL_NUM_BLOCKS）。
"""

import importlib

import torch

OP_NAME = "flash_attn_varlen_func"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False  # 主输出返回；out 张量同时被写（paged 路径）

_HEAD_DIM = 256

# 复刻 set_shapes（Qwen3.6-35B-A3B）：每档 cu_seq_lens_q。
_ALL_CU_SEQ_LENS_Q = [
    # TP1
    (0, 1035),
    tuple(range(257)),
    (0, 1, 2, 3, 4, 46, 4152, 8258, 12364, 16384),
    (0, 12, 16384),
    (0, 1, 2, 3, 67),
    (0, 16384),
    # TP4
    (0, 1036),
    tuple(range(513)),
    tuple(range(17))
    + (182, 1217, 2253, 3287, 4321, 5355, 6390, 7424, 8458, 9494, 10528,
       11562, 12596, 13631, 14667, 15702, 16384),
    (0, 12, 16384),
    (0, 12),
    (0, 16384),
]
_ALL_SEQUSED_K = [
    (1035,),
    (1,) * 256,
    (4110, 4108, 4107, 4107, 4106, 4106, 4106, 4106, 4020),
    (32780, 16372),
    (65560, 65555, 65550, 65546),
    (65536,),
    (1036,),
    (32,) * 512,
    (1038, 1035, 1035, 1037, 1035, 1035, 1035, 1035, 1035, 1035, 1036, 1035,
     1037, 1035, 1035, 1035, 1034, 1035, 1036, 1034, 1034, 1034, 1035, 1034,
     1034, 1036, 1034, 1034, 1034, 1035, 1036, 1035, 682),
    (65548, 16372),
    (32780,),
    (65536,),
]
# 每档 (num_heads, num_heads_k)、block_size、num_blocks。
_ALL_NUM_HEADS = [(16, 2)] * 6 + [(4, 1)] * 6
_ALL_BLOCK_SIZES = [32] * 6 + [16] * 6
_ALL_NUM_BLOCKS = [73920] * 6 + [605550, 16896] + [605550] * 4


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


def grid():
    return [{"shape_idx": idx} for idx in range(len(_ALL_CU_SEQ_LENS_Q))]


def _config(binding):
    idx = binding["shape_idx"]
    return _ALL_CU_SEQ_LENS_Q[idx], _ALL_SEQUSED_K[idx]


def _dims(binding):
    idx = binding["shape_idx"]
    num_heads, num_heads_k = _ALL_NUM_HEADS[idx]
    return num_heads, num_heads_k, _ALL_BLOCK_SIZES[idx], _ALL_NUM_BLOCKS[idx]


def build_inputs(binding, dtype, device):
    cu_query_lens, seqused_k = _config(binding)
    num_heads, num_heads_k, block_size, num_blocks = _dims(binding)

    num_seqs = len(cu_query_lens) - 1
    max_query_len = max(
        b - a for a, b in zip(cu_query_lens[:-1], cu_query_lens[1:])
    )
    max_kv_len = max(seqused_k)
    window_size = (-1, -1)
    scale = _HEAD_DIM ** -0.5

    query = torch.randn(
        cu_query_lens[-1], num_heads, _HEAD_DIM, dtype=dtype, device=device
    )
    out = torch.empty_like(query)
    key_cache = torch.randn(
        num_blocks, block_size, num_heads_k, _HEAD_DIM,
        dtype=dtype, device=device,
    )
    value_cache = torch.randn_like(key_cache)
    cu_query_lens_t = torch.tensor(cu_query_lens, dtype=torch.int32, device=device)
    seqused_k_t = torch.tensor(seqused_k, dtype=torch.int32, device=device)

    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    block_tables = torch.randint(
        0, num_blocks, (num_seqs, max_num_blocks_per_seq),
        dtype=torch.int32, device=device,
    )

    # 位置参数顺序完全对齐 benchmark：
    #   q, k, v, max_seqlen_q, cu_seqlens_q, max_seqlen_k, cu_seqlens_k(None),
    #   seqused_k, q_v(None), dropout_p, softmax_scale, causal, window_size,
    #   softcap, alibi_slopes(None), deterministic, return_attn_probs,
    #   block_table, return_softmax_lse, out
    args = (
        query,
        key_cache,
        value_cache,
        max_query_len,
        cu_query_lens_t,
        max_kv_len,
        None,          # cu_seqlens_k
        seqused_k_t,
        None,          # q_v
        0.0,           # dropout_p
        scale,         # softmax_scale
        True,          # causal
        window_size,
        0,             # softcap (soft_cap=None -> 0)
        None,          # alibi_slopes
        False,         # deterministic
        False,         # return_attn_probs
        block_tables,
        False,         # return_softmax_lse
        out,
    )
    kwargs = {
        "scheduler_metadata": None,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "s_aux": None,
        "num_splits": 0,
        "cp_world_size": 1,
        "cp_rank": 0,
        "cp_tot_seqused_k": None,
        "fa_version": 2,
    }
    return args, kwargs


def key_shape(binding):
    cu_query_lens, seqused_k = _config(binding)
    num_heads, _num_heads_k, _bs, _nb = _dims(binding)
    num_seqs = len(cu_query_lens) - 1
    return [cu_query_lens[-1], num_seqs, num_heads, _HEAD_DIM, max(seqused_k)]


def config(binding, dtype):
    cu_query_lens, seqused_k = _config(binding)
    num_heads, num_heads_k, block_size, num_blocks = _dims(binding)
    num_seqs = len(cu_query_lens) - 1
    total_q = cu_query_lens[-1]
    max_kv_len = max(seqused_k)
    max_num_blocks_per_seq = (max_kv_len + block_size - 1) // block_size
    dt = str(dtype)
    return {
        "inputs": {
            "q": {"shape": [total_q, num_heads, _HEAD_DIM], "dtype": dt},
            "k": {"shape": [num_blocks, block_size, num_heads_k, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "v": {"shape": [num_blocks, block_size, num_heads_k, _HEAD_DIM],
                  "dtype": dt, "note": "paged KV cache"},
            "cu_seqlens_q": {"shape": [num_seqs + 1], "dtype": "torch.int32"},
            "seqused_k": {"shape": [num_seqs], "dtype": "torch.int32"},
            "block_table": {"shape": [num_seqs, max_num_blocks_per_seq],
                            "dtype": "torch.int32"},
            "max_seqlen_q": {"scalar": max(
                b - a for a, b in zip(cu_query_lens[:-1], cu_query_lens[1:])
            )},
            "max_seqlen_k": {"scalar": max_kv_len},
            "causal": {"scalar": True},
            "window_size": {"scalar": [-1, -1]},
        },
        "outputs": {
            "out": {"shape": [total_q, num_heads, _HEAD_DIM], "dtype": dt},
        },
        "dims": {
            "total_q": total_q,
            "num_seqs": num_seqs,
            "num_heads": num_heads,
            "num_heads_k": num_heads_k,
            "head_dim": _HEAD_DIM,
            "block_size": block_size,
            "num_blocks": num_blocks,
            "max_kv_len": max_kv_len,
        },
    }
