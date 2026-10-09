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

"""concat_and_cache_mla baseline（方案 B）。

native：vllm._custom_ops.concat_and_cache_mla(kv_c, k_pe, kv_cache, slot_mapping,
        kv_cache_dtype, scale) -> None（底层 torch.ops._C_cache_ops.concat_and_cache_mla，
        原地写 kv_cache）。签名见 vllm/_custom_ops.py#L2868（已回源核对）。

输入构造复刻 FlagGems-vllm/benchmark/test_concat_and_cache_mla.py 的 input_kwargs，
shape=(kv_lora_rank, qk_rope_head_dim, num_tokens, block_size, num_blocks)：
    total_slots = num_blocks * block_size
    slot_mapping = sample(range(total_slots), num_tokens)       long
    kv_c   = randn(num_tokens, kv_lora_rank)
    k_pe   = randn(num_tokens, qk_rope_head_dim)
    entry_size = kv_lora_rank + qk_rope_head_dim
    scale  = tensor(0.1, float32)
    kv_cache = zeros(num_blocks, block_size, entry_size)
    kwargs: kv_cache_dtype="auto", scale=scale
shape 取自 core_shapes.yaml「ConcatAndCacheMLABenchmark」。
"""

import importlib
import random

import torch

OP_NAME = "concat_and_cache_mla"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # kv_cache 原地写回

# core_shapes.yaml「ConcatAndCacheMLABenchmark」：
# (kv_lora_rank, qk_rope_head_dim, num_tokens, block_size, num_blocks)
_SHAPES = [
    (16, 8, 84, 32, 4),
    (32, 4, 21, 16, 16),
    (64, 8, 21, 8, 4),
    (512, 64, 42, 16, 8),
    (1024, 64, 84, 16, 8),
]

_KV_CACHE_DTYPE = "auto"


def native():
    """解析 vllm._custom_ops.concat_and_cache_mla；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "concat_and_cache_mla", None)
    return op if callable(op) else None


def grid():
    return [
        {"kv_lora_rank": r, "qk_rope_head_dim": p, "num_tokens": t,
         "block_size": b, "num_blocks": n}
        for (r, p, t, b, n) in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    r = binding["kv_lora_rank"]
    p = binding["qk_rope_head_dim"]
    t = binding["num_tokens"]
    b = binding["block_size"]
    n = binding["num_blocks"]

    total_slots = n * b
    slot_mapping_lst = random.sample(range(total_slots), t)
    slot_mapping = torch.tensor(slot_mapping_lst, dtype=torch.int64, device=device)

    kv_c = torch.randn(t, r, dtype=dtype, device=device)
    k_pe = torch.randn(t, p, dtype=dtype, device=device)
    entry_size = r + p
    scale = torch.tensor(0.1, dtype=torch.float32, device=device)
    kv_cache = torch.zeros(n, b, entry_size, dtype=dtype, device=device)

    args = (kv_c, k_pe, kv_cache, slot_mapping, _KV_CACHE_DTYPE, scale)
    return args, {}


def key_shape(binding):
    return [binding["kv_lora_rank"], binding["qk_rope_head_dim"],
            binding["num_tokens"], binding["block_size"], binding["num_blocks"]]


def config(binding, dtype):
    r = binding["kv_lora_rank"]
    p = binding["qk_rope_head_dim"]
    t = binding["num_tokens"]
    b = binding["block_size"]
    n = binding["num_blocks"]
    entry_size = r + p
    dt = str(dtype)
    return {
        "inputs": {
            "kv_c": {"shape": [t, r], "dtype": dt},
            "k_pe": {"shape": [t, p], "dtype": dt},
            "kv_cache": {"shape": [n, b, entry_size], "dtype": dt,
                         "note": "原地写回"},
            "slot_mapping": {"shape": [t], "dtype": "torch.int64"},
            "kv_cache_dtype": {"scalar": _KV_CACHE_DTYPE},
            "scale": {"shape": [], "dtype": "torch.float32"},
        },
        "outputs": {
            "kv_cache": {"shape": [n, b, entry_size], "dtype": dt},
        },
        "dims": {"kv_lora_rank": r, "qk_rope_head_dim": p, "num_tokens": t,
                 "block_size": b, "num_blocks": n, "entry_size": entry_size},
    }
