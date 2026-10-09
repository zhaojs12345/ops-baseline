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

"""apply_rotary_pos_emb baseline（方案 B）。

native：vllm._custom_ops.rotary_embedding(positions, query, key, head_size,
        cos_sin_cache, is_neox) -> None（底层 torch.ops._C.rotary_embedding，原地改
        query/key）。签名见 vllm/_custom_ops.py#L222；调用点见
        vllm/model_executor/layers/rotary_embedding/base.py#L244（forward_cuda）。

注意 native 签名与 gems 的 apply_rotary_pos_emb(q, k, cos, sin) 不同但功能对应
（见 ops_interface_1.csv 备注）。因此输入**按 native 签名重构**，而非照搬 benchmark
的 gems 形态：
    - query/key 展平成 [num_tokens, num_heads * head_size]（kernel 内部 view 成
      [num_tokens, -1, head_size]）；
    - cos_sin_cache 为 [max_pos, rotary_dim]（cos||sin 拼接，rotary_dim == head_size
      时即 [max_pos, head_size]）；
    - positions 为 [num_tokens] 的 long，取值在 [0, seq_len)。

维度取自 FlagGems-vllm/benchmark/test_apply_rotary_pos_emb.py 的 rope_input_fn：
    batch_size=4, q_heads=8, k_heads=1, head_dim=64, seq_len=shape[0]。
shape 取自 core_shapes.yaml 的「RopeBenchmark」键（benchmark 用 RopeBenchmark，
set_more_shapes 返回 []）：seq_len ∈ {128, 192, 256, 384, 512}。
num_tokens = batch_size * seq_len。is_neox=True（benchmark 参考实现为 rotate_half 风格）。
"""

import importlib

import torch

OP_NAME = "apply_rotary_pos_emb"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = True  # query/key 原地写回

_BATCH = 4
_Q_HEADS = 8
_K_HEADS = 1
_HEAD_DIM = 64
_IS_NEOX = True

# core_shapes.yaml「RopeBenchmark」：shape[0] 为 seq_len。
_SEQ_LENS = [128, 192, 256, 384, 512]


def native():
    """解析 vllm._custom_ops.rotary_embedding；解析不到返回 None。"""
    try:
        mod = importlib.import_module("vllm._custom_ops")
    except ImportError:
        return None
    op = getattr(mod, "rotary_embedding", None)
    return op if callable(op) else None


def grid():
    return [{"seq_len": s} for s in _SEQ_LENS]


def build_inputs(binding, dtype, device):
    seq_len = binding["seq_len"]
    num_tokens = _BATCH * seq_len
    head_size = _HEAD_DIM
    rotary_dim = _HEAD_DIM

    # positions 在 [0, seq_len)：每个 batch 复用同一段位置。
    positions = (torch.arange(num_tokens, device=device) % seq_len).long()
    query = torch.randn(num_tokens, _Q_HEADS * head_size,
                        dtype=dtype, device=device)
    key = torch.randn(num_tokens, _K_HEADS * head_size,
                      dtype=dtype, device=device)
    # cos_sin_cache: [max_pos, rotary_dim]（cos||sin），max_pos = seq_len。
    cos_sin_cache = torch.randn(seq_len, rotary_dim, dtype=dtype, device=device)

    return (positions, query, key, head_size, cos_sin_cache, _IS_NEOX), {}


def key_shape(binding):
    seq_len = binding["seq_len"]
    return (f"bs{_BATCH}_sq{seq_len}_qh{_Q_HEADS}_kvh{_K_HEADS}_hd{_HEAD_DIM}")


def config(binding, dtype):
    seq_len = binding["seq_len"]
    num_tokens = _BATCH * seq_len
    head_size = _HEAD_DIM
    rotary_dim = _HEAD_DIM
    dt = str(dtype)
    return {
        "inputs": {
            "positions": {"shape": [num_tokens], "dtype": "torch.int64"},
            "query": {"shape": [num_tokens, _Q_HEADS * head_size], "dtype": dt,
                      "note": "原地写回"},
            "key": {"shape": [num_tokens, _K_HEADS * head_size], "dtype": dt,
                    "note": "原地写回"},
            "head_size": {"scalar": head_size},
            "cos_sin_cache": {"shape": [seq_len, rotary_dim], "dtype": dt},
            "is_neox": {"scalar": _IS_NEOX},
        },
        "outputs": {
            "query": {"shape": [num_tokens, _Q_HEADS * head_size], "dtype": dt},
            "key": {"shape": [num_tokens, _K_HEADS * head_size], "dtype": dt},
        },
        "dims": {
            "batch_size": _BATCH, "seq_len": seq_len, "num_tokens": num_tokens,
            "q_heads": _Q_HEADS, "k_heads": _K_HEADS, "head_dim": _HEAD_DIM,
            "shape_source": "native 签名重构（benchmark 为 gems 签名，不同但功能对应）",
        },
    }
