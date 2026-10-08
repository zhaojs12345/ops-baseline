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

"""beam_search_score_ baseline（方案 B）——beam_search_score 的 in-place 变体。

native：vllm.entrypoints.generate.beam_search.utils.get_beam_search_score(
        tokens, cumulative_logprob, eos_token_id, length_penalty) -> float。
    签名见 .../beam_search/utils.py#L137（已回源核对）。vLLM 无 in-place 专用接口
    （见 ops_interface_1.csv 备注「vllm 无 in-place 专用接口，同上」），故与
    ops/beam_search_score.py 复用同一标量函数。

重要说明（存疑点）：同 ops/beam_search_score.py——该 vLLM 符号是标量 CPU 辅助函数，
与 benchmark 的张量算子（test_beam_search_score.py::test_beam_search_score_，
gems_op = flaggems_vllm.beam_search_score_，is_inplace=True）功能对应但签名不同，
且无 CUDA kernel。native() 解析真实标量函数，build_inputs 按其真实标量签名构造实参，
主键沿用 benchmark 的 (batch, vocab) 形状对齐看板。

shape 来源：benchmark BeamSearchScoreBenchmark.DEFAULT_SHAPES（同 beam_search_score）。
"""

import importlib

import torch

OP_NAME = "beam_search_score_"
DTYPES = [torch.float32]  # 标量函数无 dtype；占位以满足契约
IS_INPLACE = True  # gems 侧 in-place 变体（vLLM 侧无专用 in-place 接口）

# benchmark DEFAULT_SHAPES：(batch_size, vocab_size)
_SHAPES = [
    (16, 512),
    (32, 1024),
    (64, 2048),
    (128, 4096),
    (256, 8192),
]

_EOS_TOKEN_ID = 2
_LENGTH_PENALTY = 1.0
_CUMULATIVE_LOGPROB = -12.5


def native():
    """解析 get_beam_search_score（标量 CPU 函数）；解析不到返回 None。"""
    try:
        mod = importlib.import_module(
            "vllm.entrypoints.generate.beam_search.utils")
    except ImportError:
        return None
    op = getattr(mod, "get_beam_search_score", None)
    return op if callable(op) else None


def grid():
    return [{"batch_size": b, "vocab_size": v} for (b, v) in _SHAPES]


def build_inputs(binding, dtype, device):
    seq_len = binding["vocab_size"]
    tokens = list(range(seq_len))
    args = (tokens, _CUMULATIVE_LOGPROB, _EOS_TOKEN_ID, _LENGTH_PENALTY)
    return args, {}


def key_shape(binding):
    return [binding["batch_size"], binding["vocab_size"]]


def config(binding, dtype):
    return {
        "inputs": {
            "tokens": {"shape": [binding["vocab_size"]], "dtype": "list[int]",
                       "note": "名义：vocab_size 作 tokens 长度"},
            "cumulative_logprob": {"scalar": _CUMULATIVE_LOGPROB},
            "eos_token_id": {"scalar": _EOS_TOKEN_ID},
            "length_penalty": {"scalar": _LENGTH_PENALTY},
        },
        "outputs": {
            "score": {"scalar": "float"},
        },
        "dims": {
            "batch_size": binding["batch_size"],
            "vocab_size": binding["vocab_size"],
            "shape_source": "benchmark DEFAULT_SHAPES（标量 native，无 GPU kernel；"
                            "主键沿用张量形状对齐看板）",
        },
    }
