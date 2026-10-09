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

"""beam_search_score baseline（方案 B）。

native：vllm.entrypoints.generate.beam_search.utils.get_beam_search_score(
        tokens, cumulative_logprob, eos_token_id, length_penalty) -> float。
    签名见 .../beam_search/utils.py#L137（已回源核对）。

重要说明（存疑点）：该 vLLM 符号是**标量 CPU 辅助函数**（输入 tokens: list[int] 与
几个标量，返回一个 float），与 FlagGems-vllm/benchmark/test_beam_search_score.py 里
BeamSearchScoreBenchmark 的**张量**算子（torch_op = log_probs + beam_scores.unsqueeze(-1)，
gems_op = flaggems_vllm.beam_search_score，输入 log_probs[B,V]/beam_scores[B]）功能对应
但签名不同——vLLM 侧没有张量化的 beam_search_score custom op。因此：
    - native() 解析真实标量函数（不编造张量接口）；
    - build_inputs 按 native 的**真实标量签名**构造实参，使 native 可被真实调用；
    - 该函数无 CUDA kernel，真实采集时 NCU 捕获不到 GPU kernel（采集器会按
      「无 CUDA 实现 / 无 kernel」处理）；此处主键仍沿用 benchmark 的 (batch, vocab)
      形状以便与验收看板对齐。

shape 来源：benchmark 的 BeamSearchScoreBenchmark.DEFAULT_SHAPES（init_user_config 跳过
yaml，直接用 DEFAULT_SHAPES）：(batch_size, vocab_size) ∈
{(16,512),(32,1024),(64,2048),(128,4096),(256,8192)}。标量 native 无 batch/vocab 维度，
这里用 vocab_size 作为 tokens 序列长度的名义取值构造一条样例输入。
"""

import importlib

OP_NAME = "beam_search_score"
# 标量函数无 dtype 概念；保留一个占位 dtype 以满足契约（不会影响标量调用）。
import torch  # noqa: E402

DTYPES = [torch.float32]
IS_INPLACE = False

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
    # native 真实签名：(tokens: list[int], cumulative_logprob: float,
    #                   eos_token_id: int, length_penalty: float)
    seq_len = binding["vocab_size"]  # 名义：用 vocab_size 当 tokens 长度
    tokens = list(range(seq_len))
    args = (tokens, _CUMULATIVE_LOGPROB, _EOS_TOKEN_ID, _LENGTH_PENALTY)
    return args, {}


def key_shape(binding):
    # 主键沿用 benchmark 的 (batch, vocab) 形状，便于验收看板对齐。
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
