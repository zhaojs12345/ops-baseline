# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Five-output aten._flash_attention_forward, not the nine-output SDPA API.

The frozen recipes preserve both original benchmark modules, dense/varlen
layouts, GQA, causal/window masks, dropout and debug options. No input builder
or host oracle is timed. ALiBi was absent from the original H20 core catalog.
"""
import copy
import json
import math
from pathlib import Path

import torch

OP_NAME = "_flash_attention_forward"
DTYPES = [torch.float16, torch.bfloat16]
IS_INPLACE = False
_CATALOG = json.loads((Path(__file__).with_name("_inspect216_flash_forward_cases.json")).read_text())


def native():
    return torch.ops.aten._flash_attention_forward.default


def grid():
    return [{"recipe": index} for index in range(len(_CATALOG["recipes"]))]


def _recipe(binding):
    return _CATALOG["recipes"][binding["recipe"]]


def key_shape(binding):
    return _recipe(binding)["key"]


def build_inputs(binding, dtype, device):
    inp = _recipe(binding)["inputs"]
    if inp["seed"] is not None:
        torch.manual_seed(inp["seed"])
    q, k, v = (torch.empty(inp[name]["shape"], dtype=dtype, device=device).uniform_(-0.05, 0.05)
               for name in ("query", "key", "value"))
    cumulative = [None if inp[name] is None else torch.tensor(inp[name], dtype=torch.int32, device=device)
                  for name in ("cum_seq_q", "cum_seq_k")]
    args = (q, k, v, *cumulative, inp["max_q"], inp["max_k"], inp["dropout_p"],
            inp["is_causal"], inp["return_debug_mask"])
    kwargs = {"scale": inp["scale"]}
    for name in ("window_size_left", "window_size_right", "seqused_k", "alibi_slopes"):
        if inp[name] is not None:
            if name in ("seqused_k", "alibi_slopes"):
                raise ValueError("Frozen core catalog does not contain " + name)
            kwargs[name] = inp[name]
    return args, kwargs


def _tensor(shape, dtype, values=None):
    result = dict(shape=list(shape), dtype=str(dtype),
                  stride=[math.prod(shape[i + 1:]) for i in range(len(shape))],
                  requires_grad=False)
    if values is not None:
        result["values"] = values
    return result


def config(binding, dtype):
    recipe = _recipe(binding)
    inp = recipe["inputs"]
    args = [_tensor(inp[name]["shape"], dtype) for name in ("query", "key", "value")]
    args.extend(None if inp[name] is None else _tensor([len(inp[name])], torch.int32, inp[name])
                for name in ("cum_seq_q", "cum_seq_k"))
    args.extend(inp[name] for name in ("max_q", "max_k", "dropout_p", "is_causal", "return_debug_mask"))
    kwargs = {"scale": inp["scale"]}
    kwargs.update({name: inp[name] for name in ("window_size_left", "window_size_right")
                   if inp[name] is not None})
    return dict(inputs=dict(args=args, kwargs=kwargs), source_commit=_CATALOG["source_commit"],
                benchmark=recipe["benchmark"], scope="core", variant=recipe["ordinal"],
                case_id=recipe["case_ids"][str(dtype)], case_shape=copy.deepcopy(recipe["shape"]),
                case_params=copy.deepcopy(recipe["params"]),
                reference_implementation="pytorch_native_operator")
