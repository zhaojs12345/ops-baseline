"""Native ATen composite matching existing dense dropout0 nine-tuple semantics.

Not a timing of the literal PrivateUse1 overrideable stub. Efficient CUDA
attention supplies math; natural LSE is converted to existing API log2 LSE.
"""

import torch

OP_NAME = "_scaled_dot_product_fused_attention_overrideable"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
_SHAPES = [
    (4, 32, 1024, 64),
    (4, 32, 1024, 128),
    (4, 32, 2048, 128),
    (4, 32, 4096, 128),
    (4, 32, 8192, 128),
]
LOG2E = 1.4426950408889634


def native_composite(
    query,
    key,
    value,
    attn_bias=None,
    dropout_p=0.0,
    is_causal=False,
    return_debug_mask=False,
    *,
    scale=None
):
    assert attn_bias is None and dropout_p == 0.0 and return_debug_mask is False
    assert scale is None and query.shape == key.shape == value.shape
    out, lse, _, _ = torch.ops.aten._scaled_dot_product_efficient_attention.default(
        query, key, value, None, True, 0.0, is_causal
    )
    batch, _, length, _ = query.shape
    return (
        out.contiguous(),
        (lse[..., :length] * LOG2E).contiguous(),
        torch.tensor(
            [0, length] * batch, dtype=torch.int32, device=query.device
        ).reshape(batch, 2),
        torch.tensor([0, length] * batch, dtype=torch.int32, device=key.device).reshape(
            batch, 2
        ),
        length,
        length,
        torch.tensor([0], dtype=torch.int64, device=query.device),
        torch.tensor([0], dtype=torch.int64, device=query.device),
        torch.empty(0, dtype=query.dtype, device=query.device),
    )


def native():
    return native_composite


def grid():
    return [
        {"shape": list(shape), "is_causal": causal}
        for causal in (False, True)
        for shape in _SHAPES
    ]


def build_inputs(binding, dtype, device):
    shape = tuple(binding["shape"])
    query, key, value = (
        torch.randn(shape, dtype=dtype, device=device) for _ in range(3)
    )
    return (query, key, value, None, 0.0, binding["is_causal"], False), {"scale": None}


def key_shape(binding):
    return dict(shape=list(binding["shape"]), is_causal=binding["is_causal"])


def config(binding, dtype):
    shape = list(binding["shape"])
    batch, heads, length, _ = shape
    stride = [1] * 4
    for i in range(2, -1, -1):
        stride[i] = stride[i + 1] * shape[i + 1]
    tensor = dict(
        shape=shape,
        stride=stride,
        storage_offset=0,
        dtype=str(dtype),
        requires_grad=False,
    )
    return {
        "native_expression": "native_ATen_efficient_attention_dense_dropout0_nine_tuple",
        "literal_overrideable_stub": False,
        "math_API": "aten::_scaled_dot_product_efficient_attention",
        "logsumexp_convention": "log2 = native natural LSE * LOG2E; trim padding; contiguous",
        "metadata_recipe": "existing dense p0 API cum_seq[0,L]*B int32, maxQ=maxK=L, seed/offset target int64[0], debugempty",
        "source_commit": "2a3a455fa9eef530dc11a6b96ff333ffb47235fd",
        "source_benchmark_sha256": "47e8b5cc73898f03a300e1dce5892c1333d3eab3cd26a40490a8f13e11241799",
        "source_builder_ast_sha256": "43f272587d2de9c060cac48a2ef2bd50e4b9557632f37da2f073c3c37326f5e8",
        "benchmark": "benchmark/test_scaled_dot_product_fused_attention_overrideable.py",
        "builder": "_build_inputs_fn",
        "scope": "core",
        "descriptor": binding,
        "recipe": {
            "query": "torch.randn",
            "key": "torch.randn",
            "value": "torch.randn",
            "on_requested_device": True,
            "seed": "original unseeded builder",
        },
        "inputs": {
            "args": [
                tensor.copy(),
                tensor.copy(),
                tensor.copy(),
                None,
                0.0,
                binding["is_causal"],
                False,
            ],
            "kwargs": {"scale": None},
        },
        "outputs": [
            {"name": "output", "shape": shape, "dtype": str(dtype)},
            {
                "name": "logsumexp",
                "shape": [batch, heads, length],
                "dtype": "torch.float32",
                "base": 2,
            },
            {"name": "cum_seq_q", "shape": [batch, 2], "dtype": "torch.int32"},
            {"name": "cum_seq_k", "shape": [batch, 2], "dtype": "torch.int32"},
            {"name": "max_q", "value": length},
            {"name": "max_k", "value": length},
            {"name": "philox_seed", "shape": [1], "dtype": "torch.int64"},
            {"name": "philox_offset", "shape": [1], "dtype": "torch.int64"},
            {"name": "debug_attn_mask", "shape": [0], "dtype": str(dtype)},
        ],
        "effects": {"mutates": [], "returns_alias_of": {}},
    }
