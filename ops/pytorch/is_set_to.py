# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native host-metadata contract for aten::is_set_to.

Use tools/collect_baseline_host.py with the original FlagGems catalog. This
module intentionally has no GPU/NCU recipe: the native predicate reads storage
identity and view metadata and launches no device computation.
"""

import torch

OP_NAME = "is_set_to"
HOST_ONLY = True


def native():
    return torch.ops.aten.is_set_to


def view_contract(args, kwargs):
    if kwargs or len(args) != 2 or any(not isinstance(x, torch.Tensor) for x in args):
        raise ValueError("is_set_to host catalog requires two positional tensors")
    a, b = args
    if any(x.layout != torch.strided for x in args):
        raise ValueError("is_set_to host catalog requires dense views")
    return {
        "same_object": a is b,
        "same_storage": a.untyped_storage()._cdata == b.untyped_storage()._cdata,
        "storage_offsets": [x.storage_offset() for x in args],
        "storage_nbytes": [x.untyped_storage().nbytes() for x in args],
        "layouts": [str(x.layout) for x in args],
        "conjugate": [x.is_conj() for x in args],
        "negative": [x.is_neg() for x in args],
    }


def expected_result(args, kwargs):
    config = view_contract(args, kwargs)
    a, b = args
    return bool(
        config["same_storage"]
        and not any(config["conjugate"] + config["negative"])
        and a.storage_offset() == b.storage_offset()
        and a.size() == b.size()
        and a.stride() == b.stride()
    )


def validate_native(op, args, kwargs):
    if op is not native():
        raise ValueError("is_set_to host collection must time the native ATen API")
    result = op(*args, **kwargs)
    if type(result) is not bool or result is not expected_result(args, kwargs):
        raise ValueError("Native is_set_to did not match its storage-view contract")
    return result
