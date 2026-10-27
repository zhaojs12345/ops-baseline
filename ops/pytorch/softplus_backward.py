# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Native backward recipes from 9d57a23b4386bd669152a935fe88a96db907da7c.

The default retains that benchmark's prepared autograd backward timer. Explicit
current profiles preserve the Enflame main and generic direct core input recipes
at 2ff0931a646b97ec98ebd50642b5889619125a6b. Forward, cloning and random-gradient
setup occur outside the timed native callable.
"""

import copy
import os

import torch

from _kernelgen import generate_tensor_input

OP_NAME = "softplus_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False
LEGACY_SOURCE = "9d57a23b4386bd669152a935fe88a96db907da7c"
CURRENT_SOURCE = "2ff0931a646b97ec98ebd50642b5889619125a6b"
SELECTOR = "OPS_BASELINE_SOFTPLUS_BACKWARD_CORE_PROFILE"
_selected = os.environ.get(SELECTOR)
if _selected not in (None, "legacy", "main", "direct"):
    raise ValueError(f"{SELECTOR} must be legacy, main or direct")
PROFILE = _selected or "legacy"
_COMMON_SHAPES = (
    (1073741824,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 1024),
)
_MAIN_SHAPES = (
    (536870912,),
    (64, 64),
    (4096, 4096),
    (64, 512, 512),
    (1024, 1024, 512),
)
# Source-text hashes identify the actual input/setup methods. They do not claim
# that source files at different commits are otherwise identical.
_SOURCE_BINDINGS = {
    "direct": {
        "direct_builder": "520400ce881f6b15ed6cbd26c5c5ca8011e7c23d179a885670b5a0de2d04e4ea",
        "direct_casefn": "608d3a609644e739d87402a137d6d98e99a6e11dcf0c3cc70f700c1a8e817957",
        "generic_framework_source_sha256": "ba10830841e14f4bc59460a3dd59eab51a00ada5a0e1593299ff092f00fcedd5",
    },
    "legacy": {
        "consts_sha256": "004daeb8f201f67da3c8bac6a87c8deb71a030d0382c4fa69f219738f673a8ec",
        "core_shapes_yaml_sha256": "720ad7776ce6e172e6c6eb78ee94b84b243ec7ee84a295543ac76fddd63b664d",
        "legacy_callable": "95acdf392196ee81ac1c822226f179f9cb434b014a48e2786d3390f13d4578da",
        "legacy_generator": "67b686d91b6ec438eea768c81b2ae154182be8a52a9939d0c57af8c9aa10a79b",
        "legacy_iterator": "888387394050427be55becf89b72d8aed4cbc9558b9e9039d4b4fd25666faac4",
        "legacy_unpack": "3b1a1bbd45f98219a8b2126f367fa67e8b7075f29cf5a12a3c7ebeba72904e8c",
    },
    "main": {
        "adapter_source_snapshot": "4a0724463fe32c5fd5641413c5f7ad8b02ca8fa7",
        "backend_core_shapes_sha256": "69f31cd4f11f227449f7b57fad62f31dbd28730c19688c28cd127053e5f62ff1",
        "current_adapter_builder": "50844457a2bfabed3f3d2bd842c794f13da30b70bb66ca7f7ec983a1a25a3a8a",
        "current_adapter_callable": "94cafb82b3a9ef4635c3502d7239c82f7a0b31f4c064c1a2f443db2a8fd83a07",
        "current_adapter_dout": "fcd15ba4d207b3aed0d380003fdf7c1dfab56494986750d4baa6b53354813b11",
        "current_base_callable": "95acdf392196ee81ac1c822226f179f9cb434b014a48e2786d3390f13d4578da",
        "current_builder": "93625f7fbccd258d315a7148b445bb25bec81d3c868c899bab82938cf05198b8",
        "current_generator": "67b686d91b6ec438eea768c81b2ae154182be8a52a9939d0c57af8c9aa10a79b",
        "current_unpack": "3b1a1bbd45f98219a8b2126f367fa67e8b7075f29cf5a12a3c7ebeba72904e8c",
    },
}


def _shape(binding):
    if set(binding) != {"shape", "variant"}:
        raise ValueError("Unexpected softplus core binding fields")
    if type(binding["variant"]) is not int or binding["variant"] != 0:
        raise ValueError("Only the original single variant is supported")
    shape = tuple(binding["shape"])
    if shape not in (_MAIN_SHAPES if PROFILE == "main" else _COMMON_SHAPES):
        raise ValueError("Unrecognized original softplus core descriptor")
    return shape


def _prepared_autograd_backward(out, inp, grad):
    return torch.autograd.grad((out,), (inp,), grad_outputs=(grad,), retain_graph=True)


def native():
    if PROFILE == "direct":
        return torch.ops.aten.softplus_backward.default
    return _prepared_autograd_backward


def grid():
    shapes = _MAIN_SHAPES if PROFILE == "main" else _COMMON_SHAPES
    return [{"shape": list(shape), "variant": 0} for shape in shapes]


def key_shape(binding):
    return list(_shape(binding))


def _autograd_inputs(shape, dtype, device):
    # The original base.unpacked floating input is a fresh requires-grad leaf.
    inp = generate_tensor_input(shape, dtype, device).clone().requires_grad_()
    out = torch.nn.functional.softplus(inp)
    grad = torch.randn_like(out)
    if not inp.is_leaf or not out.requires_grad or out.grad_fn is None:
        raise ValueError("Native backward profile requires the original forward graph")
    return (out, inp, grad), {}


def build_inputs(binding, dtype, device):
    shape = _shape(binding)
    if dtype not in DTYPES:
        raise ValueError("Only the original three floating dtypes are supported")
    if PROFILE == "direct":
        torch.manual_seed(0)
        grad = generate_tensor_input(shape, dtype, device)
        inp = generate_tensor_input(shape, dtype, device)
        return (grad, inp, 1.5, 20.0), {}
    return _autograd_inputs(shape, dtype, device)


def _metadata(value):
    if not isinstance(value, torch.Tensor):
        if value is None or type(value) in (int, float, bool, str):
            return value
        raise TypeError("Unsupported native argument metadata")
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "stride": list(value.stride()),
        "requires_grad": value.requires_grad,
    }


def config(binding, dtype):
    shape = _shape(binding)
    if dtype not in DTYPES:
        raise ValueError("Only the original three floating dtypes are supported")
    if PROFILE == "direct":
        # Metadata must not reset process RNG merely to describe a seeded case.
        args = tuple(torch.empty(shape, dtype=dtype, device="meta") for _ in range(2))
        args = (*args, 1.5, 20.0)
        kwargs = {}
        timed = {"args": [_metadata(t) for t in args], "kwargs": kwargs}
        logical = copy.deepcopy(timed)
        result = {
            "inputs": copy.deepcopy(timed),
            "native_timed_inputs": timed,
            "logical_case_inputs": logical,
            "native_timer_contract": {
                "kind": "direct_aten_backward",
                "native_schema": "aten::softplus_backward.default",
                "beta": 1.5,
                "threshold": 20.0,
                "returns": {"container": "Tensor", "metadata": _metadata(args[1])},
                "setup_timed": False,
            },
            "random_policy": "source_manual_seed0_then_randn_grad_then_randn_self",
            "input_recipe": "manual_seed(0); generate(grad_output); generate(self)",
            "benchmark": "benchmark/test_softplus_backward.py",
        }
    else:
        args, kwargs = _autograd_inputs(shape, dtype, "meta")
        out, inp, grad = args
        timed = {"args": [_metadata(t) for t in args], "kwargs": kwargs}
        logical = {"args": [_metadata(inp)], "kwargs": {}}
        result = {
            # Default remains the existing generic producer ABI contract. Only
            # the explicitly versioned current main profile maps the one-input
            # FlagGems case to the fully described prepared native three-input
            # timer; it never represents the native timer as a one-argument op.
            "inputs": copy.deepcopy(logical if PROFILE == "main" else timed),
            "native_timed_inputs": timed,
            "logical_case_inputs": logical,
            "native_timer_contract": {
                "kind": "prepared_autograd_backward",
                "forward": "torch.nn.functional.softplus",
                "backward": "torch.autograd.grad",
                "argument_order": ["forward_output", "self", "grad_output"],
                "beta": 1,
                "threshold": 20,
                "retain_graph": True,
                "create_graph": False,
                "returns": {"container": "tuple", "items": [_metadata(grad)]},
                "graph": {
                    "self_is_leaf": inp.is_leaf,
                    "forward_output_is_leaf": out.is_leaf,
                    "forward_output_has_grad_fn": out.grad_fn is not None,
                    "forward_output_created_from_this_self": True,
                },
                "setup_timed": False,
            },
            "random_policy": "source_unseeded_global_generator_no_reset",
            "input_recipe": "generate(self); clone.requires_grad; softplus; randn_like(out)",
            "benchmark": "benchmark/test_softplus.py",
        }
    result.update(
        source_commit=LEGACY_SOURCE if PROFILE == "legacy" else CURRENT_SOURCE,
        source_test="test_softplus_backward",
        descriptor=list(shape),
        scope="core",
        variant=0,
        grid_contract=f"softplus_backward.{PROFILE}.core/v1",
        input_contract=(
            "softplus-backward-main-logical-self-to-prepared-native/v1"
            if PROFILE == "main"
            else (
                "softplus-backward-direct/v1"
                if PROFILE == "direct"
                else "softplus-backward-legacy-native-prepared/v1"
            )
        ),
        producer_selector={} if PROFILE == "legacy" else {SELECTOR: PROFILE},
        source_bindings=copy.deepcopy(_SOURCE_BINDINGS[PROFILE]),
    )
    return result
