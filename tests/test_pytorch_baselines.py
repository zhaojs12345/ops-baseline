# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Contract checks over every added recipe; no CUDA or FlagGems installation needed."""

import importlib.util
import json
import os
import random
import subprocess
import tempfile
from pathlib import Path
import sys
import unittest

import torch

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops/pytorch"
sys.path.insert(0, str(OPS))
from _kernelgen import (
    generate_tensor_input,
    unpack_inputs,
    repeat_interleave_repeats,
)  # noqa: E402


def describe(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, dict):
        return {
            key: "cuda" if key == "device" and child == "meta" else describe(child)
            for key, child in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [describe(child) for child in value]
    if isinstance(value, torch.dtype):
        return str(value)
    return value


class PytorchCatalogTests(unittest.TestCase):
    @unittest.skipUnless(
        os.environ.get("FLAGGEMS_SOURCE_REPO"),
        "Set FLAGGEMS_SOURCE_REPO for pinned source integration",
    )
    def test_nvidia_cases_match_pinned_source(self):
        names = [
            name
            for name, info in self.manifest["operators"].items()
            if info.get("source_backend") == "nvidia"
        ]
        with tempfile.TemporaryDirectory() as directory:
            subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "tools/compare_pytorch_source_cases.py"),
                    "--source-repo",
                    os.environ["FLAGGEMS_SOURCE_REPO"],
                    "--ops",
                    ",".join(names),
                    "--output",
                    str(Path(directory) / "comparison.json"),
                ],
                check=True,
            )

    def test_half_fft_retains_only_source_power_of_two_transforms(self):
        full = self.modules["fft_rfft"]
        half = self.modules["fft_rfft_float16"]
        self.assertEqual(half.DTYPES, [torch.float16])
        self.assertEqual(full.DTYPES, [torch.float32, torch.float64])
        for binding in half.grid():
            args, kwargs = half.build_inputs(binding, torch.float16, "meta")
            length = kwargs.get("n", args[0].shape[kwargs.get("dim", -1)])
            self.assertGreater(length, 0)
            self.assertEqual(length & (length - 1), 0)

    def test_repeat_interleave_repeats_preserve_rng_and_source_contract(self):
        for name in ("repeat_interleave_tensor", "repeat_interleave_self_tensor"):
            for binding in self.modules[name].grid():
                state = torch.random.get_rng_state()
                repeats = repeat_interleave_repeats(
                    binding["shape"], torch.int32, "cpu"
                )
                self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
                self.assertEqual(repeats.dtype, torch.int64)
                self.assertEqual(list(repeats.shape), [binding["shape"][0]])
                self.assertTrue(bool(((repeats >= 0) & (repeats < 31)).all()))

    def test_repeat_interleave_workload_replays_across_processes(self):
        script = """
import torch, sys, json, importlib
sys.path.insert(0, sys.argv[1])
torch.manual_seed(int(sys.argv[2]))
result = {}
for name in ("repeat_interleave_tensor", "repeat_interleave_self_tensor"):
    module = importlib.import_module(name)
    args, kwargs = module.build_inputs(module.grid()[0], module.DTYPES[0], "cpu")
    repeats = args[0 if name == "repeat_interleave_tensor" else 1]
    output = module.native()(*args, **kwargs)
    result[name] = [repeats.tolist(), list(output.shape)]
print(json.dumps(result))
"""
        outputs = [
            json.loads(
                subprocess.check_output(
                    [sys.executable, "-c", script, str(OPS), str(seed)], text=True
                )
            )
            for seed in (123, 456)
        ]
        self.assertEqual(outputs[0], outputs[1])

    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OPS / "_kernelgen/manifest.json").read_text())
        cls.cases = json.loads((OPS / "_kernelgen/cases.json").read_text())
        cls.modules = {}
        for name, info in cls.manifest["operators"].items():
            path = OPS / (info.get("module", name) + ".py")
            spec = importlib.util.spec_from_file_location("baseline_test_" + name, path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            cls.modules[name] = module

    def test_inventory_and_source_provenance_match_files(self):
        names = set(self.modules)
        self.assertEqual(names, set(self.cases))
        expected = {
            info.get("module", name)
            for name, info in self.manifest["operators"].items()
        } | {"silu"}
        self.assertEqual(expected, {p.stem for p in OPS.glob("*.py")})
        coverage = json.loads((OPS / "coverage.json").read_text())
        self.assertEqual(coverage["summary"]["added_operators"], len(names))
        for name, module in self.modules.items():
            with self.subTest(op=name):
                info = self.manifest["operators"][name]
                self.assertEqual(module.OP_NAME, name)
                self.assertFalse(module.IS_INPLACE)
                self.assertEqual([str(d) for d in module.DTYPES], info["dtypes"])
                self.assertIn(self.manifest["source_commit"], module.__doc__)
                self.assertTrue(callable(module.native()))

    def test_keys_are_unique_and_ncu_filenames_fit(self):
        for name, module in self.modules.items():
            with self.subTest(op=name):
                bindings = module.grid()
                self.assertEqual(
                    len(bindings), self.manifest["operators"][name]["cases"]
                )
                keys = [str(module.key_shape(b)) for b in bindings]
                self.assertEqual(len(set(keys)), len(keys))
                for b in bindings:
                    key = module.key_shape(b)
                    tag = f"{name}_{'x'.join(map(str, key))}_complex128.ncu-rep"
                    self.assertLessEqual(len(tag.encode()), 255)

    def test_all_source_descriptors_dtypes_and_variants_reconstruct(self):
        for name, module in self.modules.items():
            for binding in module.grid():
                for dtype in module.DTYPES:
                    with self.subTest(op=name, binding=binding, dtype=dtype):
                        before = json.loads(json.dumps(binding))
                        args, kwargs = module.build_inputs(binding, dtype, "meta")
                        self.assertIsInstance(args, tuple)
                        self.assertIsInstance(kwargs, dict)
                        config = module.config(binding, dtype)
                        self.assertEqual(
                            describe({"args": args, "kwargs": kwargs}), config["inputs"]
                        )
                        self.assertEqual(
                            config["source_commit"], self.manifest["source_commit"]
                        )
                        self.assertEqual(binding, before)
                        json.dumps(config)

    def test_bindings_and_config_are_not_shared_mutable_state(self):
        module = self.modules["mm"]
        original = module.grid()[0]
        changed = module.grid()[0]
        changed["shape"][0] = -123
        self.assertEqual(module.grid()[0], original)
        config = module.config(original, module.DTYPES[0])
        config["inputs"]["args"][0]["shape"][0] = -123
        self.assertNotEqual(module.config(original, module.DTYPES[0]), config)

    def test_fixed_geometry_is_listed_once_per_source_case(self):
        for name, info in self.manifest["operators"].items():
            if not info["shape_source"].startswith("fixed get_input_iter"):
                continue
            module = self.modules[name]
            variants = [b["variant"] for b in module.grid()]
            self.assertEqual(len(variants), len(set(variants)))
        pool = self.modules["max_pool3d"]
        self.assertEqual(pool.grid()[0]["shape"], [4, 3, 16, 56, 56])
        self.assertEqual(len(pool.grid()), 4)

    def test_out_buffer_and_boolean_dtype_semantics(self):
        module = self.modules["exp_out"]
        args, kwargs = module.build_inputs(
            {"shape": [4, 8], "variant": 0}, torch.float32, "cpu"
        )
        output = module.native()(*args, **kwargs)
        self.assertIs(output, kwargs["out"])
        torch.testing.assert_close(output, torch.exp(args[0]))
        boolean = generate_tensor_input((32,), torch.bool, "cpu")
        self.assertEqual(boolean.dtype, torch.bool)
        self.assertEqual(boolean.shape, (32,))

    def test_custom_constructor_rejections_are_fully_classified(self):
        coverage = json.loads((OPS / "coverage.json").read_text())
        review = coverage["constructor_review"]
        self.assertEqual(review["reviewed_records"], 78)
        self.assertEqual(len(review["results"]), 78)
        self.assertEqual(
            review["implemented_records"] + review["remaining_records"], 78
        )
        for row in coverage["pending"]:
            self.assertNotIn("custom __init__ needs manual adaptation", row["reason"])

    def test_concat_and_stack_constructor_inputs_match_native(self):
        for name in (
            "concat",
            "cat",
            "stack",
            "column_stack",
            "row_stack",
            "vstack",
            "hstack",
        ):
            module = self.modules[name]
            args, kwargs = module.build_inputs(
                {"shape": [4, 8], "variant": 0}, torch.float32, "cpu"
            )
            self.assertEqual(len(args), 1)
            self.assertEqual(len(args[0]), 3)
            self.assertTrue(all(t.shape == (4, 8) for t in args[0]))
            actual = module.native()(*args, **kwargs)
            reference = getattr(torch, "cat" if name == "concat" else name)(
                *args, **kwargs
            )
            torch.testing.assert_close(actual, reference)

    def test_source_stack_memory_filters_are_preserved(self):
        for name in ("cat", "column_stack", "row_stack", "stack", "_stack", "vstack"):
            self.assertTrue(
                all(
                    __import__("math").prod(b["shape"]) <= 2**29
                    for b in self.modules[name].grid()
                )
            )
        self.assertEqual(
            [b["shape"] for b in self.modules["concat"].grid()],
            [[64, 64], [256, 256], [512, 512], [512, 1024], [512, 2048]],
        )

    def test_multi_margin_constructor_parameter_product_and_output_shapes(self):
        for name in (
            "multi_margin_loss",
            "multi_margin_loss_out",
            "multi_margin_loss_backward",
            "multi_margin_loss_backward_out",
        ):
            module = self.modules[name]
            seen = set()
            for variant in range(12):
                args, kwargs = module.build_inputs(
                    {"shape": [3, 5], "variant": variant}, torch.float32, "cpu"
                )
                backward = "backward" in name
                offset = 1 if backward else 0
                seen.add(
                    (args[offset + 2], args[offset + 4] is not None, args[offset + 5])
                )
                reduction = args[offset + 5]
                expected = (3,) if reduction == 0 else ()
                if backward:
                    self.assertEqual(args[0].shape, expected)
                if name.endswith("_out"):
                    self.assertEqual(
                        next(iter(kwargs.values())).shape,
                        (3, 5) if backward else expected,
                    )
            self.assertEqual(len(seen), 12)

    def test_paired_quantization_factory_uses_native_arguments_only(self):
        module = self.modules["convert_weight_to_int4pack"]
        for variant, inner_tiles in enumerate((2, 4, 8)):
            args, kwargs = module.build_inputs(
                {"shape": [16, 64], "variant": variant}, torch.uint8, "cpu"
            )
            self.assertEqual(len(args), 2)
            self.assertIsInstance(args[0], torch.Tensor)
            self.assertEqual(args[0].dtype, torch.uint8)
            self.assertEqual(args[0].shape, (16, 64))
            self.assertEqual(args[1], inner_tiles)

    def test_parameterized_einsum_and_copy_source_dtype_are_not_lost(self):
        module = self.modules["einsum"]
        for variant, equation in enumerate(("ij,jk->ik", "bij,bjk->bik")):
            args, kwargs = module.build_inputs(
                {"shape": [2, 4, 5, 3], "variant": variant}, torch.float32, "cpu"
            )
            self.assertEqual(args[0], equation)
            result = module.native()(*args, **kwargs)
            self.assertEqual(result.shape, (4, 5) if variant == 0 else (2, 4, 5))
        for source in (
            "float32",
            "float16",
            "bfloat16",
            "int8",
            "int16",
            "int32",
            "int64",
        ):
            module = self.modules["to_copy_from_" + source]
            for target in module.DTYPES:
                args, kwargs = module.build_inputs(
                    {"shape": [4, 8], "variant": 0}, target, "cpu"
                )
                self.assertEqual(args[0].dtype, getattr(torch, source))
                self.assertEqual(kwargs["dtype"], target)
                self.assertNotEqual(args[0].dtype, target)

    def test_sparse_mask_replay_matches_recorded_identity_and_native_values(self):
        import hashlib

        module = self.modules["sparse_mask"]
        binding = module.grid()[0]
        state = torch.random.get_rng_state().clone()
        first = module._mask(binding["shape"])
        self.assertTrue(torch.equal(state, torch.random.get_rng_state()))
        second = module._mask(binding["shape"])
        self.assertTrue(torch.equal(first.indices(), second.indices()))
        config = module.config(binding, torch.float32)["mask"]
        self.assertEqual(first._nnz(), config["nnz"])
        self.assertEqual(hashlib.sha256(first.indices().numpy().tobytes()).hexdigest(),
                         config["indices_sha256"])
        args, kwargs = module.build_inputs(binding, torch.float32, "cpu")
        output = module.native()(*args, **kwargs).coalesce()
        self.assertTrue(torch.equal(output.indices(), first.indices()))
        torch.testing.assert_close(output.values(), args[0][tuple(first.indices())])

    def test_sparse_mask_rejects_changed_recorded_identity(self):
        from unittest.mock import patch

        module = self.modules["sparse_mask"]
        binding = module.grid()[0]
        for field, changed in (("nnz", -1), ("indices_sha256", "0" * 64)):
            config = module.config(binding, torch.float32)
            config["mask"][field] = changed
            with self.subTest(field=field), patch.object(module, "config", return_value=config):
                with self.assertRaisesRegex(ValueError, "Sparse mask replay differs"):
                    module.build_inputs(binding, torch.float32, "cpu")

    def test_sparse_mask_identity_replays_across_processes(self):
        script = """
import sys, json, hashlib, torch
sys.path.insert(0, sys.argv[1])
import sparse_mask
# Ambient RNG differs between the latency and NCU-style child processes.
torch.manual_seed(int(sys.argv[2]))
result = []
for binding in sparse_mask.grid():
    mask = sparse_mask._mask(binding["shape"])
    result.append([mask._nnz(), hashlib.sha256(mask.indices().numpy().tobytes()).hexdigest()])
print(json.dumps(result))
"""
        outputs = [subprocess.check_output(
            [sys.executable, "-c", script, str(OPS), str(seed)], text=True
        ) for seed in (123, 456)]
        self.assertEqual(outputs[0], outputs[1])
        expected = [self.modules["sparse_mask"].config(b, torch.float32)["mask"]
                    for b in self.modules["sparse_mask"].grid()]
        self.assertEqual(json.loads(outputs[0]),
                         [[row["nnz"], row["indices_sha256"]] for row in expected])

    def test_unpack_matches_upstream_keyword_convention(self):
        value = torch.ones(2)
        args, kwargs = unpack_inputs((value, 1, {"keepdim": True}))
        self.assertEqual(args, (value, 1))
        self.assertEqual(kwargs, {"keepdim": True})
        self.assertEqual(unpack_inputs(({"size": (4, 8)},)), ((), {"size": (4, 8)}))

    def test_random_scalar_parameters_replay_without_polluting_host_rng(self):
        module = self.modules["pad"]
        binding = {"shape": [8, 8], "variant": 0}
        cpu_state = torch.random.get_rng_state().clone()
        python_state = random.getstate()
        first = module.build_inputs(binding, torch.float32, "meta")[1]
        second = module.build_inputs(binding, torch.float32, "meta")[1]
        self.assertEqual(first, second)
        self.assertEqual(random.getstate(), python_state)
        self.assertTrue(torch.equal(torch.random.get_rng_state(), cpu_state))


if __name__ == "__main__":
    unittest.main()
