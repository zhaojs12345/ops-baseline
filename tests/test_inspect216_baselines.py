# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU/meta contracts for the explicit extensions, separate from frozen recipes."""

import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest import mock

import torch

ROOT = Path(__file__).resolve().parents[1]
OPS = ROOT / "ops/pytorch"
sys.path.insert(0, str(OPS))

# Exact new inventory. A stray module must still fail the directory inventory.
EXPECTED = {
    "_flash_attention_forward": (32, ["torch.float16", "torch.bfloat16"]),
    "_scaled_dot_product_flash_attention": (15, ["torch.float16", "torch.bfloat16"]),
    "addmm_": (10, ["torch.float16", "torch.float32", "torch.bfloat16"]),
    "baddbmm_": (10, ["torch.float16", "torch.float32", "torch.bfloat16"]),
    "linalg_polar": (4, ["torch.float32"]),
    "linalg_polar_out": (4, ["torch.float32"]),
    "pack_padded_sequence": (4, ["torch.float16", "torch.float32", "torch.bfloat16"]),
    "quantized_max_pool2d": (5, ["torch.qint8"]),
    "quantized_max_pool2d_out": (5, ["torch.qint8"]),
}


class Inspect216BaselineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest = json.loads((OPS / "_inspect216_manifest.json").read_text())
        cls.modules = {name: importlib.import_module(name) for name in EXPECTED}

    def assert_input_descriptor(self, actual, expected):
        if isinstance(actual, torch.Tensor):
            self.assertIsInstance(expected, dict)
            required = {"shape", "dtype", "stride", "requires_grad"}
            allowed = required | {
                "kind",
                "device",
                "values",
                "qscheme",
                "scale",
                "zero_point",
            }
            self.assertTrue(required <= set(expected) <= allowed)
            observed = {
                "shape": list(actual.shape),
                "dtype": str(actual.dtype),
                "stride": list(actual.stride()),
                "requires_grad": actual.requires_grad,
            }
            if "kind" in expected:
                observed["kind"] = "tensor"
            if "device" in expected:
                observed["device"] = str(actual.device)
            if "values" in expected:
                observed["values"] = actual.tolist()
            if "qscheme" in expected:
                self.assertTrue(actual.is_quantized)
                observed.update(
                    qscheme=str(actual.qscheme()),
                    scale=actual.q_scale(),
                    zero_point=actual.q_zero_point(),
                )
            self.assertEqual(observed, expected)
        elif isinstance(actual, (tuple, list)):
            self.assertIsInstance(expected, list)
            self.assertEqual(len(actual), len(expected))
            for value, descriptor in zip(actual, expected):
                self.assert_input_descriptor(value, descriptor)
        elif isinstance(actual, dict):
            self.assertEqual(set(actual), set(expected))
            for key in actual:
                self.assert_input_descriptor(actual[key], expected[key])
        else:
            self.assertEqual(actual, expected)

    def test_explicit_inventory_counts_and_closed_source_pins(self):
        self.assertEqual(set(self.manifest["operators"]), set(EXPECTED))
        legacy = json.loads((OPS / "_kernelgen/manifest.json").read_text())
        legacy_modules = {
            info.get("module", name) for name, info in legacy["operators"].items()
        }
        self.assertFalse(set(EXPECTED) & legacy_modules)
        self.assertIn("scaled_dot_product_flash_attention", legacy_modules)
        self.assertNotIn("scaled_dot_product_flash_attention", EXPECTED)
        self.assertEqual(
            self.manifest["summary"],
            {"operators": 9, "recipes": 89, "configurations": 184},
        )
        for name, (recipes, dtypes) in EXPECTED.items():
            with self.subTest(operator=name):
                info = self.manifest["operators"][name]
                module = self.modules[name]
                self.assertEqual(info["module"], module.OP_NAME)
                self.assertEqual(module.OP_NAME, name)
                self.assertEqual(info["recipes"], len(module.grid()))
                self.assertEqual(info["recipes"], recipes)
                self.assertEqual(info["dtypes"], [str(dt) for dt in module.DTYPES])
                self.assertEqual(info["dtypes"], dtypes)
                self.assertEqual(info["configurations"], recipes * len(dtypes))
                self.assertEqual(info["is_inplace"], module.IS_INPLACE)
                self.assertTrue(callable(module.native()))
                self.assertTrue(info["source_pins"])
                for commit, files in info["source_pins"].items():
                    self.assertRegex(commit, r"^[0-9a-f]{40}$")
                    self.assertTrue(files)
                    for path, digest in files.items():
                        self.assertTrue(
                            path.startswith(
                                ("benchmark/", "tests/", "definitions/", "src/")
                            )
                        )
                        self.assertRegex(digest, r"^[0-9a-f]{64}$")

    def test_all_184_configs_have_unique_keys_and_real_input_descriptions(self):
        identities, configs = set(), set()
        tensor = torch.tensor
        for name, module in self.modules.items():
            info = self.manifest["operators"][name]
            for binding in module.grid():
                for dtype in module.DTYPES:
                    with self.subTest(operator=name, binding=binding, dtype=dtype):
                        key = (name, str(module.key_shape(binding)), str(dtype))
                        self.assertNotIn(key, identities)
                        identities.add(key)
                        config = module.config(binding, dtype)
                        identity = (
                            name,
                            json.dumps(config, sort_keys=True, allow_nan=False),
                        )
                        self.assertNotIn(identity, configs)
                        configs.add(identity)
                        commit = config.get(
                            "source_commit", getattr(module, "SOURCE_COMMIT", None)
                        )
                        self.assertIn(commit, info["source_pins"])
                        if "benchmark" in config:
                            self.assertIn(
                                config["benchmark"], info["source_pins"][commit]
                            )
                        device = "cpu" if dtype == torch.qint8 else "meta"
                        requested_int_devices = []

                        def materialize_small_ints(values, **kwargs):
                            # Q/K/V stay meta even for the largest attention grid.
                            # The two tiny cumulative arrays retain real values;
                            # separately verify the builder requested assigned device.
                            requested_int_devices.append(kwargs["device"])
                            self.assertEqual(kwargs["device"], "meta")
                            self.assertEqual(kwargs["dtype"], torch.int32)
                            return tensor(values, **{**kwargs, "device": "cpu"})

                        with torch.random.fork_rng(devices=[]):
                            if name == "_flash_attention_forward":
                                with mock.patch.object(
                                    torch, "tensor", side_effect=materialize_small_ints
                                ):
                                    args, kwargs = module.build_inputs(
                                        binding, dtype, device
                                    )
                            else:
                                args, kwargs = module.build_inputs(
                                    binding, dtype, device
                                )
                        self.assertIsInstance(args, tuple)
                        self.assertIsInstance(kwargs, dict)
                        self.assert_input_descriptor(
                            {"args": args, "kwargs": kwargs}, config["inputs"]
                        )
                        if name == "_flash_attention_forward" and args[3] is not None:
                            self.assertEqual(requested_int_devices, ["meta", "meta"])
        self.assertEqual(len(identities), 184)
        self.assertEqual(len(configs), 184)

    def test_return_schemas_keep_five_output_flash_and_nine_output_sdpa_distinct(self):
        five = self.modules["_flash_attention_forward"].native()
        nine = self.modules["_scaled_dot_product_flash_attention"].native()
        self.assertEqual(five._schema.name, "aten::_flash_attention_forward")
        self.assertEqual(nine._schema.name, "aten::_scaled_dot_product_flash_attention")
        self.assertEqual(len(five._schema.returns), 5)
        self.assertEqual(len(nine._schema.returns), 9)
        self.assertIsNot(five, nine)

    def test_catalog_source_hashes_are_declared_and_all_recipe_ordinals_remain(self):
        for name, file, field in (
            (
                "_flash_attention_forward",
                "_inspect216_flash_forward_cases.json",
                "source_files_sha256",
            ),
            (
                "_scaled_dot_product_flash_attention",
                "_inspect216_sdpa_flash_cases.json",
                "source_files",
            ),
        ):
            catalog = json.loads((OPS / file).read_text())
            info = self.manifest["operators"][name]
            self.assertIn(catalog["source_commit"], info["source_pins"])
            for path, digest in catalog[field].items():
                self.assertEqual(
                    info["source_pins"][catalog["source_commit"]][path], digest
                )
            groups = {}
            for recipe in catalog["recipes"]:
                commit = recipe.get("source_commit", catalog["source_commit"])
                self.assertIn(recipe["benchmark"], info["source_pins"][commit])
                backend = recipe.get("backend_recipe", "original")
                groups.setdefault((commit, recipe["benchmark"], backend), []).append(
                    recipe["ordinal"]
                )
            for ordinals in groups.values():
                self.assertEqual(sorted(ordinals), list(range(len(ordinals))))

    def test_configs_do_not_mutate_frozen_recipes(self):
        for name, module in self.modules.items():
            with self.subTest(operator=name):
                binding = module.grid()[0]
                config = module.config(binding, module.DTYPES[0])
                config["inputs"]["args"][0]["shape"][0] = -123
                self.assertNotEqual(config, module.config(binding, module.DTYPES[0]))

    def test_inplace_native_keeps_beta_one_alias_and_column_major_operand(self):
        for name in ("addmm_", "baddbmm_"):
            for column_major in (False, True):
                with self.subTest(operator=name, column_major=column_major):
                    batch = (2,) if name == "baddbmm_" else ()
                    inp = torch.full((*batch, 3, 4), 3.0)
                    left = torch.ones(*batch, 3, 5)
                    right = torch.ones(*batch, 4, 5).transpose(-1, -2)
                    if not column_major:
                        right = right.contiguous()
                    saved_left, saved_right = left.clone(), right.clone()
                    result = self.modules[name].native()(inp, left, right)
                    self.assertIs(result, inp)
                    self.assertTrue(torch.equal(result, torch.full_like(result, 8)))
                    self.assertTrue(torch.equal(left, saved_left))
                    self.assertTrue(torch.equal(right, saved_right))
                    self.assertEqual(right.stride()[-2:] == (1, 5), column_major)

    def test_fresh_input_protocol_is_detectable_and_restores_all_operands(self):
        timing = importlib.import_module("_inspect216_inplace")
        for dtype in (torch.float16, torch.float32, torch.bfloat16):
            for column_major in (False, True):
                with self.subTest(dtype=dtype, column_major=column_major):
                    inp = torch.full((3, 4), 3.0, dtype=dtype)
                    left = torch.ones(3, 5, dtype=dtype)
                    right = torch.ones(4, 5, dtype=dtype).t()
                    if not column_major:
                        right = right.contiguous()
                    original = (inp, left, right)
                    saved = tuple(
                        x.clone(memory_format=torch.preserve_format) for x in original
                    )
                    calls = []

                    def watched(*values):
                        for value, expected in zip(values, saved):
                            self.assertTrue(torch.equal(value, expected))
                            self.assertEqual(value.stride(), expected.stride())
                            self.assertNotEqual(value.data_ptr(), expected.data_ptr())
                        values[0].add_(5)
                        values[1].zero_()
                        values[2].zero_()
                        calls.append(1)

                    def five_calls(operation, prepare):
                        for _ in range(5):
                            args, kwargs = prepare()
                            operation(*args, **kwargs)
                        return 2.0

                    with mock.patch.object(
                        timing, "event_bench", side_effect=five_calls
                    ):
                        self.assertEqual(
                            timing.measure_latency(watched, original, {}), 2.0
                        )
                    self.assertEqual(len(calls), 5)
                    self.assertTrue(
                        all(torch.equal(a, b) for a, b in zip(original, saved))
                    )
                    # A shared snapshot is a red control: the second call must
                    # be rejected by the same guard instead of accumulating.
                    bad = tuple(
                        x.clone(memory_format=torch.preserve_format) for x in saved
                    )
                    watched(*bad)
                    with self.assertRaises(AssertionError):
                        watched(*bad)

    def event_runtime(self, latencies):
        clock = [0.0]
        events, samples = [], []
        sequence = iter(latencies)
        test = self

        class Event:
            def __init__(self, start):
                self.start = start

            def record(self):
                events.append("start" if self.start else "end")
                self.time = clock[0]

            def synchronize(self):
                events.append("end_sync")

            def elapsed_time(self, other):
                test.assertEqual(other.time - self.time, 2.0)
                value = next(sequence, 2.0)
                samples.append(value)
                return value

        cuda = SimpleNamespace()
        created = []

        def make_event(**kwargs):
            self.assertEqual(kwargs, {"enable_timing": True})
            event = Event(not created)
            created.append(event)
            return event

        def sync():
            events.append("sync")

        def empty(*args, **kwargs):
            self.assertEqual(args, (256_000_000,))
            self.assertEqual(kwargs, {"dtype": torch.int8, "device": "cuda"})

            def flush():
                clock[0] += 50
                events.append("flush")

            return SimpleNamespace(zero_=flush)

        cuda.Event, cuda.synchronize = make_event, sync

        def prepare():
            clock[0] += 100
            events.append("prepare")
            return (1,), {}

        def operation(value):
            self.assertEqual(value, 1)
            clock[0] += 2
            events.append("operation")

        fake = SimpleNamespace(empty=empty, int8=torch.int8, cuda=cuda)
        return fake, prepare, operation, events, samples

    def test_event_timer_excludes_slow_preparation_and_flush(self):
        timing = importlib.import_module("_inspect216_inplace")
        fake, prepare, operation, events, samples = self.event_runtime([])
        with mock.patch.object(timing, "torch", fake):
            latency = timing.event_bench(operation, prepare, warmup_ms=0, rep_ms=0.01)
        self.assertEqual(latency, 2.0)
        self.assertEqual(len(samples), 11)
        self.assertEqual(events[:3], ["prepare", "operation", "sync"])
        interval = ["prepare", "flush", "sync", "start", "operation", "end", "end_sync"]
        self.assertEqual(events[3:], interval * 11)

    def test_event_timer_rejects_invalid_calibration_and_samples(self):
        timing = importlib.import_module("_inspect216_inplace")
        for latencies in (
            [0.0] * 5,
            [-1.0] * 5,
            [float("nan")] * 5,
            [float("inf")] * 5,
            [2.0] * 5 + [float("nan")] * 5,
        ):
            with self.subTest(latencies=latencies):
                fake, prepare, operation, _, _ = self.event_runtime(latencies)
                with (
                    mock.patch.object(timing, "torch", fake),
                    self.assertRaises(RuntimeError),
                ):
                    timing.event_bench(operation, prepare, warmup_ms=0, rep_ms=0.01)

    def test_polar_cpu_composition_and_out_write_contract(self):
        torch.manual_seed(3)
        inp = torch.randn(8, 4)
        saved = inp.clone()
        expected_u, expected_h = self.modules["linalg_polar"].native()(inp)
        eigenvalues, vectors = torch.linalg.eigh(inp.double().mT @ inp.double())
        truth_h = (vectors * eigenvalues.clamp_min(0).sqrt().unsqueeze(-2)) @ vectors.mT
        torch.testing.assert_close(expected_h.double(), truth_h, atol=1e-4, rtol=1e-4)
        torch.testing.assert_close(expected_u @ expected_h, inp, atol=1e-4, rtol=1e-4)
        out_u, out_h = torch.empty_like(inp), torch.empty(4, 4)
        actual_u, actual_h = self.modules["linalg_polar_out"].native()(
            inp, U=out_u, H=out_h
        )
        self.assertIs(actual_u, out_u)
        self.assertIs(actual_h, out_h)
        torch.testing.assert_close(actual_u, expected_u)
        torch.testing.assert_close(actual_h, expected_h)
        self.assertTrue(torch.equal(inp, saved))

    def test_small_packing_preserves_host_batch_sizes(self):
        inp = torch.arange(4 * 3 * 2).reshape(4, 3, 2).float()
        lengths = torch.tensor([4, 2, 1], dtype=torch.int64)
        actual, counts = self.modules["pack_padded_sequence"].native()(
            inp, lengths, False
        )
        self.assertEqual(counts.device.type, "cpu")
        self.assertEqual(counts.dtype, torch.int64)
        self.assertEqual(counts.tolist(), [3, 2, 1, 1])
        expected = torch.cat([inp[0, :3], inp[1, :2], inp[2, :1], inp[3, :1]])
        self.assertTrue(torch.equal(actual, expected))

    def test_small_quantized_pooling_out_and_parameters(self):
        raw = torch.arange(1 * 2 * 5 * 5).reshape(1, 2, 5, 5).float() - 25
        inp = torch.quantize_per_tensor(raw, 0.1, 0, torch.qint8)
        expected = torch.nn.functional.max_pool2d(inp.int_repr().float(), 3, 2, 1).to(
            torch.int8
        )
        params = dict(kernel_size=3, stride=2, padding=1, dilation=1, ceil_mode=False)
        actual = self.modules["quantized_max_pool2d"].native()(inp, **params)
        self.assertTrue(torch.equal(actual.int_repr(), expected))
        self.assertEqual((actual.q_scale(), actual.q_zero_point()), (0.1, 0))
        out = torch._empty_affine_quantized(
            actual.shape, scale=0.1, zero_point=0, dtype=torch.qint8
        )
        result = self.modules["quantized_max_pool2d_out"].native()(
            inp, out=out, **params
        )
        self.assertIs(result, out)
        self.assertTrue(torch.equal(result.int_repr(), expected))


if __name__ == "__main__":
    unittest.main()
