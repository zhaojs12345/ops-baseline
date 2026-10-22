# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Focused CPU input/native-contract tests; no CUDA timing claims."""

import copy
import importlib
import random
import sys
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import torch

OPS = Path(__file__).resolve().parents[1] / "ops" / "pytorch"
sys.path.insert(0, str(OPS))
fixture = importlib.import_module("_sparse_softmax_inputs")
forward = importlib.import_module("_sparse_softmax")
backward = importlib.import_module("_sparse_softmax_backward_data")


class CanonicalSparseSoftmaxTests(unittest.TestCase):
    def test_original_catalog_and_native_binding(self):
        for module in (forward, backward):
            self.assertEqual(module.DTYPES, [torch.float32, torch.float64])
            self.assertEqual(len(module.grid()), 6)
            self.assertEqual(len({module.key_shape(b) for b in module.grid()}), 6)
            self.assertIs(
                module.native(), getattr(torch.ops.aten, module.OP_NAME).default
            )
            self.assertFalse(module.IS_INPLACE)
            for binding in module.grid():
                mutated = copy.deepcopy(binding)
                mutated["nnz"] += 1
                with self.assertRaises(ValueError):
                    module._checked(mutated, torch.float32)
                with self.assertRaises(TypeError):
                    module._checked(binding, torch.float16)
            original = module.grid()[0]
            changed = module.grid()[0]
            changed["shape"][0] = 1
            self.assertNotEqual(original, changed)
            self.assertEqual(module.grid()[0], original)
        self.assertNotIn("_sparse_softmax_inputs", {p.stem for p in OPS.glob("*.py")})

    def test_rng_is_private_and_replay_exact(self):
        torch_before = torch.random.get_rng_state().clone()
        numpy_before = np.random.get_state()
        python_before = random.getstate()
        for dtype in fixture.DTYPES:
            a = {"input": fixture.forward_cpu((3, 5), 17, 1, dtype)}
            b = {"input": fixture.forward_cpu((3, 5), 17, 1, dtype)}
            self.assertEqual(fixture.input_identity(a), fixture.input_identity(b))
            g, o, s = fixture.backward_cpu((3, 5, 2), 8, 2, 2, dtype)
            c = {"grad": g, "output": o, "self": s}
            g2, o2, s2 = fixture.backward_cpu((3, 5, 2), 8, 2, 2, dtype)
            self.assertEqual(
                fixture.input_identity(c),
                fixture.input_identity({"grad": g2, "output": o2, "self": s2}),
            )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_before))
        after = np.random.get_state()
        self.assertEqual(numpy_before[0], after[0])
        np.testing.assert_array_equal(numpy_before[1], after[1])
        self.assertEqual(numpy_before[2:], after[2:])
        self.assertEqual(random.getstate(), python_before)

    def test_stored_zero_and_duplicate_order(self):
        indices = np.array([[1, 0, 1, 0], [2, 1, 2, 1]], dtype=np.int64)
        for dtype in [np.float32, np.float64]:
            values = np.array([2.5, 3.0, -2.5, 1.0], dtype=dtype)
            i, v = fixture._canonical_coalesce(indices, values, (2, 3))
            np.testing.assert_array_equal(i, [[0, 1], [1, 2]])
            np.testing.assert_array_equal(v, [4.0, 0.0])
            self.assertEqual(v.dtype, dtype)
            self.assertEqual(v.size, 2)  # stored zero support is not pruned

    def test_empty_forward_geometry_and_transport(self):
        for dtype in fixture.DTYPES:
            inp = fixture.forward_cpu((4, 5), 0, 1, dtype)
            old = fixture.input_identity({"input": inp})
            new = fixture.to_device({"input": inp}, "cpu")
            self.assertEqual(old, fixture.input_identity(new))
            self.assertTrue(inp.is_coalesced())
            self.assertEqual(inp._indices().stride(), (1, 1))
            self.assertEqual(inp._values().stride(), (1,))
            self.assertEqual(inp._nnz(), 0)
            actual = forward.native()(new["input"], dim=1, half_to_float=False)
            self.assertEqual(actual.dtype, dtype)
            self.assertEqual(actual._nnz(), 0)

    def test_backward_independence_uncoalesced_and_full_width(self):
        for dtype in fixture.DTYPES:
            g, o, s = fixture.backward_cpu((3, 5, 2), 8, 2, 2, dtype)
            inputs = {"grad": g, "output": o, "self": s}
            before = fixture.input_identity(inputs)
            self.assertFalse(g.is_coalesced())
            self.assertFalse(o.is_coalesced())
            self.assertTrue(torch.equal(g._indices(), o._indices()))
            self.assertFalse(any(before["component_storage_aliases"].values()))
            self.assertEqual(len(before["component_storage_aliases"]), 10)
            self.assertFalse(torch.equal(g._values(), o._values()))
            moved = fixture.to_device(inputs, "cpu")
            self.assertEqual(before, fixture.input_identity(moved))
            for value in moved.values():
                self.assertEqual(value.device, torch.device("cpu"))
                self.assertEqual(value.dtype, dtype)
            if dtype == torch.float64:
                self.assertTrue(
                    bool((g._values() != g._values().float().double()).any())
                )
        literal = {
            "x": torch.tensor([2**53 + 1], dtype=torch.int64),
            "scalar": torch.tensor(1 + 2**-40, dtype=torch.float64),
        }
        self.assertEqual(
            fixture.input_identity(literal),
            fixture.input_identity(fixture.to_device(literal, "cpu")),
        )

    def test_actual_cpu_native_formula_and_input_readonly(self):
        indices = torch.tensor([[0, 0, 1], [0, 2, 1]], dtype=torch.int64)
        for dtype in fixture.DTYPES:
            inp = torch.sparse_coo_tensor(
                indices.clone(),
                torch.tensor([0.0, 0.0, 4.0], dtype=dtype),
                (2, 3),
                is_coalesced=True,
            )
            before = fixture.input_identity({"input": inp})
            result = forward.native()(inp, dim=1, half_to_float=False)
            torch.testing.assert_close(
                result.values(),
                torch.tensor([0.5, 0.5, 1.0], dtype=dtype),
                rtol=0,
                atol=0,
            )
            self.assertEqual(before, fixture.input_identity({"input": inp}))
            grad = torch.sparse_coo_tensor(
                indices.clone(), torch.tensor([1.0, 2.0, 3.0], dtype=dtype), (2, 3)
            )
            output = torch.sparse_coo_tensor(
                indices.clone(), torch.tensor([-0.5, 0.25, 1.5], dtype=dtype), (2, 3)
            )
            self_tensor = torch.ones(1, dtype=dtype)
            inputs = {"grad": grad, "output": output, "self": self_tensor}
            before = fixture.input_identity(inputs)
            actual = backward.native()(grad, output, 1, self_tensor)
            expected = torch.tensor([[-0.5, 0.0, 0.5], [0.0, -2.25, 0.0]], dtype=dtype)
            torch.testing.assert_close(actual.to_dense(), expected, rtol=0, atol=0)
            self.assertEqual(before, fixture.input_identity(inputs))

    def test_corrupted_target_copy_is_rejected(self):
        inp = fixture.forward_cpu((3, 5), 17, 1, torch.float64)
        original = torch.Tensor.to
        calls = []

        def corrupt(part, *args, **kwargs):
            result = original(part, *args, **kwargs)
            calls.append(1)
            if len(calls) == 2:  # values copy; do not corrupt coordinate bounds
                result[0] ^= 1
            return result

        before = fixture.input_identity({"input": inp})
        with mock.patch.object(torch.Tensor, "to", corrupt):
            with self.assertRaisesRegex(ValueError, "payload"):
                fixture.to_device({"input": inp}, "cpu")
        self.assertEqual(before, fixture.input_identity({"input": inp}))

    def test_wrong_geometry_alias_width_and_payload_detected(self):
        g, o, s = fixture.backward_cpu((3, 5), 8, 2, 1, torch.float64)
        inputs = {"grad": g, "output": o, "self": s}
        expected = fixture.input_identity(inputs)
        aliased = torch.sparse_coo_tensor(g._indices(), o._values(), g.shape)
        self.assertNotEqual(
            expected, fixture.input_identity({"grad": g, "output": aliased, "self": s})
        )
        s[0] += 2**-40
        self.assertNotEqual(expected, fixture.input_identity(inputs))
        with self.assertRaises(ValueError):
            fixture.input_identity({"bad": torch.arange(8).reshape(2, 4).t()})
        with self.assertRaises(ValueError):
            fixture.input_identity({"bad": torch.arange(8)[1:]})
        with self.assertRaises(ValueError):
            fixture.input_identity({"bad": torch._neg_view(torch.ones(3))})
        wrong = torch.sparse_coo_tensor(
            g._indices().clone(), g._values().float(), g.shape
        )
        self.assertNotEqual(
            expected, fixture.input_identity({"grad": wrong, "output": o, "self": s})
        )


if __name__ == "__main__":
    unittest.main()
