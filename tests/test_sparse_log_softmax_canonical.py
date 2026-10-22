# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""CPU-only literal/native and complete input identity controls. No GPU timing."""

import copy
import importlib
import math
from pathlib import Path
import random
import sys
import unittest
from unittest import mock
import numpy as np
import torch

OPS = Path(__file__).resolve().parents[1] / "ops/pytorch"
sys.path.insert(0, str(OPS))
f = importlib.import_module("_sparse_log_softmax_inputs")
forward = importlib.import_module("_sparse_log_softmax")
backward = importlib.import_module("_sparse_log_softmax_backward_data")


class CanonicalLogSoftmaxTests(unittest.TestCase):
    def test_complete_original_descriptors_and_binding(self):
        expected_fwd = [
            ((1024, 1024), 1024, 1, 0),
            ((1024, 1024), 1024, 1, 1),
            ((20, 320, 15), 6400, 2, 0),
            ((20, 320, 15), 6400, 2, 2),
            ((16, 128, 64, 60), 8192, 3, 0),
            ((16, 128, 64, 60), 8192, 3, 2),
            ((16, 128, 64, 60), 8192, 3, 3),
            ((16, 7, 57, 32, 29), 8192, 4, 0),
            ((16, 7, 57, 32, 29), 8192, 4, 4),
        ]
        self.assertEqual(list(forward.SHAPES), expected_fwd)
        self.assertEqual(
            [x["nnz"] for x in backward.grid()],
            [1048576] * 5 + [1024, 8192, 8192, 1, 0, 0],
        )
        self.assertEqual([x["shape"] for x in backward.grid()][8:], [[], [], [1048576]])
        for module, count in [(forward, 9), (backward, 11)]:
            self.assertEqual(module.DTYPES, [torch.float32, torch.float64])
            self.assertEqual(len(module.grid()), count)
            self.assertEqual(len({module.key_shape(x) for x in module.grid()}), count)
            self.assertIs(
                module.native(), getattr(torch.ops.aten, module.OP_NAME).default
            )
            self.assertFalse(module.IS_INPLACE)
            b = module.grid()[0]
            for field in ["nnz", "dim", "sparse_dim"]:
                wrong = copy.deepcopy(b)
                wrong[field] += 1
                with self.assertRaises(ValueError):
                    module._checked(wrong, torch.float32)
            with self.assertRaises(TypeError):
                module._checked(b, torch.float16)
            b["shape"][0] = 7
            self.assertNotEqual(b, module.grid()[0])
        self.assertNotIn(
            "_sparse_log_softmax_inputs", {p.stem for p in OPS.glob("*.py")}
        )

    def test_original_group_support_literals_and_duplicates(self):
        np.testing.assert_array_equal(
            f.forward_indices((3, 4), 2, 6, 1), [[0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]]
        )
        np.testing.assert_array_equal(
            f.backward_indices((2, 3), 2, 6, 1),
            [[0, 0, 0, 1, 1, 1], [0, 1, 2, 0, 1, 2]],
        )
        np.testing.assert_array_equal(
            f.backward_indices((2, 3), 2, 6, 0),
            [[0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2]],
        )
        np.testing.assert_array_equal(
            f.backward_indices((3,), 1, 6, 0), [[0, 1, 2, 0, 1, 2]]
        )
        np.testing.assert_array_equal(f.backward_indices((4, 3), 1, 2, 1), [[0, 2]])
        self.assertEqual(f.backward_indices((), 0, 0, -1).shape, (0, 0))
        self.assertEqual(f.backward_indices((), 0, 1, -1).shape, (0, 1))

    def test_rng_private_exact_width_and_reconstruction(self):
        ts = torch.random.get_rng_state().clone()
        ns = np.random.get_state()
        ps = random.getstate()
        for dtype in f.DTYPES:
            a = {"input": f.forward_cpu((3, 4, 2), 6, 2, 1, dtype)}
            self.assertEqual(
                f.input_identity(a),
                f.input_identity({"input": f.forward_cpu((3, 4, 2), 6, 2, 1, dtype)}),
            )
            triple = f.backward_cpu((3, 4, 2), 6, 2, 1, dtype)
            named = dict(zip(["grad", "output", "self"], triple))
            identity = f.input_identity(named)
            self.assertEqual(identity, f.input_identity(f.to_device(named, "cpu")))
            self.assertEqual(len(identity["component_storage_aliases"]), 15)
            self.assertFalse(any(identity["component_storage_aliases"].values()))
            for value in triple:
                self.assertEqual(value.layout, torch.sparse_coo)
                self.assertEqual(value.shape, torch.Size([3, 4, 2]))
                self.assertEqual(value.dtype, dtype)
                self.assertFalse(value.is_coalesced())
                self.assertTrue(torch.equal(value._indices(), triple[0]._indices()))
            self.assertFalse(torch.equal(triple[0]._values(), triple[1]._values()))
            self.assertFalse(torch.equal(triple[0]._values(), triple[2]._values()))
            if dtype == torch.float64:
                self.assertTrue(
                    bool(
                        (
                            triple[0]._values() != triple[0]._values().float().double()
                        ).any()
                    )
                )
        self.assertTrue(torch.equal(ts, torch.random.get_rng_state()))
        self.assertEqual(ps, random.getstate())
        after = np.random.get_state()
        self.assertEqual(ns[0], after[0])
        np.testing.assert_array_equal(ns[1], after[1])
        self.assertEqual(ns[2:], after[2:])

    def test_native_cpu_independent_literal_formula_and_readonly(self):
        indices = torch.tensor([[0, 0, 1], [0, 2, 1]], dtype=torch.int64)
        for dtype in f.DTYPES:
            inp = torch.sparse_coo_tensor(
                indices.clone(),
                torch.tensor([0.0, 0.0, 4.0], dtype=dtype),
                (2, 3),
                is_coalesced=True,
            )
            before = f.input_identity({"input": inp})
            out = forward.native()(inp, 1, half_to_float=False)
            torch.testing.assert_close(
                out.values(),
                torch.tensor([-math.log(2), -math.log(2), 0.0], dtype=dtype),
                rtol=1e-6,
                atol=1e-7,
            )
            self.assertEqual(before, f.input_identity({"input": inp}))
            grad = torch.sparse_coo_tensor(
                indices.clone(), torch.tensor([1.0, 2.0, 3.0], dtype=dtype), (2, 3)
            )
            output = torch.sparse_coo_tensor(
                indices.clone(),
                torch.tensor([-math.log(2), -math.log(2), 0.0], dtype=dtype),
                (2, 3),
            )
            self_tensor = torch.sparse_coo_tensor(
                indices.clone(), torch.tensor([8.0, 9.0, 10.0], dtype=dtype), (2, 3)
            )
            named = {"grad": grad, "output": output, "self": self_tensor}
            before = f.input_identity(named)
            actual = backward.native()(grad, output, 1, self_tensor)
            expected = torch.tensor([[-0.5, 0.0, 0.5], [0.0, 0.0, 0.0]], dtype=dtype)
            torch.testing.assert_close(
                actual.to_dense(), expected, rtol=1e-6, atol=1e-7
            )
            self.assertEqual(before, f.input_identity(named))

    def test_scalar_empty_and_duplicate_original_contract(self):
        for dtype in f.DTYPES:
            for shape, nnz, sd, dim in [
                ((), 1, 0, -1),
                ((), 0, 0, -1),
                ((7,), 0, 1, 0),
                ((3,), 6, 1, 0),
            ]:
                g, o, s = f.backward_cpu(shape, nnz, sd, dim, dtype)
                identity = f.input_identity({"grad": g, "output": o, "self": s})
                self.assertEqual(s.layout, torch.sparse_coo)
                self.assertEqual(s._nnz(), nnz)
                self.assertEqual(
                    identity,
                    f.input_identity(
                        f.to_device({"grad": g, "output": o, "self": s}, "cpu")
                    ),
                )
                result = backward.native()(g, o, dim, s)
                self.assertEqual(result.shape, g.shape)
                self.assertEqual(result.dtype, dtype)
                self.assertEqual(
                    identity, f.input_identity({"grad": g, "output": o, "self": s})
                )
            self.assertEqual(f.backward_cpu((3,), 6, 1, 0, dtype)[0]._nnz(), 6)

    def test_wrong_storage_width_alias_payload_and_lowbits(self):
        g, o, s = f.backward_cpu((3, 4), 6, 2, 1, torch.float64)
        named = {"grad": g, "output": o, "self": s}
        before = f.input_identity(named)
        s._values()[0] += 2**-40
        self.assertNotEqual(before, f.input_identity(named))
        wrong = torch.sparse_coo_tensor(g._indices(), o._values(), g.shape)
        self.assertNotEqual(
            before, f.input_identity({"grad": g, "output": wrong, "self": s})
        )
        for wrong in [
            torch.arange(8).reshape(2, 4).t(),
            torch.arange(8)[1:],
            torch._neg_view(torch.ones(3)),
        ]:
            with self.assertRaises(ValueError):
                f.input_identity({"bad": wrong})
        wrong = torch.sparse_coo_tensor(
            g._indices().clone(), g._values().float(), g.shape
        )
        self.assertNotEqual(f.input_identity({"x": g}), f.input_identity({"x": wrong}))

    def test_actual_copy_payload_corruption_rejected(self):
        inp = f.forward_cpu((3, 4), 6, 2, 1, torch.float64)
        before = f.input_identity({"input": inp})
        original = torch.Tensor.to
        calls = []

        def corrupt(part, *args, **kwargs):
            result = original(part, *args, **kwargs)
            calls.append(1)
            if len(calls) == 2:
                result[0] ^= 1
            return result

        with mock.patch.object(torch.Tensor, "to", corrupt):
            with self.assertRaisesRegex(ValueError, "payload"):
                f.to_device({"input": inp}, "cpu")
        self.assertEqual(before, f.input_identity({"input": inp}))

    def test_full_forty_config_builder_identity_and_source_case_ids(self):
        for module in [forward, backward]:
            for binding in module.grid():
                for dtype in module.DTYPES:
                    args, kwargs = module.build_inputs(binding, dtype, "cpu")
                    named = (
                        {"input": args[0]}
                        if module is forward
                        else {"grad": args[0], "output": args[1], "self": args[3]}
                    )
                    cfg = module.config(binding, dtype)
                    self.assertEqual(cfg["inputs"], f.input_identity(named))
                    self.assertEqual(cfg["requested_nnz"], binding["nnz"])
                    self.assertEqual(cfg["dim"], binding["dim"])
                    expected = (
                        "benchmark/test_"
                        + module.OP_NAME
                        + ".py::test_"
                        + module.OP_NAME
                        + ("_benchmark" if module is backward else "")
                        + "::core::"
                        + str(dtype).removeprefix("torch.")
                        + "::"
                        + str(binding["ordinal"])
                    )
                    self.assertEqual(cfg["case_id"], expected)
                    self.assertEqual(
                        kwargs, {} if module is backward else {"half_to_float": False}
                    )
                    for tensor in named.values():
                        self.assertEqual(tensor._nnz(), binding["nnz"])
                        self.assertEqual(list(tensor.shape), binding["shape"])
                        self.assertEqual(tensor.dtype, dtype)
                        self.assertEqual(tensor.device, torch.device("cpu"))
                    jsonable = __import__("json").dumps(cfg)
                    self.assertTrue(jsonable)


if __name__ == "__main__":
    unittest.main()
