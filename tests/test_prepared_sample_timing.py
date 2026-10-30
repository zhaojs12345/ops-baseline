"""CPU-only fake-event ordering tests for the actual optional-preparation collector.
These are not real torch/CUDA events, latencies or baseline records.
"""

from pathlib import Path
import copy, hashlib, importlib.util, json, sys, types, unittest
from unittest.mock import patch

B = Path(__file__).resolve().parents[1]


class Harness:
    def __init__(self, elapsed_override=None):
        self.log = []
        self.clock = 0.0
        self.native_calls = 0
        self.default_calls = []
        self.snapshots = []
        self.elapsed_override = elapsed_override
        self.args = ([2.0, -3.0],)
        self.kwargs = {"out": [91.0, 92.0]}
        self.original = copy.deepcopy((self.args, self.kwargs))
        owner = self

        class Event:
            def __init__(self, enable_timing):
                assert enable_timing
                self.timestamp = None
                owner.log.append("create_event")

            def record(self):
                self.timestamp = owner.clock
                owner.log.append("record")

            def synchronize(self):
                owner.log.append("end_sync")

            def elapsed_time(self, end):
                owner.log.append("elapsed")
                return (
                    end.timestamp - self.timestamp
                    if owner.elapsed_override is None
                    else owner.elapsed_override
                )

        def synchronize():
            self.log.append("prepare_sync")
            self.clock += 500

        def do_bench(fn, **kwargs):
            self.default_calls.append(kwargs)
            fn()
            return 2.5

        self.torch = types.SimpleNamespace(
            cuda=types.SimpleNamespace(Event=Event, synchronize=synchronize)
        )
        self.triton = types.SimpleNamespace(
            testing=types.SimpleNamespace(do_bench=do_bench)
        )

    def prepare(self, args, kwargs):
        self.log.append("prepare")
        self.clock += 1000
        a, k = copy.deepcopy((args, kwargs))
        # Each sample has an independent snapshot relative to the original. No
        # global id/data_ptr uniqueness assertion: real allocators reuse memory.
        assert a[0] is not args[0] and k["out"] is not kwargs["out"]
        assert (a, k) == self.original
        self.snapshots.append((copy.deepcopy(a), copy.deepcopy(k)))
        return a, k

    def native(self, x, *, out):
        self.log.append("native")
        self.clock += 3
        self.native_calls += 1
        assert x == self.original[0][0]
        out[:] = [1.0, -1.0]
        return out

    def load(self):
        spec = importlib.util.spec_from_file_location(
            "prepared_collector_under_test", B / "tools/collect_baseline_nvidia.py"
        )
        module = importlib.util.module_from_spec(spec)
        with patch.dict(
            sys.modules, {"torch": self.torch, "triton": self.triton}
        ), patch.object(sys, "path", [str(B / "tools"), *sys.path]):
            spec.loader.exec_module(module)
        return vars(module)

    def assert_correct_order(self):
        expected = [
            "prepare",
            "prepare_sync",
            "create_event",
            "create_event",
            "record",
            "native",
            "record",
            "end_sync",
            "elapsed",
        ]
        assert len(self.log) % len(expected) == 0
        for i in range(0, len(self.log), len(expected)):
            assert self.log[i : i + len(expected)] == expected
        assert (self.args, self.kwargs) == self.original


class PreparedSampleTests(unittest.TestCase):
    def test_default_uses_original_do_bench_and_unchanged_kwargs(self):
        h = Harness()
        f = h.load()["_ops_latency_ms"]
        v = f(types.SimpleNamespace(), h.native, h.args, h.kwargs, False)
        self.assertEqual(v, 2.5)
        self.assertEqual(h.default_calls, [dict(warmup=25, rep=100)])
        self.assertNotIn("prepare", h.log)
        self.assertFalse(any(x == "create_event" for x in h.log))

    def test_preparation_is_outside_clock_every_sample_and_original_unchanged(self):
        h = Harness()
        v = h.load()["_ops_latency_ms"](
            types.SimpleNamespace(prepare_sample=h.prepare),
            h.native,
            h.args,
            h.kwargs,
            False,
        )
        self.assertEqual(v, 3.0)
        h.assert_correct_order()
        self.assertEqual(h.native_calls, 43)
        self.assertEqual(len(h.snapshots), 43)
        self.assertFalse(h.default_calls)
        self.assertEqual((h.args, h.kwargs), h.original)

    def test_old_prepare_inside_clock_is_actually_rejected(self):
        h = Harness()
        start = h.torch.cuda.Event(enable_timing=True)
        end = h.torch.cuda.Event(enable_timing=True)
        start.record()
        a, k = h.prepare(h.args, h.kwargs)
        h.torch.cuda.synchronize()
        h.native(*a, **k)
        end.record()
        end.synchronize()
        wrong = start.elapsed_time(end)
        self.assertEqual(wrong, 1503.0)
        with self.assertRaises(AssertionError):
            h.assert_correct_order()

    def test_ncu_hook_rejected_before_prepare_or_native(self):
        h = Harness()
        with self.assertRaisesRegex(ValueError, "only with --no-ncu"):
            h.load()["_ops_latency_ms"](
                types.SimpleNamespace(prepare_sample=h.prepare),
                h.native,
                h.args,
                h.kwargs,
                True,
            )
        self.assertEqual(h.log, [])

    def test_collector_ncu_and_invalid_hook_fail_before_native_resolution(self):
        h = Harness()
        ns = h.load()
        op = types.SimpleNamespace(
            OP_NAME="sgn_out",
            prepare_sample=h.prepare,
            native=lambda: self.fail("must reject before resolve"),
        )
        with self.assertRaisesRegex(ValueError, "only with --no-ncu"):
            ns["_collect_one_op_ops"](op, True, None)
        op.prepare_sample = 13
        with self.assertRaises(TypeError):
            ns["_collect_one_op_ops"](op, False, None)
        self.assertEqual(h.log, [])

    def test_bad_container_and_nonpositive_nonfinite_timing_fail_closed(self):
        h = Harness()
        fn = h.load()["_ops_latency_ms"]
        for value in [None, ([],), ([], None), {"latency": 1.0}]:
            with self.subTest(value=value), self.assertRaises(TypeError):
                fn(
                    types.SimpleNamespace(prepare_sample=lambda a, k: value),
                    h.native,
                    h.args,
                    h.kwargs,
                    False,
                )
        for v in [0.0, -1.0, float("inf"), float("nan")]:
            h = Harness(v)
            with self.subTest(v=v), self.assertRaises(RuntimeError):
                h.load()["_ops_latency_ms"](
                    types.SimpleNamespace(prepare_sample=h.prepare),
                    h.native,
                    h.args,
                    h.kwargs,
                    False,
                )

    def test_bool_latency_rejected_before_mean_aggregation(self):
        for value in [True, False]:
            h = Harness(value)
            with self.subTest(value=value), self.assertRaisesRegex(
                RuntimeError, "nonpositive/nonfinite"
            ):
                h.load()["_ops_latency_ms"](
                    types.SimpleNamespace(prepare_sample=h.prepare),
                    h.native,
                    h.args,
                    h.kwargs,
                    False,
                )
            self.assertEqual(h.native_calls, 1)

    def test_invalid_hook_type_rejected(self):
        h = Harness()
        with self.assertRaises(TypeError):
            h.load()["_ops_latency_ms"](
                types.SimpleNamespace(prepare_sample=1.2),
                h.native,
                h.args,
                h.kwargs,
                False,
            )
        self.assertEqual(h.log, [])

    def test_actual_collector_default_preserves_record_config_and_units(self):
        h = Harness()
        ns = h.load()
        op = types.SimpleNamespace(
            OP_NAME="test_out",
            __name__="test_out",
            native=lambda: h.native,
            DTYPES=["torch.float32"],
            grid=lambda: [{"shape": [2]}],
            key_shape=lambda b: "[2]",
            build_inputs=lambda *args: copy.deepcopy(h.original),
            config=lambda *args: {"original_config": True},
        )
        result = ns["_collect_one_op_ops"](op, False, None)
        cell = result["shapes"]["[2]"]["torch.float32"]
        self.assertEqual(cell["T_us"], 2500.0)
        self.assertEqual(cell["latency_ms"], 2.5)
        self.assertEqual(cell["config"], {"original_config": True})
        self.assertEqual(result["native_api"], "ops.test_out")
        self.assertEqual(h.default_calls, [dict(warmup=25, rep=100)])
        self.assertNotIn("create_event", h.log)

    def test_actual_collector_preflight_then_every_prepared_sample(self):
        h = Harness()
        ns = h.load()
        op = types.SimpleNamespace(
            OP_NAME="test_out",
            __name__="test_out",
            native=lambda: h.native,
            DTYPES=["torch.float32"],
            grid=lambda: [{"shape": [2]}],
            key_shape=lambda b: "[2]",
            build_inputs=lambda *args: copy.deepcopy(h.original),
            prepare_sample=h.prepare,
        )
        result = ns["_collect_one_op_ops"](op, False, None)
        self.assertEqual(h.log[:2], ["native", "prepare_sync"])
        h.log = h.log[2:]
        h.assert_correct_order()
        self.assertEqual(h.native_calls, 44)
        self.assertEqual(result["shapes"]["[2]"]["torch.float32"]["T_us"], 3000.0)
        self.assertFalse(h.default_calls)



class CurrentMainCompatibilityTests(unittest.TestCase):
    def test_nonhook_return_mode_is_forwarded(self):
        h = Harness()
        module = h.load()
        result = module["_ops_latency_ms"](
            types.SimpleNamespace(), lambda x, **kw: None,
            h.args, h.kwargs, False, {"return_mode": "median"})
        self.assertEqual(result, 2.5)
        self.assertEqual(h.default_calls, [
            {"warmup": 25, "rep": 100, "return_mode": "median"}])

    def test_prepared_return_override_fails_before_native(self):
        h = Harness()
        module = h.load()
        with self.assertRaisesRegex(ValueError, "owns mean"):
            module["_ops_latency_ms"](
                types.SimpleNamespace(prepare_sample=h.prepare), h.native,
                h.args, h.kwargs, False, {"return_mode": "median"})
        self.assertEqual(h.native_calls, 0)
        self.assertEqual(h.log, [])

if __name__ == "__main__":
    result = unittest.TextTestRunner(verbosity=2).run(
        unittest.defaultTestLoader.loadTestsFromTestCase(PreparedSampleTests)
    )
    print(
        json.dumps(
            dict(
                status=(
                    "PASS_HOST_FAKE_EVENT_COLLECTOR_CONTROLS"
                    if result.wasSuccessful()
                    else "FAIL"
                ),
                tests=result.testsRun,
                failures=len(result.failures),
                errors=len(result.errors),
                collector_source_sha256=hashlib.sha256(
                    (B / "tools/collect_baseline_nvidia.py").read_bytes()
                ).hexdigest(),
                scope="External temporary fake torch/triton modules; no CUDA/Torch runtime, no GPU timing or data collected",
                old_prepare_inside_clock_red_synthetic_units=1503.0,
                prepared_event_elapsed_green_synthetic_units=3.0,
                units_note="Synthetic clock units only; never baseline performance",
                no_cross_sample_global_address_uniqueness_requirement=True,
            ),
            indent=2,
        )
    )
    raise SystemExit(0 if result.wasSuccessful() else 1)
