"""Native-wrapper collection, overload selection and cross-device host replay."""

import contextlib
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import collect_baseline_host as collector
import host_baseline_plugin as plugin
import pytest_host_baseline as replay

OP = "_histogramdd_from_bin_cts"
WRAPPER = """def _cpu_reference(inp, *, bins, range=None, weight=None, density=False):
    cpu_weight = weight.detach().to("cpu") if weight is not None else None
    result = torch._histogramdd_from_bin_cts(
        inp.detach().to("cpu"),
        bins,
        range=range,
        weight=cpu_weight,
        density=density,
    )
    return result.to(inp.device)
"""


class HistogramHostTests(unittest.TestCase):
    def test_only_original_native_wrapper_is_eligible(self):
        name = "benchmark.test_histogramdd_from_bin_cts"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "histogram.py"
            for source, accepted in [
                (WRAPPER, True),
                (
                    WRAPPER.replace(
                        "return result.to(inp.device)",
                        "return result.to(inp.device) * 0",
                    ),
                    False,
                ),
            ]:
                path.write_text(source)
                module = types.ModuleType(name)
                exec(compile(source, str(path), "exec"), module.__dict__)
                with patch.dict(sys.modules, {name: module}):
                    self.assertEqual(
                        plugin.reviewed_histogram_wrapper(module._cpu_reference),
                        accepted,
                    )
                    self.assertFalse(plugin.reviewed_histogram_wrapper(lambda *a: None))

    def test_real_collector_commands_select_only_base_overload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source"
            (source / "benchmark").mkdir(parents=True)
            for name in ["base.py", "test_histogramdd_from_bin_cts.py"]:
                (source / "benchmark" / name).touch()
            commands = []

            def run(args, **kwargs):
                commands.append(args)
                output = Path(args[args.index("--output") + 1])
                if "--list-cases" in args:
                    data = {
                        "benchmarks": [{"op_name": OP, "cases": [{"case_id": "base"}]}]
                    }
                else:
                    data = {
                        "status": "PASSED",
                        "records": [
                            {
                                "operator": OP,
                                "case_id": "base",
                                "status": "PASSED",
                                "latency_ms": 0.1,
                                "host_measurement": {
                                    "kind": "cuda_dispatch_host_with_transfers"
                                },
                            }
                        ],
                    }
                output.write_text(json.dumps(data))
                return types.SimpleNamespace(wait=lambda **kw: 0)

            with patch.object(
                collector, "git_info", return_value={"commit": "a" * 40, "dirty": False}
            ), patch.object(
                collector.subprocess, "Popen", side_effect=run
            ), contextlib.redirect_stdout(
                io.StringIO()
            ):
                output = root / "run/host.json"
                self.assertEqual(
                    collector.main(
                        [
                            "--flaggems",
                            str(source),
                            "--operators",
                            OP,
                            "--output",
                            str(output),
                        ]
                    ),
                    0,
                )
            self.assertEqual(len(commands), 2)
            for args in commands:
                self.assertIn(
                    str(
                        source.resolve() / "benchmark/test_histogramdd_from_bin_cts.py"
                    ),
                    args,
                )
                self.assertIn(
                    ["-m", "underscore_histogramdd_from_bin_cts"],
                    [args[i : i + 2] for i in range(len(args) - 1)],
                )
            self.assertEqual(list(json.loads(output.read_text())["operators"]), [OP])

    def artifact(self):
        return {
            "_format": "flaggems_native_host_v1",
            "_source": {"flaggems": {"commit": "a" * 40, "dirty": False}},
            "_tasks": {OP: {"status": "completed", "returncode": 0}},
            "operators": {
                OP: [
                    {
                        "case_id": "case",
                        "operator": OP,
                        "status": "PASSED",
                        "latency_ms": 1.25,
                        "host_measurement": {
                            "kind": "cuda_dispatch_host_with_transfers",
                            "synchronized_cuda": True,
                            "native_api": "aten._histogramdd_from_bin_cts.default",
                            "sample_count": 3,
                            "number_per_run": 1,
                            "reference_wrapper": {
                                "source_ast_sha256": plugin._HISTOGRAM_WRAPPER_AST_SHA256
                            },
                            "environment": {
                                "cpu_info": "model",
                                "torch": "version",
                                "threads": 1,
                                "timer": "torch.utils.benchmark.Timer",
                                "cuda_device": "NVIDIA H800",
                                "cuda": "12.9",
                            },
                            "inputs": {
                                "args": [
                                    {
                                        "shape": [1024, 2],
                                        "dtype": "torch.float32",
                                        "device": "cuda:0",
                                    }
                                ],
                                "kwargs": {
                                    "bins": [20, 20],
                                    "range": [-30.0, 30.0] * 2,
                                },
                            },
                        },
                    }
                ]
            },
        }

    def lookup(self, data, actual=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host.json"
            path.write_text(json.dumps(data))
            actual = actual or data["operators"][OP][0]["host_measurement"]["inputs"]
            with patch.object(replay, "input_contract", return_value=actual):
                return replay.HostArtifact(path).lookup(OP, "case", (), {})

    def test_gcu_replay_keeps_complete_native_host_scope(self):
        data = self.artifact()
        actual = json.loads(
            json.dumps(data["operators"][OP][0]["host_measurement"]["inputs"])
        )
        actual["args"][0]["device"] = "gcu:0"
        self.assertEqual(self.lookup(data, actual)[0], 1.25)
        actual["kwargs"]["bins"] = [21, 20]
        with self.assertRaises(ValueError):
            self.lookup(data, actual)
        for field, bad in [
            ("kind", "cuda_kernel"),
            ("synchronized_cuda", False),
            ("native_api", "python.custom_histogram"),
        ]:
            data = self.artifact()
            data["operators"][OP][0]["host_measurement"][field] = bad
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.lookup(data)
        data = self.artifact()
        data["operators"][OP][0]["host_measurement"]["reference_wrapper"][
            "source_ast_sha256"
        ] = "bad"
        with self.assertRaises(ValueError):
            self.lookup(data)


if __name__ == "__main__":
    unittest.main()
