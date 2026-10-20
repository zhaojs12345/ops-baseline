import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import pytest_host_baseline as host


class HostReplayTests(unittest.TestCase):
    def artifact(self):
        return {
            "_format": "flaggems_native_host_v1",
            "_source": {"flaggems": {"commit": "a" * 40, "dirty": False}},
            "_tasks": {
                "q_per_channel_scales": {"status": "completed", "returncode": 0}
            },
            "operators": {
                "q_per_channel_scales": [
                    {
                        "case_id": "case",
                        "operator": "q_per_channel_scales",
                        "status": "PASSED",
                        "latency_ms": 0.002,
                        "host_measurement": {
                            "kind": "native_cpu",
                            "synchronized_cuda": False,
                            "native_api": "aten.q_per_channel_scales",
                            "sample_count": 3,
                            "number_per_run": 100,
                            "environment": {
                                "cpu_info": "CPU model",
                                "torch": "2.8",
                                "timer": "torch.utils.benchmark.Timer",
                                "threads": 1,
                            },
                            "inputs": {
                                "args": [{"shape": [2, 4], "device": "cpu"}],
                                "kwargs": {},
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
            inputs = (
                actual
                or data["operators"]["q_per_channel_scales"][0]["host_measurement"][
                    "inputs"
                ]
            )
            with patch.object(host, "input_contract", return_value=inputs):
                return host.HostArtifact(path).lookup(
                    "q_per_channel_scales", "case", (), {}
                )

    def test_clock_source_kind_and_device_contract(self):
        self.assertEqual(self.lookup(self.artifact())[0], 0.002)
        for key, values in {
            "latency_ms": [0, -1, True, "1", float("nan"), float("inf")],
            "status": ["SKIP", "FAILED"],
            "case_id": ["other"],
        }.items():
            for value in values:
                data = self.artifact()
                data["operators"]["q_per_channel_scales"][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    self.lookup(data)
        with self.assertRaises(ValueError):
            self.lookup(
                self.artifact(),
                {"args": [{"shape": [2, 4], "device": "npu:0"}], "kwargs": {}},
            )
        for kind in ["portable_cpu", "cuda_kernel", "external_host"]:
            data = self.artifact()
            data["operators"]["q_per_channel_scales"][0]["host_measurement"][
                "kind"
            ] = kind
            with self.subTest(kind=kind), self.assertRaises(ValueError):
                self.lookup(data)

    def test_incomplete_dirty_and_duplicate_sources_rejected(self):
        data = self.artifact()
        data["operators"]["q_per_channel_scales"] *= 2
        with self.assertRaises(ValueError):
            self.lookup(data)
        for field, value in [
            ("status", "failed"),
            ("returncode", 1),
            ("returncode", False),
        ]:
            data = self.artifact()
            data["_tasks"]["q_per_channel_scales"][field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.lookup(data)
        data = self.artifact()
        data["_source"]["flaggems"]["dirty"] = True
        with self.assertRaises(ValueError):
            self.lookup(data)

    def test_native_cpu_does_not_accept_a_different_timing_scope(self):
        data = self.artifact()
        data["operators"]["q_per_channel_scales"][0]["host_measurement"][
            "synchronized_cuda"
        ] = True
        with self.assertRaises(ValueError):
            self.lookup(data)

    def test_only_accelerator_roles_are_portable(self):
        self.assertEqual(
            host._canonical({"device": "cuda:4"}), host._canonical({"device": "npu:0"})
        )
        self.assertNotEqual(
            host._canonical({"device": "cpu"}), host._canonical({"device": "cuda:0"})
        )
        self.assertNotEqual(host._canonical({"axis": 1}), host._canonical({"axis": 2}))


if __name__ == "__main__":
    unittest.main()
