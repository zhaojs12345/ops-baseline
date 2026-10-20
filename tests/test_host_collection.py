import contextlib
import importlib.util
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "host_collector",
    Path(__file__).resolve().parents[1] / "tools/collect_baseline_host.py",
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class HostReportTests(unittest.TestCase):
    def test_source_change_during_catalog_cannot_complete_a_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "flaggems"
            (source / "benchmark").mkdir(parents=True)
            for name in ("base.py", "test__use_cudnn_ctc_loss.py"):
                (source / "benchmark" / name).touch()
            output = root / "run/host.json"
            clean = {"commit": "a" * 40, "dirty": False}
            dirty = {"commit": "a" * 40, "dirty": True}
            with (
                patch.object(
                    host, "git_info", side_effect=[clean, clean, clean, dirty]
                ),
                patch.object(host.subprocess, "Popen") as launch,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                launch.return_value.wait.return_value = 0
                code = host.main(
                    [
                        "--flaggems",
                        str(source),
                        "--output",
                        str(output),
                        "--operators",
                        "_use_cudnn_ctc_loss",
                    ]
                )
            self.assertEqual(code, 1)
            self.assertEqual(launch.call_count, 1)
            data = json.loads(output.read_text())
            self.assertFalse(data["operators"])
            task = data["_tasks"]["_use_cudnn_ctc_loss"]
            self.assertEqual(task["status"], "failed")
            self.assertIn("source changed", task["error"])

    def test_unreplayable_source_is_rejected_before_allocating_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "flaggems"
            (source / "benchmark").mkdir(parents=True)
            (source / "benchmark/base.py").touch()
            for info, output in [
                ({"commit": "a" * 40, "dirty": True}, root / "runs/dirty.json"),
                ({"commit": None, "dirty": None}, root / "runs/unknown.json"),
                ({"commit": "a" * 40, "dirty": False}, source / "runs/inside.json"),
            ]:
                with (
                    self.subTest(info=info, output=output),
                    patch.object(host, "git_info", return_value=info),
                    patch.object(host.subprocess, "Popen") as launch,
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    host.main(["--flaggems", str(source), "--output", str(output)])
                self.assertEqual(error.exception.code, 2)
                launch.assert_not_called()
                self.assertFalse(output.parent.exists())

    def report(self):
        return {
            "status": "PASSED",
            "records": [
                {
                    "operator": "q_per_channel_scales",
                    "case_id": "case1",
                    "status": "PASSED",
                    "latency_ms": 0.002,
                    "host_measurement": {"kind": "native_cpu"},
                }
            ],
        }

    def test_catalog_cannot_hide_or_mix_workloads(self):
        good = {
            "benchmarks": [
                {"op_name": "q_per_channel_scales", "cases": [{"case_id": "case1"}]}
            ]
        }
        self.assertEqual(host.validate_catalog(good, "q_per_channel_scales"), ["case1"])
        for data in [
            {},
            {"benchmarks": [{"op_name": "other", "cases": [{"case_id": "x"}]}]},
            {"benchmarks": [{"op_name": "q_per_channel_scales", "cases": []}]},
            {
                "benchmarks": [
                    {
                        "op_name": "q_per_channel_scales",
                        "cases": [{"case_id": "x"}, {"case_id": "x"}],
                    }
                ]
            },
        ]:
            with self.subTest(data=data), self.assertRaises(ValueError):
                host.validate_catalog(data, "q_per_channel_scales")

    def test_partial_reports_and_duplicate_cases_are_rejected(self):
        for status in ["FAILED", "UNSUPPORTED", None]:
            report = self.report()
            report["status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                host.validate_report(report, "q_per_channel_scales")
        report = self.report()
        report["records"] *= 2
        with self.assertRaises(ValueError):
            host.validate_report(report, "q_per_channel_scales")

    def test_missing_or_added_case_is_rejected(self):
        for expected in [["other"], ["case1", "case2"], []]:
            with self.subTest(expected=expected), self.assertRaises(ValueError):
                host.validate_report(self.report(), "q_per_channel_scales", expected)
        host.validate_report(self.report(), "q_per_channel_scales", ["case1"])

    def test_native_positive_clock_and_provenance_are_required(self):
        for key, values in {
            "latency_ms": [True, 0, -1, float("nan"), float("inf"), "1"],
            "host_measurement": [None, {}],
            "operator": ["other"],
            "status": ["SKIP", "FAILED"],
            "case_id": [None, ""],
        }.items():
            for value in values:
                report = self.report()
                report["records"][0][key] = value
                with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                    host.validate_report(report, "q_per_channel_scales")
        self.assertEqual(
            len(host.validate_report(self.report(), "q_per_channel_scales")), 1
        )


if __name__ == "__main__":
    unittest.main()
