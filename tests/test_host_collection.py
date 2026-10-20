import importlib.util
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location(
    "host_collector",
    Path(__file__).resolve().parents[1] / "tools/collect_baseline_host.py",
)
host = importlib.util.module_from_spec(spec)
spec.loader.exec_module(host)


class HostReportTests(unittest.TestCase):
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
