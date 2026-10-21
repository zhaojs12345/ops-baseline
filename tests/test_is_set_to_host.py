import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
import collect_baseline_host as collector
import host_baseline_plugin as plugin
import pytest_host_baseline as replay

contract = plugin.is_set_to_module()


class IsSetToHostTests(unittest.TestCase):
    def test_storage_identity_is_not_a_numerical_pointer(self):
        empty_a, empty_b = torch.empty(0), torch.empty(0)
        self.assertEqual(empty_a.data_ptr(), empty_b.data_ptr())
        self.assertFalse(contract.view_contract((empty_a, empty_b), {})["same_storage"])
        base = torch.empty(4, 8)
        for other, expected in [
            (base, True),
            (base.detach(), True),
            (base.T, False),
            (base.clone(), False),
        ]:
            self.assertIs(
                contract.validate_native(contract.native(), (base, other), {}), expected
            )
        for lazy in [
            torch._neg_view(base),
            torch.empty(4, 8, dtype=torch.complex64).conj(),
        ]:
            self.assertFalse(
                contract.validate_native(contract.native(), (lazy, lazy), {})
            )
        view = base.view(torch.int32)
        self.assertTrue(contract.validate_native(contract.native(), (base, view), {}))
        for other, expected in [(base[:0], True), (base[1:1], False)]:
            self.assertIs(
                contract.validate_native(contract.native(), (base[:0], other), {}),
                expected,
            )
        with self.assertRaises(ValueError):
            contract.validate_native(lambda *args: True, (base, base), {})

    def artifact(self, args):
        inputs = plugin.input_contract("is_set_to", args, {})
        for a in inputs["args"]:
            a["device"] = "cuda:0"
        record = {
            "operator": "is_set_to",
            "case_id": "source-index-7",
            "dtype": str(args[0].dtype),
            "params": {"relation": "same_object"},
            "status": "PASSED",
            "latency_ms": 0.002,
            "host_measurement": {
                "kind": "cuda_dispatch_host_predicate",
                "native_api": "aten.is_set_to",
                "synchronized_cuda": False,
                "sample_count": 3,
                "number_per_run": 100,
                "native_validation": True,
                "inputs": inputs,
                "environment": {
                    "cpu_info": "unit-test CPU",
                    "torch": "2.8",
                    "timer": "torch.utils.benchmark.Timer",
                    "threads": 1,
                    "cuda_device": "NVIDIA H800",
                    "cuda": "12.9",
                },
            },
        }
        return {
            "_format": "flaggems_native_host_v1",
            "_source": {
                "flaggems": {"commit": "a" * 40, "dirty": False},
                "collector": {"commit": "b" * 40, "dirty": False},
            },
            "_tasks": {
                "is_set_to": {
                    "status": "completed",
                    "returncode": 0,
                    "scope": "explicit_core_selection",
                    "selected_case_ids": ["source-index-7"],
                    "catalog_case_ids": ["source-index-0", "source-index-7"],
                }
            },
            "operators": {"is_set_to": [record]},
        }

    def lookup(self, data, actual, args, params=None):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "host.json"
            path.write_text(json.dumps(data))
            with patch.object(replay, "input_contract", return_value=actual):
                return replay.HostArtifact(path).lookup(
                    "is_set_to",
                    "target-index-0",
                    args,
                    {},
                    params=params or {"relation": "same_object"},
                )

    def test_matching_uses_actual_views_and_relation_not_backend_ordinal(self):
        x = torch.empty(2, 3)
        args = (x, x)
        data = self.artifact(args)
        actual = copy.deepcopy(
            data["operators"]["is_set_to"][0]["host_measurement"]["inputs"]
        )
        for a in actual["args"]:
            a["device"] = "npu:0"
        latency, provenance = self.lookup(data, actual, args)
        self.assertEqual(latency, 0.002)
        self.assertEqual(provenance["source_case_id"], "source-index-7")
        self.assertEqual(provenance["target_case_id"], "target-index-0")
        for key, value in [
            ("same_storage", False),
            ("storage_offsets", [0, 1]),
            ("conjugate", [False, True]),
        ]:
            wrong = copy.deepcopy(actual)
            wrong["storage_view"][key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.lookup(data, wrong, args)
        wrong = copy.deepcopy(actual)
        wrong["args"][0]["shape"] = [3, 2]
        with self.assertRaises(ValueError):
            self.lookup(data, wrong, args)
        with self.assertRaises(ValueError):
            self.lookup(data, actual, args, {"relation": "detached"})

    def test_source_scope_native_result_and_h800_provenance_are_required(self):
        x = torch.empty(2, 3)
        args = (x, x)
        original = self.artifact(args)
        actual = original["operators"]["is_set_to"][0]["host_measurement"]["inputs"]
        mutations = [
            lambda d: d["_source"]["collector"].update(dirty=True),
            lambda d: d["_tasks"]["is_set_to"].update(selected_case_ids=["missing"]),
            lambda d: d["_tasks"]["is_set_to"].update(catalog_case_ids=[]),
            lambda d: d["operators"]["is_set_to"].append(
                copy.deepcopy(d["operators"]["is_set_to"][0])
            ),
            lambda d: d["operators"]["is_set_to"][0]["host_measurement"].update(
                native_validation=False
            ),
            lambda d: d["operators"]["is_set_to"][0]["host_measurement"][
                "environment"
            ].update(cuda_device="NVIDIA H100"),
            lambda d: d["operators"]["is_set_to"][0].update(dtype="torch.int32"),
        ]
        for mutate in mutations:
            data = copy.deepcopy(original)
            mutate(data)
            with self.subTest(mutation=mutate), self.assertRaises(ValueError):
                self.lookup(data, actual, args)

    def test_declared_scope_rejects_duplicate_and_malformed_ids(self):
        x = torch.empty(2, 3)
        args = (x, x)
        original = self.artifact(args)
        actual = original["operators"]["is_set_to"][0]["host_measurement"]["inputs"]
        for key in ("selected_case_ids", "catalog_case_ids"):
            valid = original["_tasks"]["is_set_to"][key]
            for invalid in (
                valid + [valid[-1]],
                [],
                "source-index-7",
                [None],
                [1],
                [""],
            ):
                data = copy.deepcopy(original)
                data["_tasks"]["is_set_to"][key] = invalid
                with (
                    self.subTest(key=key, invalid=invalid),
                    self.assertRaises(ValueError),
                ):
                    self.lookup(data, actual, args)

    def test_full_scope_cannot_claim_a_partial_catalog(self):
        x = torch.empty(2, 3)
        args = (x, x)
        data = self.artifact(args)
        actual = data["operators"]["is_set_to"][0]["host_measurement"]["inputs"]
        task = data["_tasks"]["is_set_to"]
        for invalid in (None, "unknown", "full_core_catalog"):
            task["scope"] = invalid
            with self.subTest(scope=invalid), self.assertRaises(ValueError):
                self.lookup(data, actual, args)
        del task["scope"]
        with self.assertRaises(ValueError):
            self.lookup(data, actual, args)
        task["scope"] = "full_core_catalog"
        task["catalog_case_ids"] = task["selected_case_ids"][:]
        self.assertEqual(self.lookup(data, actual, args)[0], 0.002)

    def test_selection_cli_rejects_duplicate_operator_keys_before_execution(self):
        with tempfile.TemporaryDirectory() as tmp:
            selection = Path(tmp) / "selection.json"
            selection.write_text('{"is_set_to":["source-a"],"is_set_to":["source-b"]}')
            with patch.object(collector.subprocess, "Popen") as launch:
                with self.assertRaises(SystemExit) as exc:
                    collector.main(
                        [
                            "--flaggems",
                            tmp,
                            "--output",
                            str(Path(tmp) / "result.json"),
                            "--operators",
                            "is_set_to",
                            "--case-selection",
                            str(selection),
                        ]
                    )
                self.assertEqual(exc.exception.code, 2)
                launch.assert_not_called()

    def collect(self, selection, available=("source-0", "source-7")):
        with tempfile.TemporaryDirectory() as tmp:
            r = Path(tmp)
            fg = r / "fg"
            (fg / "benchmark").mkdir(parents=True)
            (fg / "benchmark/base.py").touch()
            (fg / "benchmark/test_is_set_to.py").touch()
            output = r / "run/result.json"
            selection_path = r / "selection.json"
            selection_path.write_text(json.dumps(selection))
            calls = []

            def launch(cmd, **kwargs):
                calls.append(cmd)
                path = Path(cmd[cmd.index("--output") + 1])
                if "--list-cases" in cmd:
                    path.write_text(
                        json.dumps(
                            {
                                "benchmarks": [
                                    {
                                        "op_name": "is_set_to",
                                        "cases": [{"case_id": i} for i in available],
                                    }
                                ]
                            }
                        )
                    )
                else:
                    ids = [
                        cmd[i + 1] for i, x in enumerate(cmd) if x == "--case-id"
                    ] or list(available)
                    path.write_text(
                        json.dumps(
                            {
                                "status": "PASSED",
                                "records": [
                                    {
                                        "operator": "is_set_to",
                                        "case_id": i,
                                        "status": "PASSED",
                                        "latency_ms": 0.002,
                                        "host_measurement": {
                                            "kind": "cuda_dispatch_host_predicate"
                                        },
                                    }
                                    for i in ids
                                ],
                            }
                        )
                    )
                return unittest.mock.Mock(wait=lambda **kwargs: 0)

            with (
                patch.object(
                    collector,
                    "git_info",
                    return_value={"commit": "a" * 40, "dirty": False},
                ),
                patch.object(collector.subprocess, "Popen", side_effect=launch),
            ):
                code = collector.main(
                    [
                        "--flaggems",
                        str(fg),
                        "--output",
                        str(output),
                        "--operators",
                        "is_set_to",
                        "--case-selection",
                        str(selection_path),
                    ]
                )
            return code, json.loads(output.read_text()), calls

    def test_explicit_selection_preserves_full_catalog_and_exact_source_scope(self):
        code, data, calls = self.collect({"is_set_to": ["source-7"]})
        self.assertEqual(code, 0)
        self.assertEqual(len(calls), 2)
        task = data["_tasks"]["is_set_to"]
        self.assertEqual(task["scope"], "explicit_core_selection")
        self.assertEqual(task["selected_case_ids"], ["source-7"])
        self.assertEqual(task["catalog_case_ids"], ["source-0", "source-7"])
        self.assertEqual(
            [x["case_id"] for x in data["operators"]["is_set_to"]], ["source-7"]
        )
        code, data, calls = self.collect({"is_set_to": ["not-in-catalog"]})
        self.assertEqual(code, 1)
        self.assertEqual(len(calls), 1)
        self.assertEqual(data["_tasks"]["is_set_to"]["status"], "failed")


if __name__ == "__main__":
    unittest.main()
