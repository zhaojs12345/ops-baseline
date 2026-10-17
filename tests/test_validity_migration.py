"""Offline migration preserves data and refuses active or changing collections."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[1] / 'tools'
sys.path.insert(0, str(TOOLS))
spec = importlib.util.spec_from_file_location('migration_collector', TOOLS / 'collect_baseline_nvidia.py')
c = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'triton': SimpleNamespace()}):
    spec.loader.exec_module(c)
spec = importlib.util.spec_from_file_location('migration', TOOLS / 'annotate_baseline_validity.py')
m = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'collect_baseline_nvidia': c}):
    spec.loader.exec_module(m)


class MigrationTests(unittest.TestCase):
    def data(self):
        return {'_collection': {'configuration': {'operators': ['_reshape_alias_copy']},
                                'tasks': {'_reshape_alias_copy': {'status': 'completed'}}},
                '_environment': {'source': 'old source'},
                '_reshape_alias_copy': {'shapes': {'[1]': {'torch.float16': {
                    'T_us': 123.456, 'latency_ms': .123456, 'config': {'shape': [1]},
                    'num_kernels': 0, 'B_mem': 0, 'U_mem': 0, 'report_path': None}}}}}

    def test_unknown_preserves_latency_and_underscore_operator(self):
        data = self.data()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            raw = json.dumps(data).encode()
            path.write_bytes(raw)
            metadata = m.migrate(path, ensure_idle=lambda data: None)
            result = json.loads(path.read_text())
            r = dict(m.records(result))['_reshape_alias_copy', '[1]', 'torch.float16']
            self.assertEqual(r['ncu_status'], 'unknown')
            self.assertEqual(r['ncu_reason'], 'legacy_reason_unknown')
            self.assertIsNone(r['B_mem'])
            self.assertIsNone(r['U_mem'])
            self.assertEqual(r['T_us'], 123.456)
            self.assertEqual(r['metric_validity']['T_us'], 'measured')
            self.assertEqual(result['_collection'], data['_collection'])
            self.assertEqual(result['_environment'], data['_environment'])
            self.assertEqual(Path(metadata['backup_path']).read_bytes(), raw)
            with self.assertRaisesRegex(ValueError, 'already annotated'):
                m.migrate(path, ensure_idle=lambda data: None)

    def test_unfinished_queue_and_live_process_refuse_migration(self):
        data = self.data()
        for status in ('running', None):
            data['_collection']['tasks']['_reshape_alias_copy']['status'] = status
            with self.assertRaisesRegex(ValueError, 'unfinished'):
                m.assert_idle(data)
        data = self.data()
        for command in ('42 python /work/tools/collect_baseline_nvidia.py --resume',
                        '42 /opt/ncu --export result python work.py'):
            with patch.object(m.subprocess, 'check_output', return_value=command):
                with self.assertRaisesRegex(ValueError, 'still running'):
                    m.assert_idle(data)
        with patch.object(m.subprocess, 'check_output', return_value='42 ncu --import old.ncu-rep'):
            m.assert_idle(data)

    def test_offline_import_preserves_measured_zero(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / 'old.ncu-rep'
            report.write_bytes(b'original report')
            old = dict(next(m.records(self.data()))[1], report_path=str(report))
            def load(path, dtype):
                r = c._empty_result()
                r.update(B_mem=0, num_kernels=1, ncu_status='partial')
                r['metric_validity'].update(B_mem='measured', num_kernels='measured')
                return r
            _, result = m.annotate_record((('op', 'shape', 'torch.float16'), old), load)
            self.assertEqual(result['B_mem'], 0)
            self.assertEqual(result['metric_validity']['B_mem'], 'measured')
            self.assertEqual(result['ncu_evidence']['sha256'], m.digest(b'original report'))

    def test_log_failure_is_scoped_to_its_record(self):
        with tempfile.TemporaryDirectory() as directory:
            log = Path(directory) / 'collector.log'
            log.write_text('  ✗ NCU 超时: [1] torch.float16\n  [1] torch.float16: 0.1 ms\n'
                           '  ✗ NCU 未生成报告文件: ignored\n  [2] torch.float16: 0.1 ms\n')
            data = self.data()
            data['_collection']['tasks']['_reshape_alias_copy']['log'] = str(log)
            failures = m.log_evidence(data)
            key, old = next(m.records(data))
            self.assertEqual(m.annotate_record((key, old), failures=failures)[1]['ncu_status'], 'failed')
            self.assertEqual(m.annotate_record(((key[0], '[2]', key[2]), old), failures=failures)[1]['ncu_status'], 'unknown')

    def test_source_change_prevents_replacement(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'result.json'
            path.write_text(json.dumps(self.data()))
            calls = []
            def idle(data):
                calls.append(1)
                if len(calls) == 2:
                    path.write_text('new writer data')
            with self.assertRaisesRegex(ValueError, 'source JSON changed'):
                m.migrate(path, ensure_idle=idle)
            self.assertEqual(path.read_text(), 'new writer data')
            self.assertEqual(len(list(Path(directory).glob('*.pre-validity.*.json'))), 1)


if __name__ == '__main__':
    unittest.main()
