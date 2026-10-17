"""NCU outcome and missing-counter semantics, without starting CUDA or NCU."""
import csv
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
spec = importlib.util.spec_from_file_location(
    'collector_validity', Path(__file__).resolve().parents[1] / 'tools/collect_baseline_nvidia.py')
c = importlib.util.module_from_spec(spec)
with patch.dict(sys.modules, {'torch': SimpleNamespace(), 'triton': SimpleNamespace()}):
    spec.loader.exec_module(c)


class NcuValidityTests(unittest.TestCase):
    def profile(self, export=None, imported=None, csv_text=None, dtype='torch.float16', ncu_timeout=600):
        def run(command, **kwargs):
            if '--export' in command:
                self.assertEqual(kwargs['timeout'], ncu_timeout)
                if isinstance(export, Exception):
                    raise export
                if csv_text is not None:
                    Path(command[command.index('--export') + 1] + '.ncu-rep').touch()
                return export or subprocess.CompletedProcess(command, 0, '', '')
            if isinstance(imported, Exception):
                raise imported
            return imported or subprocess.CompletedProcess(command, 0, csv_text, '')
        with tempfile.TemporaryDirectory() as directory, patch.object(c.subprocess, 'run', side_effect=run):
            result = c.profile_with_ncu('example', 'pass', [8, 8], dtype, directory, ncu_timeout=ncu_timeout)
        return c._finalize_record(.125, result)

    def csv(self, missing=None, rows=1):
        metrics = [*c.FLOP_CUDA_CORE, c.FLOP_TENSOR_CORE['torch.float16'],
                   c.MEMORY_BYTES, *c.UTIL.values(), c.SM_UTIL, c.DURATION]
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([c.KERNEL_NAME_COL, *metrics])
        for i in range(rows):
            writer.writerow(['copy_kernel', *['N/A' if m == missing and i == 0 else 0
                                             for m in metrics]])
        return output.getvalue()

    def assert_unavailable(self, record, status):
        self.assertEqual(record['ncu_status'], status)
        for field in ('F_cuda', 'F_tensor', 'B_mem', 'U_cuda', 'U_tensor', 'U_mem',
                      'U_bottle_neck', 'memory_bytes', 'util_memory_bw', 'sm_utilization'):
            self.assertIsNone(record[field], field)
            self.assertEqual(record['metric_validity'][field], 'unavailable')
        self.assertEqual(record['T_us'], 125)
        self.assertEqual(record['metric_validity']['T_us'], 'measured')
        json.dumps(record, allow_nan=False)

    def test_explicit_no_kernels_is_distinct_from_missing_report(self):
        for stream in ('stdout', 'stderr'):
            with self.subTest(stream=stream):
                record = self.profile(export=subprocess.CompletedProcess(
                    [], 0, **{stream: '==WARNING== No kernels were profiled.'}))
                self.assert_unavailable(record, 'no_kernel')
                self.assertEqual(record['num_kernels'], 0)
                self.assertIn('No kernels', record['ncu_diagnostic'])
        missing = self.profile()
        self.assert_unavailable(missing, 'failed')
        self.assertEqual(missing['ncu_reason'], 'missing_report')
        self.assertIsNone(missing['num_kernels'])

    def test_export_failures_and_errors_are_not_no_kernel(self):
        cases = [(FileNotFoundError(), 'ncu_not_found'),
                 (subprocess.TimeoutExpired('ncu', 600), 'export_timeout'),
                 (subprocess.CompletedProcess([], 1, 'No kernels were profiled.', 'permission denied'), 'export_failed'),
                 (subprocess.CompletedProcess([], 0, '==ERROR== Permission denied. No kernels were profiled.', ''), 'export_failed')]
        for export, reason in cases:
            with self.subTest(reason=reason):
                record = self.profile(export=export)
                self.assert_unavailable(record, 'failed')
                self.assertEqual(record['ncu_reason'], reason)

    def test_import_failure_timeout_and_empty_report(self):
        for imported, reason in [(subprocess.CompletedProcess([], 1, '', 'bad report'), 'import_failed'),
                                 (subprocess.TimeoutExpired('ncu', 120), 'import_timeout')]:
            record = self.profile(imported=imported, csv_text=self.csv())
            self.assert_unavailable(record, 'failed')
            self.assertEqual(record['ncu_reason'], reason)
            self.assertIsNotNone(record['report_path'])
        self.assertEqual(self.profile(csv_text='malformed CSV')['ncu_reason'], 'no_kernel_rows')

    def test_measured_zero_is_preserved(self):
        record = self.profile(csv_text=self.csv())
        self.assertEqual(record['ncu_status'], 'collected')
        self.assertEqual(record['num_kernels'], 1)
        self.assertEqual(record['B_mem'], 0)
        self.assertEqual(record['metric_validity']['B_mem'], 'measured')
        self.assertEqual(record['per_kernel'][0]['T_us'], 125)
        self.assertEqual(record['per_kernel'][0]['metric_validity']['T_us'], 'estimated')

    def test_missing_counter_does_not_become_zero_or_partial_sum(self):
        record = self.profile(csv_text=self.csv(missing=c.MEMORY_BYTES, rows=2))
        self.assertEqual(record['ncu_status'], 'partial')
        self.assertIsNone(record['B_mem'])
        self.assertIsNone(record['memory_bytes'])
        self.assertIsNone(record['per_kernel'][0]['B_mem'])
        self.assertEqual(record['per_kernel'][1]['B_mem'], 0)
        self.assertEqual(record['U_mem'], 0)
        for cell in ('N/A', 'nan', 'inf', ''):
            csv_text = self.csv().replace('copy_kernel,0,', 'copy_kernel,' + cell + ',')
            self.assertIsNone(self.profile(csv_text=csv_text)['F_cuda'])

    def test_disabled_has_null_metrics_and_measured_latency(self):
        module = SimpleNamespace(OP_NAME='example', __name__='example',
                                 DTYPES=['torch.float16'], native=lambda: lambda: None,
                                 grid=lambda: [{}], build_inputs=lambda *args: ((), {}),
                                 key_shape=lambda binding: '[1]')
        fake_torch = SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
        fake_triton = SimpleNamespace(testing=SimpleNamespace(do_bench=lambda *args, **kwargs: .125))
        with patch.object(c, 'torch', fake_torch), patch.object(c, 'triton', fake_triton), \
             patch.object(c, 'profile_with_ncu', side_effect=AssertionError('NCU must stay disabled')):
            entry = c._collect_one_op_ops(module, ncu_enabled=False, report_dir=None)
        record = entry['shapes']['[1]']['torch.float16']
        self.assert_unavailable(record, 'disabled')
        self.assertIsNone(record['num_kernels'])

    def test_custom_ncu_timeout_reaches_export_and_worker(self):
        self.assertEqual(self.profile(csv_text=self.csv(), ncu_timeout=1800)['ncu_status'], 'collected')
        with patch.object(c.subprocess, 'run', return_value=SimpleNamespace(returncode=0)) as run:
            c._spawn_op_worker('abs', True, '/tmp/reports', None, 14400, ncu_timeout=1800)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index('--ncu-timeout')+1], '1800')
        for timeout in (0, -1, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                self.profile(ncu_timeout=timeout)

    def test_absent_column_and_missing_duration_are_not_filled(self):
        rows = list(csv.reader(io.StringIO(self.csv(rows=2))))
        index = rows[0].index(c.MEMORY_BYTES)
        for row in rows:
            row.pop(index)
        output = io.StringIO()
        csv.writer(output).writerows(rows)
        record = self.profile(csv_text=output.getvalue())
        self.assertIsNone(record['B_mem'])
        self.assertIsNone(record['per_kernel'][1]['B_mem'])
        self.assertEqual(record['ncu_status'], 'partial')
        duration = self.profile(csv_text=self.csv(missing=c.DURATION, rows=2))
        self.assertIsNone(duration['per_kernel'][0]['T_us'])
        self.assertIsNone(duration['per_kernel'][1]['T_us'])
        self.assertEqual(duration['T_us'], 125)
        util = self.profile(csv_text=self.csv(missing=c.UTIL['memory_bw']))
        self.assertIsNone(util['U_mem'])
        self.assertIsNone(util['U_bottle_neck'])
        self.assertIsNone(util['bottle_neck_unit'])

    def test_unrequested_tensor_counter_is_not_assumed_zero(self):
        record = self.profile(csv_text=self.csv(), dtype='torch.float32')
        self.assertEqual(record['ncu_status'], 'collected')
        self.assertIsNone(record['F_tensor'])
        self.assertIsNone(record['flops_tensor_core'])
        self.assertEqual(record['metric_validity']['F_tensor'], 'not_collected')
        self.assertIsNone(record['per_kernel'][0]['F_tensor'])
        self.assertEqual(record['per_kernel'][0]['metric_validity']['F_tensor'], 'not_collected')


if __name__ == '__main__':
    unittest.main()
