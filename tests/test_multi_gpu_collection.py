# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Exercise scheduling and recovery without CUDA or torch imports."""
import importlib.util
import json
import os
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location(
    'parallel', Path(__file__).resolve().parents[1] / 'tools/collect_baseline_multi_gpu.py')
m = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(m)


class CollectionTests(unittest.TestCase):
    def args(self, directory, **kwargs):
        values = dict(devices='0,1', repo='pytorch', output=str(Path(directory)/'data.json'),
                      ops='abs,add,mul', whitelist=None, blacklist=None, no_ncu=False,
                      op_timeout=5, reference_chip='H800', resume=False)
        values.update(kwargs)
        return SimpleNamespace(**values)

    def test_dynamic_queue_isolation_merge_and_resume(self):
        calls = []
        active = set()
        lock = threading.Lock()
        simultaneous = threading.Event()
        def execute(command, environment, log, timeout, stop):
            name = command[command.index('--ops')+1]
            device = environment['CUDA_VISIBLE_DEVICES']
            with lock:
                self.assertNotIn(device, active)
                active.add(device)
                calls.append((name, device))
                if len(active) == 2:
                    simultaneous.set()
            self.assertTrue(simultaneous.wait(3))
            time.sleep(.03)
            out = command[command.index('--output')+1]
            m.atomic_json(out, {name: {'shapes': {'[1]': {'fp32': {'num_kernels': 1}}}},
                                '_environment': {'device': device},
                                '_scaling_factors': {'reference_chip': 'H800'}})
            with lock:
                active.remove(device)
            return 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory)
            self.assertEqual(m.collect(args, execute=execute), 0)
            self.assertEqual(sorted(n for n, _ in calls), ['abs', 'add', 'mul'])
            data = json.loads(Path(args.output).read_text())
            self.assertEqual(set(data['_environments']), {'0', '1'})
            self.assertEqual(len({t['result'] for t in data['_collection']['tasks'].values()}), 3)
            with self.assertRaisesRegex(ValueError, 'output exists'):
                m.collect(args, execute=execute)
            args.resume = True
            self.assertEqual(m.collect(args, execute=execute), 0)
            self.assertEqual(len(calls), 3)
            args.no_ncu = True
            with self.assertRaisesRegex(ValueError, 'configuration'):
                m.collect(args, execute=execute)

    def test_failures_and_missing_profiles_are_not_completed(self):
        def execute(command, environment, log, timeout, stop):
            name = command[command.index('--ops')+1]
            out = command[command.index('--output')+1]
            if name == 'mul':
                return 'timeout'
            m.atomic_json(out, {name: {'shapes': {'[1]': {'fp32': {'num_kernels': 0}}}}})
            return 1 if name == 'abs' else 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory)
            self.assertEqual(m.collect(args, execute=execute), 1)
            tasks = json.loads(Path(args.output).read_text())['_collection']['tasks']
            self.assertEqual(tasks['abs']['status'], 'failed')
            self.assertEqual(tasks['add']['status'], 'profiling_incomplete')
            self.assertEqual(tasks['mul']['returncode'], 'timeout')

    def test_device_and_operator_validation(self):
        with patch.dict(os.environ, {'CUDA_VISIBLE_DEVICES': '2,5'}, clear=True):
            self.assertEqual(m.parse_devices('2,5'), ['2', '5'])
            for value in ['2,2', '2,', '0,1']:
                with self.assertRaises(ValueError):
                    m.parse_devices(value)
        with self.assertRaises(ValueError):
            m.select_operators('pytorch', {'does-not-exist'})

    def test_partial_metrics_remain_incomplete_and_cannot_replace_measured_fields(self):
        old = {'shapes': {'[1]': {'fp32': {'num_kernels': 1, 'B_mem': 64}}}}
        new = {'shapes': {'[1]': {'fp32': {'num_kernels': 1, 'B_mem': None,
                                         'ncu_status': 'partial'}}}}
        self.assertFalse(m.covers_previous(old, new))
        self.assertTrue(m.covers_previous(new, old))
        def execute(command, environment, log, timeout, stop):
            m.atomic_json(command[command.index('--output')+1], {'abs': new})
            return 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory, ops='abs', devices='0')
            self.assertEqual(m.collect(args, execute=execute), 1)
            result = json.loads(Path(args.output).read_text())
            self.assertEqual(result['_collection']['tasks']['abs']['status'], 'profiling_incomplete')

    def test_long_operator_selection_is_not_a_filename(self):
        names = ['operator_' + str(i) for i in range(100)]
        self.assertEqual(m.name_list(','.join(names)), set(names))

    def test_custom_ncu_timeout_is_forwarded_and_part_of_resume_signature(self):
        def execute(command, environment, log, timeout, stop):
            self.assertEqual(command[command.index('--ncu-timeout')+1], '1800')
            m.atomic_json(command[command.index('--output')+1],
                          {'abs': {'shapes': {'[1]': {'fp32': {'num_kernels': 1}}}}})
            return 0
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory, devices='0', ops='abs', ncu_timeout=1800)
            self.assertEqual(m.collect(args, execute=execute), 0)
            args.resume = True
            args.ncu_timeout = 600
            with self.assertRaisesRegex(ValueError, 'configuration'):
                m.collect(args, execute=execute)

    def test_failed_retry_preserves_partial_measurements_and_artifacts(self):
        for retry_writes_partial in (False, True):
            with self.subTest(retry_writes_partial=retry_writes_partial), \
                 tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {}, clear=True):
                args = self.args(directory, devices='0', ops='abs')
                original = {'shapes': {'[1]': {'fp32': {'num_kernels': 1}},
                                       '[2]': {'fp32': {'num_kernels': 1}}}}
                def initial(command, environment, log, timeout, stop):
                    m.atomic_json(command[command.index('--output')+1], {'abs': original})
                    return 1
                self.assertEqual(m.collect(args, execute=initial), 1)
                data = json.loads(Path(args.output).read_text())
                first_result = Path(data['_collection']['tasks']['abs']['result'])
                args.resume = True
                def retry(command, environment, log, timeout, stop):
                    if retry_writes_partial:
                        m.atomic_json(command[command.index('--output')+1],
                                      {'abs': {'shapes': {'[1]': original['shapes']['[1]']}}})
                    return 1
                self.assertEqual(m.collect(args, execute=retry), 1)
                data = json.loads(Path(args.output).read_text())
                self.assertEqual(data.get('abs'), original)
                self.assertTrue(first_result.is_file())
                task = data['_collection']['tasks']['abs']
                self.assertNotEqual(task['result'], str(first_result))
                self.assertEqual(task['retained_result']['result'], str(first_result))
                self.assertEqual(len(task['previous_attempts']), 1)
                replacement = {'shapes': dict(original['shapes'],
                                              **{'[3]': {'fp32': {'num_kernels': 1}}})}
                def successful_retry(command, environment, log, timeout, stop):
                    current = json.loads(Path(args.output).read_text())
                    # The checkpoint may still precede the running event; either
                    # state must retain the old result's attribution.
                    origin = current['_collection']['tasks']['abs']['retained_result']
                    self.assertEqual(origin['result'], str(first_result))
                    m.atomic_json(command[command.index('--output')+1], {'abs': replacement})
                    return 0
                self.assertEqual(m.collect(args, execute=successful_retry), 0)
                data = json.loads(Path(args.output).read_text())
                self.assertEqual(data['abs'], replacement)
                self.assertTrue(first_result.is_file())
                self.assertNotIn('retained_result', data['_collection']['tasks']['abs'])
                self.assertEqual(len(data['_collection']['tasks']['abs']['previous_attempts']), 2)

    def test_resume_clears_terminal_state_before_starting_worker(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory, devices='0', ops='abs')
            self.assertEqual(m.collect(args, execute=lambda *unused: 'interrupted'), 1)
            data = json.loads(Path(args.output).read_text())
            data['_collection']['interrupted'] = True
            m.atomic_json(args.output, data)
            args.resume = True
            def execute(command, environment, log, timeout, stop):
                current = json.loads(Path(args.output).read_text())['_collection']
                self.assertNotIn('finished_at_utc', current)
                self.assertFalse(current['interrupted'])
                m.atomic_json(command[command.index('--output')+1],
                              {'abs': {'shapes': {'[1]': {'fp32': {'num_kernels': 1}}}}})
                return 0
            self.assertEqual(m.collect(args, execute=execute), 0)

    def test_checkpoint_failure_stops_consumer_before_next_task(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            args = self.args(directory, devices='0')
            calls, stopped = [], []
            real_write = m.atomic_json
            writes = 0
            def checkpoint(path, data):
                nonlocal writes
                writes += 1
                if writes == 2:
                    raise OSError('simulated disk full')
                real_write(path, data)
            def execute(command, environment, log, timeout, stop):
                calls.append(command[command.index('--ops')+1])
                stopped.append(stop.wait(.5))
                return 1
            with patch.object(m, 'atomic_json', side_effect=checkpoint):
                with self.assertRaisesRegex(OSError, 'disk full'):
                    m.collect(args, execute=execute)
            self.assertEqual(calls, ['abs'])
            self.assertEqual(stopped, [True])

    def test_custom_reports_are_isolated_by_full_output_path(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {}, clear=True):
            reports = []
            def execute(command, environment, log, timeout, stop):
                reports.append(command[command.index('--report-dir')+1])
                m.atomic_json(command[command.index('--output')+1],
                              {'abs': {'shapes': {'[1]': {'fp32': {'num_kernels': 1}}}}})
                return 0
            for parent in ['a', 'b']:
                args = self.args(directory, devices='0', ops='abs',
                                 output=str(Path(directory)/parent/'data.json'),
                                 report_dir=str(Path(directory)/'reports'))
                self.assertEqual(m.collect(args, execute=execute), 0)
            self.assertNotEqual(reports[0], reports[1])

    def test_timeout_kills_descendants(self):
        import subprocess
        import sys
        with tempfile.TemporaryDirectory() as directory:
            pidfile = Path(directory)/'child.pid'
            code = ("import subprocess,sys,time; from pathlib import Path; "
                    "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(60)']); "
                    f"Path({str(pidfile)!r}).write_text(str(p.pid)); time.sleep(60)")
            result = m.run_process([sys.executable, '-c', code], dict(os.environ),
                                   Path(directory)/'log', 1, threading.Event())
            self.assertEqual(result, 'timeout')
            pid = int(pidfile.read_text())
            # A killed child may briefly remain as an OS zombie; it must not run.
            state = subprocess.run(['ps', '-o', 'stat=', '-p', str(pid)],
                                   capture_output=True, text=True).stdout.strip()
            self.assertTrue(not state or state.startswith('Z'), state)


if __name__ == '__main__':
    unittest.main()
