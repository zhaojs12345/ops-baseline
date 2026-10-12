import importlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import collection_environment as env


class EnvironmentTests(unittest.TestCase):
    def torch(self):
        props = SimpleNamespace(name='Test GPU', total_memory=123,
                                major=9, minor=0, multi_processor_count=132,
                                uuid='GPU-selected')
        return SimpleNamespace(
            cuda=SimpleNamespace(current_device=lambda: 0,
                                 get_device_properties=lambda i: props,
                                 is_available=lambda: True,
                                 get_device_name=lambda: 'NVIDIA Test'),
            version=SimpleNamespace(cuda='12.8'))

    def test_selected_device_and_json_roundtrip(self):
        calls = []
        def command(args, cwd=None):
            calls.append(args)
            return '' if 'status' in args else 'test-value'
        with patch.object(env, '_command', side_effect=command), \
             patch.object(env, '_version', return_value=None), \
             patch.dict('os.environ', {'CUDA_VISIBLE_DEVICES': '3'}):
            result = env.collect_environment(self.torch(), {'repo': 'vllm'})
        gpu = result['hardware']['gpu']
        self.assertEqual(gpu['cuda_device_index'], 0)
        self.assertEqual(gpu['uuid'], 'GPU-selected')
        self.assertEqual(result['environment_variables']['CUDA_VISIBLE_DEVICES'], '3')
        self.assertIn(['nvidia-smi', '--id', 'GPU-selected',
                       '--query-gpu=driver_version', '--format=csv,noheader'], calls)
        self.assertFalse(result['source']['git_dirty'])
        self.assertEqual(json.loads(json.dumps(result)), result)

    def test_unavailable_probes_are_null(self):
        with patch.object(env, '_command', return_value=None), \
             patch.object(env, '_version', return_value=None):
            result = env.collect_environment(SimpleNamespace(), {})
        self.assertIsNone(result['hardware']['gpu']['name'])
        self.assertIsNone(result['software']['ncu_version'])
        self.assertIsNone(result['software']['packages']['vllm'])
        self.assertIsNone(result['source']['git_dirty'])

    def test_commands_are_bounded_and_nonfatal(self):
        for error in (FileNotFoundError(), subprocess.TimeoutExpired('ncu', 5),
                      subprocess.CalledProcessError(1, 'git')):
            with patch.object(env.subprocess, 'run', side_effect=error) as run:
                self.assertIsNone(env._command(['missing']))
                self.assertEqual(run.call_args.kwargs['timeout'], 5)

    def test_environment_survives_worker_merge_and_empty_run(self):
        # Import without loading CUDA libraries; the orchestration itself is real.
        with patch.dict(sys.modules, {'torch': self.torch(), 'triton': SimpleNamespace()}):
            collector = importlib.import_module('collect_baseline_nvidia')
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / 'baseline.json'
            for discovered in ([], [('example', object())]):
                def worker(*args, **kwargs):
                    # Metadata is already durable before starting the first worker.
                    self.assertIn('_environment', json.loads(output.read_text()))
                    return {'shapes': {}}, True
                with patch.object(collector, '_discover_ops', return_value=discovered), \
                     patch.object(collector, 'collect_environment', return_value={'schema_version': 1}), \
                     patch.object(collector, 'build_scaling_factors', return_value={'_error': 'test'}), \
                     patch.object(collector, '_spawn_op_worker', side_effect=worker):
                    collector.collect_baseline(output, ncu_enabled=False)
                result = json.loads(output.read_text())
                self.assertEqual(result['_environment'], {'schema_version': 1})
                self.assertEqual('example' in result, bool(discovered))


if __name__ == '__main__':
    unittest.main()
