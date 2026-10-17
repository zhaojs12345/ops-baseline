import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location('merge', Path(__file__).resolve().parents[1] / 'tools/merge_baseline_supplement.py')
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import collect_baseline_supplement as pipeline


class MergeTests(unittest.TestCase):
    def data(self):
        return {'_collection': {'configuration': {'repo': 'pytorch', 'operators': ['_op']},
                                'tasks': {'_op': {'status': 'completed', 'device': '0'}}},
                '_environments': {'0': {'gpu': 'H800'}},
                '_metric_validity_migration': {'source_sha256': 'original'},
                '_op': {'native_api': 'torch.test', 'shapes': {'[1]': {'fp32': {
                    'ncu_status': 'collected', 'num_kernels': 1, 'B_mem': 16, 'T_us': 10,
                    'config': {'shape': [1]}}}}}}

    def test_successful_records_are_retained_and_missing_records_added(self):
        base, retry = self.data(), self.data()
        retry['_op']['shapes']['[1]']['fp32']['T_us'] = 1
        retry['_op']['shapes']['[2]'] = copy.deepcopy(retry['_op']['shapes']['[1]'])
        merged = m.merge(base, retry, {'supplement_sha256': 'retry'})
        self.assertEqual(merged['_op']['shapes']['[1]'], base['_op']['shapes']['[1]'])
        self.assertIn('[2]', merged['_op']['shapes'])
        self.assertEqual(merged['_collection'], base['_collection'])
        self.assertEqual(merged['_supplemental_merge']['added_records'], 1)
        self.assertEqual(merged['_op']['shapes']['[2]']['fp32']['collection_origin']['environment'], {'gpu': 'H800'})

    def test_invalid_record_upgraded_but_poorer_retry_cannot_downgrade(self):
        base, retry = self.data(), self.data()
        r = base['_op']['shapes']['[1]']['fp32']
        r.update(ncu_status='unknown', num_kernels=None, B_mem=None)
        merged = m.merge(base, retry, {'supplement_sha256': 'retry'})
        self.assertEqual(merged['_supplemental_merge']['upgraded_records'], 1)
        self.assertEqual(merged['_op']['shapes']['[1]']['fp32']['B_mem'], 16)
        reversed_result = m.merge(retry, base, {'supplement_sha256': 'retry'})
        self.assertEqual(reversed_result['_op'], retry['_op'])

    def test_mismatch_unfinished_and_unannotated_sources_are_rejected(self):
        base, retry = self.data(), self.data()
        retry['_op']['shapes']['[1]']['fp32']['config'] = {'shape': [2]}
        with self.assertRaisesRegex(ValueError, 'configuration'):
            m.merge(base, retry, {'supplement_sha256': 'retry'})
        retry = self.data()
        retry['_collection']['tasks']['_op']['status'] = 'running'
        with self.assertRaisesRegex(ValueError, 'unfinished'):
            m.merge(base, retry, {'supplement_sha256': 'retry'})
        base.pop('_metric_validity_migration')
        with self.assertRaises(ValueError):
            m.merge(base, self.data(), {'supplement_sha256': 'retry'})

    def test_pipeline_collects_separately_and_merges_without_changing_base(self):
        with tempfile.TemporaryDirectory() as directory:
            base, retry, final = [Path(directory) / (n + '.json') for n in ('base', 'retry', 'final')]
            base.write_text(json.dumps(self.data()))
            original = base.read_bytes()
            waiting = self.data()
            waiting.pop('_metric_validity_migration')
            base.write_text(json.dumps(waiting))
            def annotation_finished(seconds):
                # A GPU collector must not have started while the old migration runs.
                self.assertFalse(retry.exists())
                base.write_bytes(original)
            def collect(command):
                self.assertEqual(command[command.index('--ncu-timeout')+1], '1800')
                self.assertEqual(command[command.index('--op-timeout')+1], '14400')
                data = self.data()
                data['_op']['shapes']['[2]'] = copy.deepcopy(data['_op']['shapes']['[1]'])
                retry.write_text(json.dumps(data))
                return SimpleNamespace(returncode=0)
            args = SimpleNamespace(base=str(base), supplement=str(retry), output=str(final),
                                   repo='pytorch', devices='0,1,2,3,4,5,6,7', ops='_op',
                                   op_timeout=14400, ncu_timeout=1800, wait_hours=24)
            with patch.object(pipeline.subprocess, 'run', side_effect=collect), \
                 patch.object(pipeline.subprocess, 'check_output', return_value=''), \
                 patch.object(pipeline.time, 'sleep', side_effect=annotation_finished):
                pipeline.run(args)
            self.assertEqual(base.read_bytes(), original)
            self.assertEqual(json.loads(final.with_suffix('.json.pipeline.json').read_text())['phase'], 'finished')
            self.assertIn('[2]', json.loads(final.read_text())['_op']['shapes'])


if __name__ == '__main__':
    unittest.main()
