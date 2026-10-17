#!/usr/bin/env python3
"""Collect selected operators separately, then merge after base annotation ends."""
import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from merge_baseline_supplement import finished, merge_files


def run(args):
    base, supplement, output = (Path(p).resolve() for p in (args.base, args.supplement, args.output))
    if len({base, supplement, output}) != 3:
        raise ValueError('base, supplement and final output must be distinct')
    output.parent.mkdir(parents=True, exist_ok=True)
    state_path = output.with_suffix(output.suffix + '.pipeline.json')
    def state(phase, **details):
        value = {'phase': phase, 'updated_at_utc': datetime.now(timezone.utc).isoformat(),
                 'pid': os.getpid(), 'base': str(base), 'supplement': str(supplement),
                 'output': str(output), **details}
        temporary = state_path.with_suffix(state_path.suffix + '.tmp')
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
        temporary.replace(state_path)
        print(json.dumps(value), flush=True)
    with output.with_suffix(output.suffix + '.pipeline.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if supplement.exists() or output.exists():
            raise ValueError('supplement/final output already exists; refusing another run')
        finished(json.loads(base.read_text()))
        # The running legacy annotation performs a conservative container-wide
        # collector check before replacement. Let it finish before launching GPUs.
        state('waiting_for_base_validity_before_collection')
        deadline = time.monotonic() + args.wait_hours * 3600
        previous_mtime = None
        while True:
            mtime = base.stat().st_mtime_ns
            if mtime != previous_mtime:
                if '_metric_validity_migration' in json.loads(base.read_text()):
                    break
                previous_mtime = mtime
            if time.monotonic() >= deadline:
                state('failed', error='base annotation has not finished; collection not started')
                raise TimeoutError('base annotation has not finished; collection not started')
            time.sleep(60)
        state('waiting_for_gpu_idle')
        while subprocess.check_output(
                ['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader,nounits'],
                text=True, timeout=30).strip():
            if time.monotonic() >= deadline:
                state('failed', error='GPUs still have compute processes; collection not started')
                raise TimeoutError('GPUs are not idle')
            time.sleep(60)
        command = [sys.executable, '-u', str(Path(__file__).with_name('collect_baseline_nvidia.py')),
                   '--repo', args.repo, '--devices', args.devices, '--ops', args.ops,
                   '--op-timeout', str(args.op_timeout), '--ncu-timeout', str(args.ncu_timeout),
                   '--output', str(supplement)]
        state('collecting', command=command)
        try:
            result = subprocess.run(command)
            # Nonzero exit can still contain useful partial records; never call it complete.
            finished(json.loads(supplement.read_text()))
            state('merging', collection_returncode=result.returncode)
            summary = merge_files(base, supplement, output)
            state('finished', collection_returncode=result.returncode,
                  added_records=summary['added_records'], upgraded_records=summary['upgraded_records'],
                  effective_task_counts=summary['effective_task_counts'],
                  final_ncu_status_counts=summary['final_ncu_status_counts'])
        except Exception as exc:
            state('failed', error=f'{type(exc).__name__}: {exc}')
            raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--supplement', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--repo', default='pytorch')
    parser.add_argument('--devices', default='0,1,2,3,4,5,6,7')
    parser.add_argument('--ops', required=True)
    parser.add_argument('--op-timeout', type=float, default=14400)
    parser.add_argument('--ncu-timeout', type=float, default=1800)
    parser.add_argument('--wait-hours', type=float, default=24)
    run(parser.parse_args())
