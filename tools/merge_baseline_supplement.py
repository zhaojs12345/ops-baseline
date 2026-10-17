#!/usr/bin/env python3
"""Merge a finished supplement into one JSON, retaining successful base records."""
import argparse
from collections import Counter
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
import json
from pathlib import Path
import tempfile
import os
from contextlib import ExitStack


def records(data):
    for name, entry in data.items():
        if isinstance(entry, dict) and isinstance(entry.get('shapes'), dict):
            for shape, values in entry['shapes'].items():
                for dtype, record in values.items():
                    yield (name, shape, dtype), record


def finished(data):
    collection = data.get('_collection', {})
    names = collection.get('configuration', {}).get('operators', [])
    terminal = {'completed', 'failed', 'profiling_incomplete', 'skipped'}
    if not names or any(collection.get('tasks', {}).get(n, {}).get('status') not in terminal
                        for n in names):
        raise ValueError('collection is unfinished')


def quality(record):
    fields = ('F_cuda', 'F_tensor', 'B_mem', 'U_cuda', 'U_tensor', 'U_mem', 'sm_utilization')
    return {k for k in fields if record.get(k) is not None} if record.get('num_kernels') else set()


def merge(base, supplement, origin):
    finished(base)
    finished(supplement)
    if '_metric_validity_migration' not in base:
        raise ValueError('wait for base validity annotation to finish')
    if base['_collection']['configuration']['repo'] != supplement['_collection']['configuration']['repo']:
        raise ValueError('operator repos differ')
    result = copy.deepcopy(base)
    replaced, added, retained = 0, 0, 0
    for (name, shape, dtype), candidate in records(supplement):
        old = base.get(name, {}).get('shapes', {}).get(shape, {}).get(dtype)
        if old is not None and old.get('config') != candidate.get('config'):
            raise ValueError(f'input configuration differs: {name} {shape} {dtype}')
        # Never overwrite a complete measured baseline, even if timing differs.
        complete = old is not None and old.get('ncu_status') == 'collected'
        better = (old is not None and quality(old) <= quality(candidate) and
                  (quality(old) < quality(candidate) or
                   (candidate.get('ncu_status') == 'collected' and old.get('ncu_status') != 'collected')))
        if old is not None and (complete or not better):
            retained += 1
            continue
        if name in result and result[name].get('native_api') != supplement[name].get('native_api'):
            raise ValueError('native API differs: ' + name)
        result.setdefault(name, {'native_api': supplement[name].get('native_api'), 'shapes': {}})
        new = copy.deepcopy(candidate)
        task = supplement['_collection']['tasks'][name]
        new['collection_origin'] = {'supplement_sha256': origin['supplement_sha256'],
                                    'device': task.get('device'),
                                    'environment': supplement.get('_environments', {}).get(task.get('device'))}
        result[name]['shapes'].setdefault(shape, {})[dtype] = new
        replaced += old is not None
        added += old is None
    # Keep original run/task and migration metadata as historical evidence.
    effective = {name: task['status'] for name, task in base['_collection']['tasks'].items()}
    for name, task in supplement['_collection']['tasks'].items():
        if effective.get(name) != 'completed':
            effective[name] = task['status']
    result['_supplemental_merge'] = {
        **origin, 'merged_at_utc': datetime.now(timezone.utc).isoformat(),
        'added_records': added, 'upgraded_records': replaced, 'retained_base_records': retained,
        'supplement_collection': supplement['_collection'],
        'supplement_environments': supplement.get('_environments', {}),
        'effective_task_statuses': effective,
        'effective_task_counts': dict(Counter(effective.values())),
        'final_ncu_status_counts': dict(Counter(r.get('ncu_status', 'unknown') for _, r in records(result)))}
    # All base keys remain, and every complete base record is byte-for-byte equivalent as an object.
    old_records, merged_records = dict(records(base)), dict(records(result))
    if not old_records.keys() <= merged_records.keys():
        raise ValueError('base records were lost')
    for key, old in old_records.items():
        if old.get('ncu_status') == 'collected' and merged_records[key] != old:
            raise ValueError('successful base record changed: ' + str(key))
    json.dumps(result, allow_nan=False)
    return result


def merge_files(base_path, supplement_path, output):
    base_path, supplement_path, output = map(lambda p: Path(p).resolve(), (base_path, supplement_path, output))
    if output in (base_path, supplement_path):
        raise ValueError('use a new final output path; sources remain unchanged')
    output.parent.mkdir(parents=True, exist_ok=True)
    with ExitStack() as stack:
        for path in sorted({base_path, supplement_path, output}):
            lock = stack.enter_context(path.with_suffix(path.suffix + '.lock').open('a'))
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if output.exists():
            raise ValueError('final output already exists')
        base_raw, supplement_raw = base_path.read_bytes(), supplement_path.read_bytes()
        result = merge(json.loads(base_raw), json.loads(supplement_raw), {
            'base_path': str(base_path), 'supplement_path': str(supplement_path),
            'base_sha256': hashlib.sha256(base_raw).hexdigest(),
            'supplement_sha256': hashlib.sha256(supplement_raw).hexdigest()})
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=output.parent, delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(result, handle, ensure_ascii=False, indent=2, allow_nan=False)
                handle.flush()
                os.fsync(handle.fileno())
            if base_path.read_bytes() != base_raw or supplement_path.read_bytes() != supplement_raw:
                raise ValueError('source changed during merge')
            temporary.replace(output)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
    return result['_supplemental_merge']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', required=True)
    parser.add_argument('--supplement', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(merge_files(args.base, args.supplement, args.output), indent=2))
