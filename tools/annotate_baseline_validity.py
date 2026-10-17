#!/usr/bin/env python3
"""Annotate a finished legacy collection using offline NCU imports only.

The original bytes are backed up before an in-place, atomic replacement. No
collection source, task history, latency or environment metadata is rewritten.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import tempfile

import collect_baseline_nvidia as collector


def records(data):
    for name, entry in data.items():
        if isinstance(entry, dict) and isinstance(entry.get('shapes'), dict):
            for shape, values in entry['shapes'].items():
                for dtype, record in values.items():
                    yield (name, shape, dtype), record


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def assert_idle(data):
    collection = data.get('_collection', {})
    tasks = collection.get('tasks', {})
    names = collection.get('configuration', {}).get('operators', [])
    terminal = {'completed', 'profiling_incomplete', 'failed', 'skipped'}
    if not names or any(tasks.get(n, {}).get('status') not in terminal for n in names):
        raise ValueError('operator queue is unfinished or cannot be verified')
    # Do not trust historical finished_at/interrupted flags or a potentially reused PID.
    process_list = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    for line in process_list.splitlines():
        pid, _, command = line.strip().partition(' ')
        if pid == str(os.getpid()):
            continue
        try:
            tokens = shlex.split(command)
        except ValueError:
            raise ValueError('cannot verify process list safely')
        names = {Path(token).name for token in tokens}
        if 'collect_baseline_nvidia.py' in names or (
                'ncu' in names and '--import' not in tokens):
            raise ValueError('collector or NCU workload is still running')


def import_report(path, dtype):
    try:
        imported = subprocess.run(
            ['ncu', '--import', str(path), '--csv', '--page', 'raw'],
            capture_output=True, text=True, timeout=120,
            env=dict(os.environ, CUDA_VISIBLE_DEVICES=''))
    except (OSError, subprocess.TimeoutExpired) as exc:
        return collector._empty_result(str(path), reason='offline_import_failed',
                                       diagnostic=str(exc)[:2000])
    diagnostic = '\n'.join(s for s in (imported.stdout, imported.stderr) if s)
    if imported.returncode or '==ERROR==' in diagnostic:
        return collector._empty_result(str(path), reason='offline_import_failed',
                                       diagnostic=diagnostic[:2000])
    return collector.parse_ncu_report(imported.stdout, dtype, str(path))


def log_evidence(data):
    """Associate sequential collector diagnostics with the next measured record."""
    evidence = {}
    for name, task in data.get('_collection', {}).get('tasks', {}).items():
        log = Path(task['log']) if task.get('log') else None
        if not log or not log.is_file():
            continue
        block = []
        for line in log.read_text(errors='replace').splitlines():
            match = re.match(r'^\s+(.+) (torch\.[A-Za-z0-9_]+): .* ms(?:\s|$)', line)
            if match:
                text = '\n'.join(block)
                # A missing-report line by itself is deliberately not a failure cause.
                failed = any(marker in text for marker in
                             ('NCU profiling 失败', 'NCU import 失败', 'NCU 超时', '未找到 ncu 命令'))
                if failed:
                    evidence[name, match[1], match[2]] = {
                        'path': str(log), 'diagnostic': text[-2000:]}
                block = []
            else:
                block.append(line)
    return evidence


def annotate_record(item, load_report=import_report, failures=None):
    key, old = item
    report = Path(old['report_path']) if old.get('report_path') else None
    evidence = {'kind': 'legacy_record', 'original_num_kernels': old.get('num_kernels')}
    if report and report.is_file():
        before = report.stat()
        sha = digest(report.read_bytes())
        ncu = load_report(report, key[2])
        after = report.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError('report changed during offline import: ' + str(report))
        evidence = {'kind': 'offline_report_import', 'path': str(report),
                    'sha256': sha, 'size_bytes': before.st_size}
    elif failures and key in failures:
        ncu = collector._empty_result(reason='legacy_ncu_failure',
                                      diagnostic=failures[key]['diagnostic'])
        evidence = {'kind': 'collector_log', **failures[key]}
    else:
        # Legacy export stdout/stderr was discarded. Absent artifacts, zero
        # kernels and contiguous layouts cannot establish why profiling is absent.
        ncu = collector._empty_result(old.get('report_path'), status='unknown',
                                      reason='legacy_reason_unknown')
        evidence['report_available'] = False
    result = dict(old)
    result.update(ncu)
    # Finalize kernel time estimates without recomputing either legacy latency field.
    if old.get('latency_ms') is not None:
        result = collector._finalize_record(old['latency_ms'], result)
    for field in ('T_us', 'latency_ms'):
        if field in old:
            result[field] = old[field]
        else:
            result.pop(field, None)
        valid = isinstance(old.get(field), (int, float)) and math.isfinite(old[field])
        result['metric_validity'][field] = 'measured' if valid else 'unavailable'
    result['ncu_evidence'] = evidence
    return key, result


def validate(original, annotated):
    before, after = dict(records(original)), dict(records(annotated))
    if before.keys() != after.keys():
        raise ValueError('operator/shape/dtype keys changed')
    for key, old in before.items():
        new = after[key]
        for field in ('T_us', 'latency_ms', 'config'):
            if (field in old) != (field in new) or old.get(field) != new.get(field):
                raise ValueError(f'preserved field changed: {key} {field}')
        for field, validity in new['metric_validity'].items():
            if validity in ('unavailable', 'not_collected') and new.get(field) is not None:
                raise ValueError(f'invalid field is not null: {key} {field}')
            if validity in ('measured', 'derived', 'estimated') and new.get(field) is None:
                raise ValueError(f'available field is null: {key} {field}')
    # Includes _collection history, all per-card environments, source and scaling.
    for name, entry in original.items():
        if not isinstance(entry, dict) or 'shapes' not in entry:
            if annotated.get(name) != entry:
                raise ValueError('metadata changed: ' + name)
    json.dumps(annotated, allow_nan=False)


def migrate(path, *, workers=4, load_report=import_report, ensure_idle=assert_idle):
    path = Path(path).resolve()
    if workers < 1 or workers > 8:
        raise ValueError('workers must be between 1 and 8')
    with path.with_suffix(path.suffix + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        raw = path.read_bytes()
        original = json.loads(raw)
        if '_metric_validity_migration' in original:
            raise ValueError('already annotated; refusing repeated migration')
        ensure_idle(original)
        source_sha = digest(raw)
        backup = path.with_name(path.stem + '.pre-validity.' + source_sha[:16] + path.suffix)
        if backup.exists():
            if digest(backup.read_bytes()) != source_sha:
                raise ValueError('existing backup does not match input')
        else:
            with backup.open('xb') as handle:
                handle.write(raw)
                handle.flush()
                os.fsync(handle.fileno())
            backup.chmod(0o444)
        data = json.loads(raw)
        items = list(records(original))
        counts = Counter()
        failures = log_evidence(original)
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(lambda item: annotate_record(item, load_report, failures), items)
            for index, (key, record) in enumerate(results, 1):
                name, shape, dtype = key
                data[name]['shapes'][shape][dtype] = record
                counts[record['ncu_status']] += 1
                if index % 100 == 0:
                    print(json.dumps({'processed': index, 'total': len(items),
                                      'statuses': dict(counts)}), flush=True)
        validate(original, data)
        root = Path(__file__).resolve().parents[1]
        version = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
        metadata = {'schema_version': 1, 'completed_at_utc': datetime.now(timezone.utc).isoformat(),
                    'source_sha256': source_sha, 'backup_path': str(backup),
                    'tool_commit': version, 'tool_sha256': digest(Path(__file__).read_bytes()),
                    'parser_sha256': digest(Path(collector.__file__).read_bytes()),
                    'record_count': len(items), 'ncu_status_counts': dict(counts),
                    'method': 'offline NCU import; absent legacy evidence classified unknown'}
        data['_metric_validity_migration'] = metadata
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode='w', dir=path.parent, suffix='.validity.tmp', delete=False) as handle:
                temporary = Path(handle.name)
                json.dump(data, handle, ensure_ascii=False, allow_nan=False, indent=2)
                handle.write('\n')
                handle.flush()
                os.fsync(handle.fileno())
            validate(original, json.loads(temporary.read_text()))
            ensure_idle(original)
            if digest(path.read_bytes()) != source_sha:
                raise ValueError('source JSON changed; refusing replacement')
            os.replace(temporary, path)
            directory = os.open(path.parent, os.O_RDONLY)
            try:
                os.fsync(directory)
            finally:
                os.close(directory)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
        return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', required=True, help='Finished JSON to back up and atomically annotate in place')
    parser.add_argument('--workers', type=int, default=4, help='CPU-only offline imports (1–8)')
    args = parser.parse_args()
    print(json.dumps(migrate(args.input, workers=args.workers), indent=2), flush=True)


if __name__ == '__main__':
    main()
