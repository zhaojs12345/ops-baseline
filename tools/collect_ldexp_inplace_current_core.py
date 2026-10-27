#!/usr/bin/env python3
"""Only ldexp_inplace_current_core: official collector plus explicit fresh timing.
Every sample refreshes and synchronizes BEFORE its start CUDA event, invokes the
native in-place API once, records the end event and checks ABI/state after timing.
This is not stock triton.testing.do_bench. No NCU path is supported.
"""
from pathlib import Path
import dis
import hashlib
import importlib
import json
import math
import os
import runpy
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
COLLECTOR = ROOT / 'tools/collect_baseline_nvidia.py'
OP_NAME = 'ldexp_inplace_current_core'
METHOD = 'official collector + operator-specific untimed-refresh CUDA-event timing'
SAMPLES = []
CASE_REPORTS = []


def _closure_factory(args, kwargs, op):
    return lambda: op(*args, **kwargs)


def _instructions(fn):
    return [(x.opname, x.argval) for x in dis.get_instructions(fn)
            if x.opname not in ('COPY_FREE_VARS', 'RESUME', 'CACHE')]


def resolve_stock_call(fn, native):
    """Fail closed: only the reviewed single-native-call stock closure."""
    if Path(fn.__code__.co_filename).resolve() != COLLECTOR:
        raise ValueError('Unknown collector callable origin')
    if fn.__code__.co_freevars != ('args', 'kwargs', 'op'):
        raise ValueError('Unknown collector closure variables')
    if _instructions(fn) != _instructions(_closure_factory(None, None, None)):
        raise ValueError('Collector closure does more than one native invocation')
    values = dict(zip(fn.__code__.co_freevars,
                      [c.cell_contents for c in fn.__closure__]))
    args, kwargs, op = values['args'], values['kwargs'], values['op']
    if op is not native or kwargs or not isinstance(args, (tuple, list)) or len(args) != 2:
        raise ValueError('Expected exact native ldexp_ positional ABI')
    return args


def _meta(x):
    return dict(device=str(x.device), dtype=str(x.dtype), shape=list(x.shape),
                stride=list(x.stride()), requires_grad=x.requires_grad)


def measure_fresh(args, *, warmup, rep, torch, native):
    if warmup != 25 or rep != 100:
        raise ValueError('Only original warmup25/rep100 native-ms budgets supported')
    prototype, exponent = args
    if (prototype.device.type != 'cuda' or exponent.device != prototype.device
            or prototype.dtype not in [torch.float16, torch.float32, torch.bfloat16]
            or exponent.dtype != torch.int32 or exponent.shape != prototype.shape
            or not prototype.is_contiguous() or not exponent.is_contiguous()
            or prototype.requires_grad or exponent.requires_grad):
        raise ValueError('Unexpected original ldexp_ core metadata')
    snapshots = [x.detach().clone() for x in args]
    versions = [x._version for x in args]
    case = dict(index=len(CASE_REPORTS), prototype=[_meta(x) for x in args], samples=[])

    def sample(phase):
        # Setup and validation deliberately precede the start event.
        fresh = [x.detach().clone() for x in args]
        if not all(torch.equal(x, y) for x, y in zip(fresh, snapshots)):
            raise RuntimeError('Fresh sample initial values differ from prototype')
        torch.cuda.synchronize()
        before = [_meta(x) for x in fresh]
        version = fresh[0]._version
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = native(*fresh)
        end.record()
        end.synchronize()
        ms = float(start.elapsed_time(end))
        if not math.isfinite(ms) or ms <= 0:
            raise RuntimeError('Native event timer is nonpositive/nonfinite')
        if (result is not fresh[0] or result.data_ptr() != fresh[0].data_ptr()
                or [_meta(x) for x in fresh] != before
                or fresh[0]._version != version + 1
                or not torch.equal(fresh[1], snapshots[1])):
            raise RuntimeError('Native alias/input/output ABI mismatch')
        if not torch.isfinite(result).all().item():
            raise RuntimeError('Original single-call core unexpectedly saturated')
        if not all(x._version == v and torch.equal(x, y)
                   for x, y, v in zip(args, snapshots, versions)):
            raise RuntimeError('Original prototype was modified')
        row = dict(phase=phase, native_ms=ms, native_api_calls=1,
                   fresh_initial_exact=True, clone_and_sync_before_start=True,
                   output_alias_self=True, exponent_unchanged=True,
                   prototype_unchanged=True, output_finite=True)
        case['samples'].append(row)
        return ms

    elapsed = 0.0
    while elapsed < warmup:
        elapsed += sample('warmup')
    measured = []
    elapsed = 0.0
    while not measured or elapsed < rep:
        value = sample('measurement'); measured.append(value); elapsed += value
    case.update(mean_native_ms=sum(measured)/len(measured),
                warmup_native_ms=sum(x['native_ms'] for x in case['samples']
                                     if x['phase']=='warmup'),
                measured_native_ms=elapsed, warmup_budget_ms=warmup,
                measurement_budget_ms=rep, native_api_calls=len(case['samples']))
    CASE_REPORTS.append(case)
    return case['mean_native_ms']


def main():
    argv = sys.argv[1:]
    # Explicit official worker mode avoids spawning an unadapted child entrypoint.
    required = ['--repo', 'pytorch', '--ops', OP_NAME, '--no-ncu', '--_worker']
    if any(x not in argv for x in required) or '--devices' in argv or '--check' in argv:
        raise ValueError('Only explicit one-op --no-ncu official worker mode is allowed')
    if argv[argv.index('--repo')+1] != 'pytorch' or argv[argv.index('--ops')+1] != OP_NAME:
        raise ValueError('Other repository/operator is forbidden')
    if '--whitelist' in argv or '--blacklist' in argv:
        raise ValueError('No alternative operator filters')
    output = Path(argv[argv.index('--output')+1]).resolve()
    if '--device' not in argv:
        raise ValueError('Explicit physical CUDA device is required')
    device = argv[argv.index('--device')+1]
    if not device.isdigit():
        raise ValueError('Expected physical CUDA device index')
    os.environ['CUDA_VISIBLE_DEVICES'] = device
    sys.path.insert(0, str(COLLECTOR.parent))
    import torch
    import triton.testing
    native = torch.ops.aten.ldexp_.default
    from collection_environment import collect_environment
    environment = collect_environment(torch, dict(
        repo='pytorch', device=device, ncu_enabled=False, output_path=str(output),
        only=[OP_NAME], selected_ops=[OP_NAME], official_worker=True,
        actual_driver_entrypoint=list(sys.argv), latency_method=METHOD,
        latency_warmup_ms=25, latency_rep_ms=100,
        budget_unit='accumulated native CUDA-event milliseconds',
        setup_timed=False, timed_native_api_calls_per_sample=1))

    def fresh_do_bench(fn, warmup=25, rep=100, **kwargs):
        if kwargs:
            raise ValueError('Unknown do_bench arguments')
        args = resolve_stock_call(fn, native)
        return measure_fresh(args, warmup=warmup, rep=rep, torch=torch, native=native)

    original_do_bench = triton.testing.do_bench
    # Use the same top-level module name as the official ops discovery path.
    sys.path.insert(0, str(ROOT / 'ops/pytorch'))
    op_module = importlib.import_module(OP_NAME)
    original_argv = list(sys.argv)
    sys.path.insert(0, str(COLLECTOR.parent))
    sys.argv = [str(COLLECTOR), *argv]
    code = 1
    try:
        triton.testing.do_bench = fresh_do_bench
        with op_module._fresh_timing_scope(triton.testing, fresh_do_bench):
            runpy.run_path(str(COLLECTOR), run_name='__main__')
    except SystemExit as error:
        code = error.code or 0
    finally:
        triton.testing.do_bench = original_do_bench
        sys.argv = original_argv
        output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(method=METHOD, driver=str(Path(__file__).resolve()),
                      driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      actual_entrypoint=original_argv, official_collector=str(COLLECTOR),
                      collector_sha256=hashlib.sha256(COLLECTOR.read_bytes()).hexdigest(),
                      env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES']},
                      official_exit=code, cases=CASE_REPORTS,
                      native_api_calls=sum(x['native_api_calls'] for x in CASE_REPORTS),
                      call_count_scope='Timed samples only; official unmeasured preflight is separate',
                      environment_before_measurement=environment,
                      limits=['Untimed clone/equality/sync/ABI checks may dominate walltime.',
                              'Budgets accumulate native CUDA-event milliseconds only.',
                              'No NCU measurement; no stock do_bench claim.'])
        output.with_suffix('.fresh-samples.json').write_text(json.dumps(report,indent=2)+'\n')
    if code:
        raise SystemExit(code)
    # Preserve official output before adding explicit operator-specific method.
    raw = output.read_bytes()
    output.with_suffix('.official-raw.json').write_bytes(raw)
    artifact = json.loads(raw)
    if OP_NAME not in artifact or len(CASE_REPORTS) != 15:
        raise RuntimeError('Expected complete original fifteen-case native grid')
    method = dict(name=METHOD, driver_sha256=report['driver_sha256'],
                  official_collector_sha256=report['collector_sha256'],
                  actual_entrypoint=original_argv, ncu_profiled=False,
                  raw_official_sha256=hashlib.sha256(raw).hexdigest(),
                  sample_report=output.with_suffix('.fresh-samples.json').name)
    artifact['_environment'] = environment
    artifact['_measurement_method'] = method
    for per_dtype in artifact[OP_NAME]['shapes'].values():
        for dtype, record in per_dtype.items():
            shape = record['config']['descriptor']
            matches = [c for c in CASE_REPORTS
                       if c['prototype'][0]['dtype'] == dtype
                       and c['prototype'][0]['shape'] == shape]
            if len(matches) != 1:
                raise RuntimeError('No unique original fresh-sample identity')
            case = matches[0]
            if not math.isclose(record['latency_ms'], case['mean_native_ms'],
                                rel_tol=1e-12, abs_tol=1e-12):
                raise RuntimeError('Official result differs from actual native events')
            record['measurement_method'] = dict(method, case_index=case['index'],
                                                native_api_calls=case['native_api_calls'])
            for key in list(record):
                if key not in ['config','T_us','latency_ms','measurement_method']:
                    record.pop(key)
    output.write_text(json.dumps(artifact, indent=2) + '\n')


if __name__ == '__main__':
    main()
