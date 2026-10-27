#!/usr/bin/env python3
"""Only ldexp_inplace_current_core: official collector plus explicit fresh timing.
Every sample refreshes and synchronizes BEFORE its start CUDA event, invokes the
native in-place API once, records the end event and checks ABI/state after timing.
This is not stock triton.testing.do_bench. No NCU path is supported.
"""
from pathlib import Path
import dis
import hashlib
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
    import torch
    import triton.testing
    native = torch.ops.aten.ldexp_.default

    def fresh_do_bench(fn, warmup=25, rep=100, **kwargs):
        if kwargs:
            raise ValueError('Unknown do_bench arguments')
        args = resolve_stock_call(fn, native)
        return measure_fresh(args, warmup=warmup, rep=rep, torch=torch, native=native)

    triton.testing.do_bench = fresh_do_bench
    original_argv = list(sys.argv)
    sys.path.insert(0, str(COLLECTOR.parent))
    sys.argv = [str(COLLECTOR), *argv]
    code = 1
    try:
        runpy.run_path(str(COLLECTOR), run_name='__main__')
    except SystemExit as error:
        code = error.code or 0
    finally:
        output.parent.mkdir(parents=True, exist_ok=True)
        report = dict(method=METHOD, driver=str(Path(__file__).resolve()),
                      driver_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                      actual_entrypoint=original_argv, official_collector=str(COLLECTOR),
                      collector_sha256=hashlib.sha256(COLLECTOR.read_bytes()).hexdigest(),
                      env={k:os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES']},
                      official_exit=code, cases=CASE_REPORTS,
                      native_api_calls=sum(x['native_api_calls'] for x in CASE_REPORTS),
                      limits=['Untimed clone/equality/sync/ABI checks may dominate walltime.',
                              'Budgets accumulate native CUDA-event milliseconds only.',
                              'No NCU measurement; no stock do_bench claim.'])
        output.with_suffix('.fresh-samples.json').write_text(json.dumps(report,indent=2)+'\n')
    if code:
        raise SystemExit(code)


if __name__ == '__main__':
    main()
