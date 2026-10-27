# Copyright 2026 FlagOS Contributors
# SPDX-License-Identifier: Apache-2.0
"""Original ac9 ldexp_ core; collect only with the dedicated fresh-input driver."""
from contextlib import contextmanager
import importlib
import os
from pathlib import Path
import threading

import torch
OP_NAME = 'ldexp_inplace_current_core'
IS_INPLACE = True
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
SHAPES = [(1073741824,), (64,64), (4096,4096), (64,512,512), (1024,1024,1024)]
SOURCE_COMMIT = 'ac9a1f703757537a3d54cc1ac1febeca410316c0'
_FRESH_TIMING_AUTH = None


def _validate_fresh_timer(timer, wrapper):
    driver = Path(__file__).resolve().parents[2] / 'tools/collect_ldexp_inplace_current_core.py'
    if timer is not importlib.import_module('triton.testing') or timer.do_bench is not wrapper:
        raise RuntimeError('Dedicated fresh timing wrapper is not installed')
    if not callable(wrapper) or not hasattr(wrapper, '__code__') or Path(wrapper.__code__.co_filename).resolve() != driver:
        raise RuntimeError('Unknown fresh timing driver')
    closure = dict(zip(wrapper.__code__.co_freevars,
                       [c.cell_contents for c in wrapper.__closure__ or ()]))
    if closure.get('native') is not torch.ops.aten.ldexp_.default:
        raise RuntimeError('Fresh timing wrapper is not bound to native ldexp_')
    for name in ['resolve_stock_call', 'measure_fresh']:
        helper = wrapper.__globals__.get(name)
        if not callable(helper) or not hasattr(helper, '__code__') or Path(helper.__code__.co_filename).resolve() != driver:
            raise RuntimeError('Unknown dedicated fresh timing helper')


@contextmanager
def _fresh_timing_scope(timer, wrapper):
    """Process-local driver scope; environment flags cannot authorize collection."""
    global _FRESH_TIMING_AUTH
    _validate_fresh_timer(timer, wrapper)
    if _FRESH_TIMING_AUTH is not None:
        raise RuntimeError('Nested fresh timing authorization is forbidden')
    _FRESH_TIMING_AUTH = (os.getpid(), threading.get_ident(), timer, wrapper)
    try:
        yield
    finally:
        _FRESH_TIMING_AUTH = None


def native():
    auth = _FRESH_TIMING_AUTH
    if auth is None or auth[:2] != (os.getpid(), threading.get_ident()):
        raise RuntimeError('Use tools/collect_ldexp_inplace_current_core.py; stock timing mutates reused inputs')
    _validate_fresh_timer(auth[2], auth[3])
    return torch.ops.aten.ldexp_.default
def grid():
    return [dict(shape=list(s),variant=0) for s in SHAPES]
def key_shape(binding):
    return binding['shape']
def build_inputs(binding,dtype,device):
    shape=tuple(binding['shape']);torch.manual_seed(0)
    return (torch.randn(shape,dtype=dtype,device=device),torch.randint(-8,9,shape,dtype=torch.int32,device=device)),{}
def config(binding,dtype):
    shape=binding['shape'];stride=[];v=1
    for n in reversed(shape):stride.insert(0,v);v*=n
    tensor=lambda d:dict(shape=shape,dtype=str(d),stride=stride,requires_grad=False)
    return dict(source_commit=SOURCE_COMMIT,benchmark='benchmark/test_ldexp_.py',scope='core',variant=0,descriptor=shape,native_api='torch.ops.aten.ldexp_.default',inputs=dict(args=[tensor(dtype),tensor(torch.int32)],kwargs={}),input_recipe=dict(rng='manual_seed(0)',self='randn',exponent='randint(-8,9,int32)',fresh_inputs=True),measurement_method='official collector + operator-specific untimed-refresh CUDA-event timing',mutation='self only',output_alias='self')
