"""The custom event timer must be installed before this in-place op resolves."""
import importlib
import importlib.util
from pathlib import Path
import runpy
import sys
import threading
import types

import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def entry(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'ops/pytorch'))
    monkeypatch.syspath_prepend(str(ROOT / 'tools'))
    op = importlib.import_module('ldexp_inplace_current_core')
    assert op._FRESH_TIMING_AUTH is None
    timer = types.ModuleType('triton.testing')
    stock = lambda *a, **k: None
    timer.do_bench = stock
    triton = types.ModuleType('triton')
    triton.testing = timer
    monkeypatch.setitem(sys.modules, 'triton', triton)
    monkeypatch.setitem(sys.modules, 'triton.testing', timer)
    spec = importlib.util.spec_from_file_location('ldexp_test_driver', ROOT / 'tools/collect_ldexp_inplace_current_core.py')
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    environment = types.ModuleType('collection_environment')
    environment.collect_environment = lambda *a, **k: {'control': 'CPU entry integration only'}
    monkeypatch.setitem(sys.modules, 'collection_environment', environment)
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '0')
    yield op, driver, timer, stock
    assert op._FRESH_TIMING_AUTH is None


def test_stock_preflight_and_collection_fail_before_inputs(entry, monkeypatch, tmp_path):
    op, driver, timer, stock = entry
    # An arbitrary environment declaration cannot grant timer authorization.
    monkeypatch.setenv('LDEXP_FRESH_TIMING', '1')
    monkeypatch.setattr(op, 'build_inputs', lambda *a: pytest.fail('stock must not build inputs'))
    collector = runpy.run_path(str(driver.COLLECTOR), run_name='ldexp_stock_entry_control')
    assert collector['_check_one_op'](op)['status'] == 'fail'
    output = tmp_path / 'stock.json'
    with pytest.raises(RuntimeError, match='stock timing mutates reused inputs'):
        collector['_collect_one_op_ops'](op, False, tmp_path, {}, output)
    assert not output.exists()


def _enter_driver(entry, monkeypatch, tmp_path, within):
    op, driver, timer, stock = entry
    output = tmp_path / 'bounded.json'
    monkeypatch.setattr(sys, 'argv', [str(Path(driver.__file__)), '--repo', 'pytorch', '--ops', op.OP_NAME,
                                    '--no-ncu', '--_worker', '--device', '0', '--output', str(output)])
    seen = {}

    def bounded_collector(path, run_name):
        assert path == str(driver.COLLECTOR)
        assert run_name == '__main__'
        assert op.native() is torch.ops.aten.ldexp_.default
        seen['wrapper'] = timer.do_bench
        within(op, driver, timer)
        # Bounded integration is deliberately not a formal full-grid collector.
        raise SystemExit(73)

    monkeypatch.setattr(driver.runpy, 'run_path', bounded_collector)
    with pytest.raises(SystemExit) as error:
        driver.main()
    assert error.value.code == 73
    assert timer.do_bench is stock
    assert op._FRESH_TIMING_AUTH is None
    with pytest.raises(RuntimeError):
        op.native()
    assert output.with_suffix('.fresh-samples.json').exists()
    assert not output.exists()
    return seen['wrapper']


def test_dedicated_entry_installs_timer_and_preserves_native_callable(entry, monkeypatch, tmp_path):
    def inside(op, driver, timer):
        x = torch.ones(1)
        exponent = torch.ones(1, dtype=torch.int32)
        result = op.native()(x, exponent)
        assert result is x and x.item() == 2
        # Exercise the actual installed wrapper's reviewed stock-closure route.
        namespace = {'args': (x, exponent), 'kwargs': {}, 'op': op.native()}
        factory = compile('def outer(args, kwargs, op):\n return lambda: op(*args, **kwargs)\n', str(driver.COLLECTOR), 'exec')
        exec(factory, namespace)
        closure = namespace['outer'](namespace['args'], {}, op.native())
        seen = []
        monkeypatch.setattr(driver, 'measure_fresh', lambda args, **kw: seen.append((args, kw)) or 0.25)
        # Patched helper origins are rejected by the authorization scope on
        # subsequent native resolution; timing itself receives the native API.
        assert timer.do_bench(closure) == 0.25
        assert seen[0][0] == (x, exponent)
        assert seen[0][1]['native'] is torch.ops.aten.ldexp_.default
    _enter_driver(entry, monkeypatch, tmp_path, inside)


def test_timer_replacement_and_other_thread_cannot_resolve(entry, monkeypatch, tmp_path):
    def inside(op, driver, timer):
        errors = []
        def other_thread():
            try:
                op.native()
            except RuntimeError:
                errors.append('rejected')
        thread = threading.Thread(target=other_thread)
        thread.start(); thread.join()
        assert errors == ['rejected']
        wrapper = timer.do_bench
        timer.do_bench = lambda *a: None
        with pytest.raises(RuntimeError, match='not installed'):
            op.native()
        timer.do_bench = wrapper
        assert op.native() is torch.ops.aten.ldexp_.default
    _enter_driver(entry, monkeypatch, tmp_path, inside)


def test_unknown_wrapper_cannot_authorize(entry):
    op, driver, timer, stock = entry
    timer.do_bench = lambda *a, **k: None
    with pytest.raises(RuntimeError, match='Unknown fresh timing driver'):
        with op._fresh_timing_scope(timer, timer.do_bench):
            pytest.fail('unknown wrapper authorized')


def test_exception_revokes_scope_and_restores_stock_timer(entry, monkeypatch, tmp_path):
    op, driver, timer, stock = entry
    monkeypatch.setattr(sys, 'argv', [str(Path(driver.__file__)), '--repo', 'pytorch', '--ops', op.OP_NAME,
                                    '--no-ncu', '--_worker', '--device', '0', '--output', str(tmp_path / 'failed.json')])
    def failing_collector(*a, **k):
        assert op.native() is torch.ops.aten.ldexp_.default
        with pytest.raises(RuntimeError, match='Nested'):
            with op._fresh_timing_scope(timer, timer.do_bench):
                pass
        raise RuntimeError('bounded collector failure')
    monkeypatch.setattr(driver.runpy, 'run_path', failing_collector)
    with pytest.raises(RuntimeError, match='bounded collector failure'):
        driver.main()
    assert timer.do_bench is stock and op._FRESH_TIMING_AUTH is None
    with pytest.raises(RuntimeError):
        op.native()
