"""Original inputs for latency and NCU, with legacy recipes unchanged."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


def collector(monkeypatch):
    sys.path.insert(0, str(ROOT / "tools"))
    monkeypatch.setitem(sys.modules, "triton", SimpleNamespace())
    return importlib.import_module("collect_baseline_nvidia")


@pytest.mark.parametrize("enabled", [False, True])
def test_ncu_reset_and_cache_are_outside_nvtx(enabled, monkeypatch):
    module = collector(monkeypatch)
    events, seen, builds = [], [], []
    state = [3]

    def build(*args):
        builds.append(1)
        state[0] = 3
        return (), {"state": state}

    def op(**kwargs):
        events.append("op")
        seen.append(state[0])
        state[0] = 0

    class Reset:
        def make_cache(self):
            return object()

        def prepare(self, cache):
            events.append("reset/cache")
            state[0] = 3

    recipe = SimpleNamespace(native=lambda: op, build_inputs=build)
    if enabled:
        recipe.make_input_reset = lambda **kwargs: Reset()
    monkeypatch.setitem(sys.modules, "_reset_recipe", recipe)
    fake_torch = SimpleNamespace(
        float32="float32",
        cuda=SimpleNamespace(
            synchronize=lambda: events.append("sync"),
            nvtx=SimpleNamespace(
                range_push=lambda name: events.append("start"),
                range_pop=lambda: events.append("end"),
            ),
        ),
    )
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    script = module._build_profile_script_ops(
        "/unused", "_reset_recipe", {}, "torch.float32", 3
    )
    exec(compile(script, "<actual NCU script>", "exec"), {})
    assert seen == ([3] * 4 if enabled else [3, 0, 0, 3])
    assert len(builds) == (1 if enabled else 2)
    assert events[events.index("start") + 1 : events.index("end")] == ["op"]
    assert events.count("reset/cache") == (4 if enabled else 0)


def test_latency_uses_opt_in_shared_timer_and_persists_contract(monkeypatch):
    module = collector(monkeypatch)
    monkeypatch.setattr(
        module, "torch", SimpleNamespace(cuda=SimpleNamespace(synchronize=lambda: None))
    )
    monkeypatch.setattr(
        module,
        "triton",
        SimpleNamespace(
            testing=SimpleNamespace(
                do_bench=lambda *a, **kw: pytest.fail(
                    "legacy timer used for reset recipe"
                )
            )
        ),
    )
    calls = []
    state = [3]

    def op(**kw):
        calls.append(state[0])
        state[0] = 0

    reset = object()

    def build(*args):
        state[0] = 3
        return (), {}

    def capture(**kwargs):
        assert state[0] == 3
        return reset

    def timer(fn, plan):
        assert plan is reset
        for _ in range(3):
            state[0] = 3
            fn()
        return 2.5

    recipe = SimpleNamespace(
        OP_NAME="reset_fixture",
        __name__="reset_fixture",
        native=lambda: op,
        DTYPES=["dtype"],
        grid=lambda: [{}],
        build_inputs=build,
        key_shape=lambda b: "shape",
        make_input_reset=capture,
        time_with_reset=timer,
        timing_contract=lambda: {"schema": "flaggems.reset-excluded/v1"},
    )
    result = module._collect_one_op_ops(recipe, False, None)
    record = result["shapes"]["shape"]["dtype"]
    assert calls == [3, 3, 3, 3]  # fresh preflight plus three timed calls
    assert record["latency_ms"] == 2.5
    assert record["timing"] == {"schema": "flaggems.reset-excluded/v1"}
