#!/usr/bin/env python3
"""Dedicated Adagrad host baseline: one CPU-native step plus target transfers.

Every warmup/sample gets a complete untimed clone of the original target state.
No CUDA kernel timing, NCU metrics, scaling, or repeated mutable Timer arguments.
"""

import argparse
import ast
import gc
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import torch
from collection_environment import collect_environment

ROOT = Path(__file__).resolve().parents[1]


def source_proof():
    head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    assert not subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT)
    files = subprocess.check_output(["git", "ls-tree", "-rz", head], cwd=ROOT).split(
        b"\0"
    )
    hashes = {}
    for entry in files:
        if not entry:
            continue
        path = entry.split(b"\t", 1)[1].decode()
        original = subprocess.check_output(["git", "show", head + ":" + path], cwd=ROOT)
        actual = (ROOT / path).read_bytes()
        assert actual == original, path
        hashes[path] = hashlib.sha256(actual).hexdigest()
    return dict(
        head=head,
        checked=len(hashes),
        git_clean=True,
        source_hashes=hashes,
        observed_unix=time.time(),
    )


def sample_idle(physical_device):
    def query(args):
        return subprocess.check_output(["nvidia-smi", *args], text=True).strip()

    return dict(
        observed_unix=time.time(),
        physical_device=physical_device,
        gpu=query(
            [
                "--id",
                str(physical_device),
                "--query-gpu=index,uuid,name,memory.used,memory.free,utilization.gpu",
                "--format=csv,noheader",
            ]
        ),
        applications=query(
            [
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv,noheader",
            ]
        ),
        scope="sampled boundary only; not continuous exclusivity",
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--flaggems", type=Path, required=True)
    parser.add_argument("--source-manifest", type=Path, required=True)
    parser.add_argument("--physical-device", type=int, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    out = args.output
    idle_before = sample_idle(args.physical_device)
    (out / "idle-before.json").write_text(json.dumps(idle_before, indent=2))
    before = source_proof()
    (out / "source-before.json").write_text(json.dumps(before, indent=2))
    manifest = json.loads(args.source_manifest.read_bytes())
    assert (
        manifest["head"] == "ac9a1f703757537a3d54cc1ac1febeca410316c0"
        and len(manifest["source_hashes"]) == 8289
    )

    def fg_proof():
        hashes = {
            path: hashlib.sha256((args.flaggems / path).read_bytes()).hexdigest()
            for path in manifest["source_hashes"]
        }
        assert hashes == manifest["source_hashes"]
        return dict(head=manifest["head"], checked=8289, source_hashes=hashes)

    (out / "flaggems-source-before.json").write_text(json.dumps(fg_proof(), indent=2))
    spec = importlib.util.spec_from_file_location(
        "adagrad_host_operator", ROOT / "ops/pytorch/fused_adagrad_host_current_core.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    torch.cuda.set_device(0)
    assert torch.cuda.get_device_name() == "NVIDIA H800"
    assert torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_fused_adagrad_", "CPU"
    )
    assert not torch._C._dispatch_has_kernel_for_dispatch_key(
        "aten::_fused_adagrad_", "CUDA"
    )
    run = dict(
        repo="pytorch",
        only=[module.OP_NAME],
        selected_ops=[module.OP_NAME],
        ncu_enabled=False,
        physical_device=args.physical_device,
        execution_kind=module.EXECUTION_KIND,
        latency_method="synchronized_host_wall_single_call_fresh_state",
        policy=module._ADAGRAD_HOST_POLICY,
    )
    environment = collect_environment(torch, run)
    assert (
        environment["source"]["git_commit"] == before["head"]
        and environment["source"]["git_dirty"] is False
    )
    original_text = (args.flaggems / "benchmark/test_fused_adagrad_.py").read_text()
    tree = ast.parse(original_text)
    fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_inputs_fn"
    )
    ns = dict(torch=torch)
    exec(
        compile(ast.Module(body=[fn], type_ignores=[]), "original-ac9-builder", "exec"),
        ns,
    )
    assert ast.dump(fn, include_attributes=False) == ast.dump(
        next(
            node
            for node in ast.parse(inspect.getsource(module)).body
            if isinstance(node, ast.FunctionDef) and node.name == "_build_inputs_fn"
        ),
        include_attributes=False,
    )
    records, controls = [], []
    for ordinal, case in enumerate(module.cases()):
        plan = SimpleNamespace(
            builder_args=(
                case["shape"],
                case["lr_decay"],
                case["weight_decay"],
                case["maximize"],
            )
        )
        original = ns["_build_inputs_fn"](plan, torch.float32, "cuda:0")
        post_original = torch.rand(17, device="cuda:0").cpu()
        built, kwargs = module.build_inputs(case, "cuda:0")
        post_module = torch.rand(17, device="cuda:0").cpu()
        assert module._adagrad_fingerprint(
            (tuple(original[:4]), original[4])
        ) == module._adagrad_fingerprint((built, kwargs))
        assert torch.equal(post_original, post_module)
        native_input = module._adagrad_tree_clone(built)
        expected = [
            [module._adagrad_cpu_copy(tensor) for tensor in group] for group in built
        ]
        torch._fused_adagrad_(*expected, **kwargs)
        readonly = module._adagrad_fingerprint((native_input[1], native_input[3]))
        module.host_native()(*native_input, **kwargs)
        for index in (0, 2):
            torch.testing.assert_close(
                native_input[index][0].cpu(), expected[index][0], rtol=1e-5, atol=1e-5
            )
        assert (
            module._adagrad_fingerprint((native_input[1], native_input[3])) == readonly
        )
        controls.append(
            dict(
                ordinal=ordinal,
                original_recipe_and_values_equal=True,
                post_gpu_rng_equal=True,
                native_cpu_values_match=True,
                original_readonly=True,
            )
        )
        del original, native_input, expected
        gc.collect()
        boundary = sample_idle(args.physical_device)
        measured = module._adagrad_host_samples(
            module.host_native(), built, kwargs, torch.cuda.synchronize
        )
        case_id = f"benchmark/test_fused_adagrad_.py::test_fused_adagrad_::core::float32::{ordinal}"
        records.append(
            dict(
                case_id=case_id,
                config=module.config(case),
                T_us=measured["latency_ms"] * 1000,
                latency_ms=measured["latency_ms"],
                host_measurement=measured,
                boundary=boundary,
            )
        )
        (out / "progress.json").write_text(
            json.dumps(dict(records=records, controls=controls), indent=2)
        )
        del built, kwargs
        gc.collect()
        torch.cuda.empty_cache()
    assert len(records) == 8
    functions = [
        "_adagrad_dense",
        "_adagrad_validate",
        "_adagrad_cpu_copy",
        "_adagrad_cpu_native",
        "_adagrad_cpu_apply",
        "_adagrad_tree_clone",
        "_adagrad_fingerprint",
        "_adagrad_host_samples",
    ]
    identity = {
        name: hashlib.sha256(
            inspect.getsource(getattr(module, name)).encode()
        ).hexdigest()
        for name in functions
    }
    after = source_proof()
    assert after["source_hashes"] == before["source_hashes"]
    (out / "source-after.json").write_text(json.dumps(after, indent=2))
    (out / "flaggems-source-after.json").write_text(json.dumps(fg_proof(), indent=2))
    report = dict(
        schema="adagrad.host-baseline/v1",
        operator="_fused_adagrad_",
        execution_kind=module.EXECUTION_KIND,
        native_api="torch._fused_adagrad_",
        native_schema=str(torch.ops.aten._fused_adagrad_.default._schema),
        dispatch=dict(CPU=True, CUDA=False),
        _environment=environment,
        _execution=dict(
            collection_source=before["head"],
            source_before_after_exact=True,
            source_files=before["checked"],
            source_helper_identity=identity,
            input_source=manifest["head"],
            units="milliseconds and microseconds; raw sample nanoseconds",
            scope="original8 F32 source recipes; CPU-native plus H800 transfers, not GPU kernel latency; no NCU/scaling",
        ),
        records=records,
    )
    (out / "preflight.json").write_text(json.dumps(dict(controls=controls), indent=2))
    (out / "baseline.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "idle-after.json").write_text(
        json.dumps(sample_idle(args.physical_device), indent=2)
    )
    (out / "DONE.json").write_text(
        json.dumps(
            dict(
                terminal=True,
                records=8,
                source_head=before["head"],
                artifact_sha256=hashlib.sha256(
                    (out / "baseline.json").read_bytes()
                ).hexdigest(),
            ),
            indent=2,
        )
    )
    print(
        "DONE8 " + hashlib.sha256((out / "baseline.json").read_bytes()).hexdigest(),
        flush=True,
    )


if __name__ == "__main__":
    main()
