# H800 snapshots for inspect-repair references

These snapshots preserve the original FlagGems core recipes used in the
inspect-repair work. Each JSON is an independent raw collector output with its
own `_environment` and `_collection`; no records were merged across runs.
The adjacent `.provenance.json` files record commands, source, controls and the
SHA-256 of each unchanged artifact.

| Snapshot | Records | Operators |
| --- | ---: | --- |
| `op_inspect216_h800_perf_baseline.json` | 94 | pack_padded_sequence (12), _flash_attention_forward (64), quantized_max_pool2d (5), quantized_max_pool2d_out (5), linalg_polar (4), linalg_polar_out (4) |
| `op_inspect216_h800_inplace_baseline.json` | 60 | addmm_ (30), baddbmm_ (30) |
| `op_inspect216_h800_sdpa_baseline.json` | 20 | scaled_dot_product_flash_attention (20, historical key) |
| `op_inspect216_h800_sdpa_kunlun_baseline.json` | 30 | _scaled_dot_product_flash_attention (original 20 plus 10 Kunlunxin-specific shared recipes) |

Actual device: NVIDIA H800, GPU 7, container `flaggems_test_tj`, Torch
2.8.0a0+5228986c39.nv25.05, Triton 3.6.0. All runs used
`NVIDIA_TF32_OVERRIDE=0`, `OMP_NUM_THREADS=2`, and `MKL_NUM_THREADS=2`.
Collection was latency-only (`--no-ncu`); these snapshots do not establish NCU
computation or bandwidth utilization. `T_us` is microseconds and
`latency_ms` is milliseconds.

The collector invocation for each operator set was:

```sh
NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  python tools/collect_baseline_nvidia.py --repo pytorch \
  --ops <snapshot-operator-list> --devices 7 --no-ncu \
  --op-timeout 600 --output <separate-snapshot.json>
```

The first snapshot was collected at `5c7dcbe47498d1be6eb032f52811ebd0080b612b`;
the inplace and original SDPA snapshots at `0eb30dbe5c7f3e1f3af31f2867fe877d259680bc`,
and the canonical SDPA supplement at `0fc13795647f36d59534bbc690b8b5a0f3739ee7`. Complete tracked
source hashes were unchanged before/after each run. Per-record `config` retains
input shapes, dtypes, strides and API parameters. Frozen catalog files also
retain original benchmark sources and case identities. Match the full recipe
and callable semantics when consuming records; shape/dtype alone is insufficient.

`linalg_polar` and its out variant are **right matrix polar decompositions using
an explicitly labeled native GPU SVD composition**. They are not a fused native
polar primitive and are unrelated to elementwise `torch.polar`. Both out writes
are timed. Independent CPU eigendecomposition, reconstruction and alias/input
checks passed for all eight records with TF32 disabled.

The five-output `_flash_attention_forward` and the nine-output
`aten._scaled_dot_product_flash_attention.default` are separate APIs. The latter
snapshot's historical operator key is `scaled_dot_product_flash_attention`.
Dedicated and shared recipes have distinct keys/configurations; do not exchange
their records. The canonical supplement also preserves Kunlunxin's different shared
core grid. Original case IDs repeat across backend catalogs; lookup must include
full input metadata and parameters, not just the original ordinal/case ID. The four
snapshots contain 204 measured records covering 184 distinct reference cases; the
20 original SDPA recipes are intentionally retained in both SDPA snapshots.
The legacy plain-name module and `_kernelgen` frozen catalogs remain unchanged;
`_inspect216_manifest.json` explicitly registers the nine added modules separately. Native output-schema checks and small independent CPU attention
controls passed. Quantized pooling was checked against independent integer
max-pooling, including quantization parameters and out aliasing. Packed sequence
outputs were checked against manual packing, including the host batch-size ABI.

`addmm_` and `baddbmm_` preserve beta=1, mutation/return alias and both matrix
layouts across all 60 original configurations. Their optional module-level
`measure_latency(op, args, kwargs)` hook uses CUDA events around only the native
call. Preparation clones every input from immutable initial snapshots, preserves
strides, flushes the cache and synchronizes **before** the start event. Library
initialization and calibration precede the measured samples. Other modules keep
the existing `triton.testing.do_bench` path. Native functional/alias checks passed
for all 60 configurations. Additional controls verified fresh beta=1 state at
every invocation and excluded deliberately slow preparation from event latency.

The inplace run originally also requested nine-output SDPA, whose control script
mistakenly treated shared shape lists as strings. That failed control is retained
in the inplace provenance; SDPA was excluded from that collector invocation.
Both inplace tasks completed with exit 0 and only their 60 records are published.
SDPA controls were corrected and its separate 20-record run passed all phases.

These artifacts supply external native references. Target-backend candidate
correctness, dummy injection, full case enumeration and actual benchmark
execution remain separate acceptance requirements.

The supplemental run also executed the complete Linux unit suite: 72 tests,
64 passed, one existing source-integration condition skipped, and seven old
process-cleanup fixture tests failed. Native controls, collector preflight and
all 30 collection records passed independently. The full failure is retained in
its provenance rather than labeled a passing suite. Comparison with the unchanged
main-branch process fixtures is tracked separately; no process framework changes
are part of this operator baseline repair.

A separate read-only comparison reproduced the same seven failures using main's
unchanged cleanup fixtures. The container injects `BASH_ENV=/etc/bash.bashrc`;
its shell startup conflicts with an existing test that temporarily isolates
`PATH`. In the original two-test order, removing only `BASH_ENV` makes both pass;
removing only `ENV` does not. No test, shell configuration or process framework
was changed.

A supplementary CPU-only full suite in the same frozen `0fc1379` source cwd,
with `BASH_ENV` and `ENV` omitted only from that child environment, completed
with **71 passed, one existing source-integration skip, zero failures** (72
tests, exit 0). All 1,168 tracked files matched the collection source before and
after. `_inspect216_validation/unit-suite-addendum.json` records the exact
command, environment, source-proof hashes and original versus supplementary
results; the two adjacent text reports retain both full logs. This follow-up
does not overwrite the original seven failures, collection environment, data or
provenance sidecar.
