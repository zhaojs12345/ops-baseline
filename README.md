# vllm-baseline

**Language:** English · [中文](README.zh.md)

Collects performance baselines for NVIDIA native operators (vLLM / CUDA kernels), used for cross-platform acceptance of domestic backends. Field conventions follow `算子后端缺失性能基准方案参考.md`.

## Layout

| Path | Description |
|---|---|
| `tools/collect_baseline_nvidia.py` | Collector: measures latency + captures FLOPs/bytes/utilization via Nsight Compute (ncu) |
| `tools/baseline_shape.yaml` | Shape config for declarative ops (simple, positional args only) |
| `ops/<repo>/` | Custom op modules grouped by source repo (`ops/vllm`, `ops/sglang`, …), one file per op. Selected with `--repo` (default `vllm`) |
| `tools/hardware_specs.py` | Hardware peak FLOPS / bandwidth specs |

## Run

```bash
# Full collection (latency + NCU), writes JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# Latency only, skip NCU
python tools/collect_baseline_nvidia.py --no-ncu

# Custom ncu report dir (.ncu-rep opens in ncu-ui)
python tools/collect_baseline_nvidia.py --report-dir ncu_reports

# Select the op source repo: scans ops/<repo>/ (default vllm)
python tools/collect_baseline_nvidia.py --repo vllm
python tools/collect_baseline_nvidia.py --repo sglang
python tools/collect_baseline_nvidia.py --repo pytorch --ops mm,layer_norm

# Only run specific ops (comma-separated, or a file of op names)
python tools/collect_baseline_nvidia.py --ops moe_sum,add_rms_norm

# Dry-run check: can each op's native() be called? (no collection, no JSON)
python tools/collect_baseline_nvidia.py --check --repo vllm
```

Requires: CUDA GPU, `vllm`, `triton`, `ncu` (Nsight Compute). The collector scans `ops/<repo>/` (chosen by `--repo`, default `vllm`), falling back to `baseline_shape.yaml`; on name clashes, the op module wins.

See [ops/pytorch/README.md](ops/pytorch/README.md) for the pinned FlagGems kernelgen-dev PyTorch baseline catalog, source/shape conventions, validation, and deferred operators.

## Adding an operator

Create a module under `ops/<repo>/` (e.g. `ops/vllm/`, or `ops/sglang/` for sglang ops) exporting the contract fields (`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`, plus optional `config()` for complex ops). See `ops/vllm/__init__.py` for the full contract, and `ops/vllm/fused_add_rms_norm.py` / `ops/vllm/grouped_topk.py` as references. Collect it with the matching `--repo`.

## Output format

`{op: {native_api, shapes: {shape: {dtype: record}}}}`. Each record contains (aligned with the reference doc's symbols):

| Field | Meaning | Unit |
|---|---|---|
| `T_us` | native-call latency T (independent of NCU) | us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core actual compute | FLOP |
| `B_mem` | Global Memory actual traffic | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | per-unit hardware utilization | % (0–100) |
| `U_bottle_neck` / `bottle_neck_unit` | bottleneck utilization / unit | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | kernels per call / per-kernel breakdown | — |
| `config` | real input/output shapes (complex ops) | — |

Plus redundant legacy-named fields (`flops_*`, `util_*`, `bottleneck`, …) with identical values. Missing NCU values are now `null` in both the canonical and legacy fields; numeric zero is reserved for counters that actually measured zero. Consumers must handle nullable metrics.

Each record includes `ncu_status`, `ncu_reason`, `ncu_diagnostic`, `ncu_scope`
and `metric_validity` (field name → validity):

| `ncu_status` | Meaning |
| --- | --- |
| `collected` | All requested counters were read for every profiled kernel. |
| `partial` | Kernels were profiled, but some requested counters are absent, non-numeric or non-finite. Affected aggregate fields are `null`, not partial sums. |
| `no_kernel` | NCU exited successfully and explicitly reported “No kernels were profiled” in the selected NVTX scope. This does not prove the whole process launched no kernels, nor distinguish memcpy, a view/no-op and a filter mismatch. |
| `failed` | Export/import failed, timed out, or produced no usable report/rows without explicit no-kernel evidence. |
| `disabled` | Collection used `--no-ncu`. |
| `unknown` | Offline migration lacks enough historical evidence to determine the original NCU outcome. |

`metric_validity` values are `measured`, `derived` (bottleneck utilization),
`estimated` (per-kernel `T_us` allocated from end-to-end latency), `not_collected`
(no Tensor Core FLOP counter selected for this dtype), or `unavailable`.
`T_us` / `latency_ms` remain independently measured even if NCU is unavailable.
Per-kernel entries carry their own validity map; time allocation is unavailable
when multiple kernels lack usable duration weights. An unknown dtype's Tensor
Core FLOPs are `null`, rather than an assumed zero.

For example, compare latency when `record['metric_validity']['T_us'] == 'measured'`;
only include `U_mem` in utilization statistics when its validity is `measured`.
Never use `record.get('U_mem') or 0` to fill missing values. Validate per record,
not by discarding all records of an operator marked `profiling_incomplete`.
No-kernel and partial/failed records still produce `profiling_incomplete` at task
level under the conservative completeness policy; that status does not invalidate
their independently measured latency or other measured fields.

Older JSON files have no validity fields and may contain placeholder zeros.
They are not automatically migrated, and a missing report alone must not be
retroactively classified as `no_kernel`. Updating this code does not modify a
running collection or its output.


The top-level `_environment` object records the actual GPU name, UUID, memory in bytes, compute capability, SM count, logical CUDA device index and driver; CPU information; OS, Python, Torch/Triton/vLLM/SGLang, Torch build CUDA, nvcc and NCU versions; UTC collection time; collector Git commit, branch and dirty status; and effective run settings and selected operators. Only `CUDA_VISIBLE_DEVICES` and `CUDA_DEVICE_ORDER` environment variables are captured. Unavailable fields are `null`; probe failures do not interrupt collection.

Metadata is saved before the first operator and preserved by incremental writes. Internal workers do not repeat probes. `_scaling_factors.reference_chip` is the configured scaling reference; the actual device is under `_environment.hardware.gpu`. Consumers should identify operator entries by their `shapes` object; operator names such as `_reshape_alias_copy` also start with `_`. CPU fields include model, architecture, sockets, physical cores, logical CPUs, threads per core, NUMA nodes, process CPU affinity count, and hypervisor vendor. Topology reflects the OS-visible system, not necessarily physical host topology in a VM; affinity count does not represent cgroup CPU quota. Missing lscpu fields are null. Driver lookup normalizes bare Torch UUIDs and falls back only to a consistent host-wide driver version. Triton uses the imported runtime version first, including pytorch-triton/FlagTree installations.

## Environment & configuration (full H800 collection)

To run as many operators as possible on H800, beyond the base dependencies (CUDA GPU, `vllm`, `triton`, `ncu`) you also need the following; otherwise the corresponding operators are skipped or NCU captures nothing:

| Dependency / config | Affected operators | Consequence if missing | How to configure |
|---|---|---|---|
| **`nvcc` on `PATH`** (CUDA toolkit) | `fp8_einsum`, `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` | DeepGEMM JIT compile fails (`std::filesystem::exists(nvcc_path)` assert); these ops are skipped entirely | Install CUDA toolkit and add `nvcc` to `PATH` (e.g. `export PATH=/usr/local/cuda/bin:$PATH`); verify `which nvcc` |
| **Device-specific MoE / FP8 tuning json** | `fused_experts_impl`, `fused_marlin_moe_*`, `w8a8_block_fp8_matmul` | Only a `Using default MoE config` warning; falls back to default config, **collection still works** (perf may be sub-optimal) | Optional: place `vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=NVIDIA_H800.json` etc. |
| **`flash-linear-attention` (FLA)** | `chunk_gated_delta_rule_fwd`, `chunk_kda`, other FLA ops | `native()` fails to resolve; op is skipped | Install the FLA your vllm version depends on; note the import path differs by vllm version (`vllm.third_party.flash_linear_attention...` vs `vllm.model_executor.layers.fla...`) |
| **Sufficient VRAM (large-E MoE)** | large e256/e512 shapes of `fused_experts_impl`, `fused_marlin_moe_*` | The collector now frees main-process inputs before NCU, so a single weight copy (~22GB) fits; smaller VRAM may still OOM | Tune `--op-timeout`; blacklist the largest shapes with `--blacklist` if needed |
| **`tilelang`** | `mhc_pre`, `mhc_post` | The `HAS_TILELANG_MHC` gate in `vllm.model_executor.layers.mhc` is False; `native()` returns None and the op is skipped | Install `tilelang` (vLLM's MHC kernel dependency); otherwise an expected skip |
| **Complete model subtrees (DeepSeek-V4 / qwen4_exp)** | `qsa_pre_indexer`, `qsa_select_paged_decode`, `qsa_select_paged_prefill` (`vllm.models.qwen4_exp.*`), `compressor` | The corresponding vllm model subtree is absent or requires engine context; `native()` cannot resolve | qsa ops need a vllm build with the `qwen4_exp` subtree; `compressor` is an nn.Module needing an engine `VllmConfig` and cannot be collected standalone |
| **Ascend / Hygon vendor-specific ops** | `group_list_cumsum`, `indexer_epilogue`, `lightning_indexer`, `kv_rmsnorm_rope_cache`, `sparse_attn_sharedkv`, `int8_einsum`, `fused_inv_rope_int8_quant`, etc. | No NV-native kernel exists on H800; `native()` honestly returns None and the op is skipped | Cannot be collected on H800 — expected skip (their acceptance data comes from Ascend/Hygon) |

Note: skipped operators do not affect the rest (each op runs in an isolated subprocess). `fused_marlin_moe_w4a16_int4` quantizes per-expert in pure Python (slow); it keeps the full 32-shape grid matching the benchmark, so large-E architectures may approach `--op-timeout` (default 1800s) — raise it or `--blacklist` the largest shapes if a subprocess times out. The FLA ops (`chunk_gated_delta_rule_fwd`, `chunk_kda`) and `fused_q_kv_rmsnorm` resolve `native()` via multi-candidate fallback, compatible with different vllm layouts (`vllm.model_executor.layers.fla` / `vllm.third_party.flash_linear_attention`).

## Multi-GPU collection

```bash
python tools/collect_baseline_nvidia.py --repo pytorch \
  --devices 0,1,2,3,4,5,6,7 --output pytorch_h800.json
# Repeat the same command with --resume to skip completed operators.
```

`--devices` and `--device` are mutually exclusive. Each selected visibility token
(index or UUID) gets one consumer of a shared operator queue. Inherited
`CUDA_VISIBLE_DEVICES` restricts the accepted tokens; indices are not remapped.
Tasks retain process isolation; cancellation/timeout kills their process group,
including NCU descendants. Each operator has separate logs and NCU reports.

The final deliverable is **one JSON**, atomically updated by the coordinator.
Operator records and `_scaling_factors` retain their format. `_environment` is
one representative task environment; `_environments` records environments per
GPU and `_collection.tasks` records assignments and outcomes. Internal partial
files, logs and reports live in `<output>.work/`; no manual merging is needed.
NCU records without kernels are marked `profiling_incomplete`, including host or
view operations that may legitimately launch no kernel. Inspect logs rather
than interpreting absent metrics as measured zeros. Non-completed tasks produce
exit code 1 without discarding collected data. Resume requires matching source,
operator selection, devices and settings. Use idle GPUs and verify profiling
concurrency on the installed driver/NCU version before a large run.

Retries use separate attempt directories, preserving previous logs, partial JSON
and NCU reports. A failed retry cannot remove or downgrade the retained operator
snapshot; the latest task includes `retained_result` provenance when older data
is used, and `previous_attempts` records earlier outcomes. Snapshots from different
attempts are not merged workload-by-workload. A new snapshot replaces the old one
only when it retains every existing workload and its profiler evidence.
Resume clears terminal flags before work begins and records `attempt_started_at_utc`.
Coordinator/checkpoint failures stop consumers before waiting for process cleanup.
Custom `--report-dir` paths include a digest of the full output path and an attempt
ID, so identically named outputs in separate directories do not overwrite reports.

## Offline annotation of finished legacy JSON

```bash
CUDA_VISIBLE_DEVICES= python tools/annotate_baseline_validity.py \
  --input /absolute/path/to/finished.json --workers 4
```

Run from a separate checkout after all collection writers have exited. The tool
requires terminal states for every selected operator, checks live collector/NCU
processes, and holds the existing output lock. It creates a read-only, hash-named
original backup before processing. It only imports existing NCU reports on CPU;
it never exports reports or executes a GPU workload. Explicit per-record legacy
NCU failure diagnostics yield `failed`; absent reports without such evidence
yield `unknown` / `legacy_reason_unknown`, not an inferred `no_kernel`.

Before atomic replacement it validates unchanged operator/shape/dtype keys,
latencies, input configurations and collection/environment metadata, strict JSON,
field validity, idle processes, and an unchanged source SHA256. Original task
statuses remain historical. `_metric_validity_migration` records the tool/parser
version and hashes, input SHA256, backup path, timestamp and status counts; each
record records its report hash or legacy evidence. A second migration is refused.

## Supplemental collection

`--ncu-timeout` controls each NCU export independently of the whole-operator
`--op-timeout`; defaults remain 600 s and 1800 s, respectively. The NCU timeout
is forwarded to isolated workers and recorded in resume settings. For example,
use `--op-timeout 14400 --ncu-timeout 1800` for a larger targeted retry.

`tools/collect_baseline_supplement.py --base BASE --supplement RETRY --output FINAL
--ops CSV --devices 0,1,2,3,4,5,6,7` waits for BASE's existing offline annotation to finish, then collects the
selected operators separately with these larger defaults. This ordering respects
the annotation's conservative live-collector check. It does not restart that annotation or overwrite either source.
It writes a pipeline status beside FINAL and automatically merges into one new
JSON. `tools/merge_baseline_supplement.py` is also available for an explicit merge.
Complete base records remain unchanged; a retry may add missing keys or upgrade
incomplete records only without losing existing measured metric coverage. Input
configuration mismatches abort merging. New records carry retry device/environment
provenance. Original task history remains intact; `_supplemental_merge` records
source hashes, the retry run, effective task statuses/counts and final NCU status
counts. A failed retry can still contribute partial results, but remains failed.
