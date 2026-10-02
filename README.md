# vllm-baseline

**Language:** English · [中文](README.zh.md)

Collects performance baselines for NVIDIA native operators (vLLM / CUDA kernels), used for cross-platform acceptance of domestic backends. Field conventions follow `算子后端缺失性能基准方案参考.md`.

## Layout

| Path | Description |
|---|---|
| `tools/collect_baseline_nvidia.py` | Collector: measures latency + captures FLOPs/bytes/utilization via Nsight Compute (ncu) |
| `tools/baseline_shape.yaml` | Shape config for declarative ops (simple, positional args only) |
| `ops/` | Custom op modules (complex ops: multi-tensor, constraint tensors, quantization…), one file per op |
| `tools/hardware_specs.py` | Hardware peak FLOPS / bandwidth specs |

## Run

```bash
# Full collection (latency + NCU), writes JSON
python tools/collect_baseline_nvidia.py --output op_perf_baseline.json

# Latency only, skip NCU
python tools/collect_baseline_nvidia.py --no-ncu

# Custom ncu report dir (.ncu-rep opens in ncu-ui)
python tools/collect_baseline_nvidia.py --report-dir ncu_reports
```

Requires: CUDA GPU, `vllm`, `triton`, `ncu` (Nsight Compute). The collector scans `ops/` first, falling back to `baseline_shape.yaml`; on name clashes, `ops/` wins.

## Adding an operator

Create a module under `ops/` exporting the contract fields (`OP_NAME` / `DTYPES` / `IS_INPLACE` / `native()` / `grid()` / `build_inputs()` / `key_shape()`, plus optional `config()` for complex ops). See `ops/__init__.py` for the full contract, and `ops/fused_add_rms_norm.py` / `ops/grouped_topk.py` as references.

## Output format

`{op: {native_api, shapes: {shape: {dtype: record}}}}`. Each record contains (aligned with the reference doc's symbols):

| Field | Meaning | Unit |
|---|---|---|
| `T_us` | kernel runtime T | us |
| `F_cuda` / `F_tensor` | CUDA / Tensor Core actual compute | FLOP |
| `B_mem` | Global Memory actual traffic | Byte |
| `U_cuda` / `U_tensor` / `U_mem` | per-unit hardware utilization | % (0–100) |
| `U_bottle_neck` / `bottle_neck_unit` | bottleneck utilization / unit | % / `cuda`\|`tensor`\|`mem` |
| `num_kernels` / `per_kernel` | kernels per call / per-kernel breakdown | — |
| `config` | real input/output shapes (complex ops) | — |

Plus a set of redundant legacy-compatible fields (`flops_*`, `util_*`, `bottleneck`, …) with identical values.

## Environment & configuration (full H800 collection)

To run as many operators as possible on H800, beyond the base dependencies (CUDA GPU, `vllm`, `triton`, `ncu`) you also need the following; otherwise the corresponding operators are skipped or NCU captures nothing:

| Dependency / config | Affected operators | Consequence if missing | How to configure |
|---|---|---|---|
| **`nvcc` on `PATH`** (CUDA toolkit) | `fp8_einsum`, `fp8_fp4_mqa_logits`, `fp8_fp4_paged_mqa_logits` | DeepGEMM JIT compile fails (`std::filesystem::exists(nvcc_path)` assert); these ops are skipped entirely | Install CUDA toolkit and add `nvcc` to `PATH` (e.g. `export PATH=/usr/local/cuda/bin:$PATH`); verify `which nvcc` |
| **Device-specific MoE / FP8 tuning json** | `fused_experts_impl`, `fused_marlin_moe_*`, `w8a8_block_fp8_matmul` | Only a `Using default MoE config` warning; falls back to default config, **collection still works** (perf may be sub-optimal) | Optional: place `vllm/model_executor/layers/fused_moe/configs/E=*,N=*,device_name=NVIDIA_H800.json` etc. |
| **`flash-linear-attention` (FLA)** | `chunk_gated_delta_rule_fwd`, `chunk_kda`, other FLA ops | `native()` fails to resolve; op is skipped | Install the FLA your vllm version depends on; note the import path differs by vllm version (`vllm.third_party.flash_linear_attention...` vs `vllm.model_executor.layers.fla...`) |
| **Sufficient VRAM (large-E MoE)** | large e256/e512 shapes of `fused_experts_impl`, `fused_marlin_moe_*` | The collector now frees main-process inputs before NCU, so a single weight copy (~22GB) fits; smaller VRAM may still OOM | Tune `--op-timeout`; blacklist the largest shapes with `--blacklist` if needed |
| **Ascend / Hygon vendor-specific ops** | `group_list_cumsum`, `indexer_epilogue`, `lightning_indexer`, `kv_rmsnorm_rope_cache`, `sparse_attn_sharedkv`, `int8_einsum`, `fused_inv_rope_int8_quant`, etc. | No NV-native kernel exists on H800; `native()` honestly returns None and the op is skipped | Cannot be collected on H800 — expected skip (their acceptance data comes from Ascend/Hygon) |

Note: skipped operators do not affect the rest (each op runs in an isolated subprocess). `fused_marlin_moe_w4a16_int4` quantizes per-expert in pure Python (slow), so its large-E shapes were trimmed to the two end token counts to avoid subprocess timeout.
