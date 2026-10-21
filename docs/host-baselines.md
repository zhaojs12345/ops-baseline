# Native CPU and host baselines

These FlagGems operators use a separate host artifact. Do not mix its latency
with NCU kernel timings or apply GPU peak-throughput scaling to it.

| Operator | Native source measurement | Bundled core cases |
| --- | --- | ---: |
| `q_per_channel_scales` | CPU quantizer metadata, including `.out` copies | 132 |
| `q_per_channel_zero_points` | CPU quantizer metadata, including `.out` copies | 108 |
| `fbgemm_linear_quantize_weight` | x86 CPU FBGEMM quantization | 11 |
| `fbgemm_linear_fp16_weight_fp32_activation` | x86 CPU FBGEMM linear; packing occurs before timing | 9 |
| `_to_sparse_semi_structured` | CUDA-dispatch host end-to-end time, including internal transfers and completion | 20 |
| `_use_cudnn_ctc_loss` | CUDA-dispatch host capability predicate; this is not CTC computation | 20 |
| `_nested_tensor_storage_offsets` | CUDA-dispatch host metadata accessor on a strided NestedTensor | 21 |
| `_nested_tensor_strides` | CUDA-dispatch host metadata accessor on a strided NestedTensor | 21 |

The bundled [H800 host artifact](../op_flaggems_h800_host_baseline.json) contains
all 300 cases, collected on 2026-10-20 with an Intel Xeon Platinum 8468V CPU,
NVIDIA H800, Torch `2.8.0a0+5228986c39.nv25.05` and CUDA 12.9. Its source revisions
are FlagGems `730cfdea` and this collector `08cddb82`; full revisions, file hashes,
actual environments and per-case measurement metadata are retained in the JSON.

The separate [nested metadata artifact](../op_flaggems_h800_nested_metadata_host_baseline.json)
adds 42 cases: three floating dtypes across the original seven ragged batch/rank
layouts for each accessor. It was collected with clean FlagGems
`ca1f2c01731fb5230e4b3ff9e9247e5a76139763` and clean collector
`71640d14afe20cbdfa0864b2a35eae0a4f416e60`, on H800 device 7 with the same
Xeon/Torch stack, one timing thread, four preparation threads and seed 20261020.
Its SHA256 is `434e8ffca47e3acec7aeb9e4cf0f3cd0dd6f256475cff95a5bcfa138754b616f`.
The `cuda_dispatch_host_metadata` timer measures the native host dispatch and
CPU metadata result; it launches no device computation and adds no CUDA sync.
Each input identity retains every component's shape, stride and global storage
offset, dtype, layout, gradient flag and packed storage extent. Metadata values
do not depend on random payload values. No NCU or GPU-event latency is substituted.

To collect this pair explicitly, add
`--operators _nested_tensor_storage_offsets _nested_tensor_strides` to the
collector command. FlagGems target benchmarks can consume the pinned artifact
with `FLAGGEMS_NESTED_METADATA_H800_HOST_BASELINE=/path/to/artifact.json` and an
explicit candidate override; their operator-local code executes the candidate
under a real local host timer. A CPU oracle's local latency is never substituted
for the native source latency.

Collect with the existing Torch/FlagGems runtime on an NVIDIA host. The FlagGems
checkout must be clean, identify a Git commit, and include the CPU-quantized-input
accessor builders. Put the output outside that checkout to keep it clean. The collector
uses the original complete `core` catalog, not a new shape grid:

```sh
python tools/collect_baseline_host.py \
  --flaggems /path/to/FlagGems --device 0 --threads 1 \
  --output /path/to/runs/native_host.json
```

Each operator runs in its own pytest subprocess. A temporary pytest plugin
replaces only reference timing, without editing the FlagGems checkout. Inputs
and packed weights are built before timing. CPU wall-clock samples are repeated
and the median is reported in milliseconds; CPU architecture/model, Torch,
thread count, native API, input contract and source revision are retained.
Input preparation uses four CPU threads by default (`--input-threads`); the
timer separately applies `--threads` (one by default) inside every measured region.
Collection errors and missing/duplicate catalog records are failures, not zero
latency. An operator timeout terminates its process group. Use a fresh output
path for each run; this collector does not resume or overwrite an existing run.

The artifact has a distinct `_format` (`flaggems_native_host_v1`), `_source`,
`_measurement`, `_tasks`, and `operators` mapping. Failed tasks remain visible,
and partial artifacts make the collector exit nonzero. Reports and logs stay
next to the combined output in its `.work` directory. Native latency is a
performance baseline, not a correctness golden or a transport for packed handles.

For an explicit cross-machine comparison, make `tools` importable and enable the
consumer pytest plugin in the target runtime:

```sh
PYTHONPATH=/path/to/ops-baseline/tools:$PYTHONPATH \
python -m pytest benchmark/test_q_per_channel_scales.py -m q_per_channel_scales \
  -p pytest_host_baseline --host-baseline /path/to/native_host.json \
  --level core --mode operator \
  --override q_per_channel_scales:/path/to/candidate.py:run
```

`FLAGGEMS_HOST_BASELINE` can supply the path instead of `--host-baseline`.
Use the collected level and descriptors: this collector records `core` cases.
Missing cases fail explicitly rather than falling back to a different baseline.
The plugin acts only on the supported operator-local benchmark classes. Case listing,
preflight and profiling do not read the artifact. Imported cases must match the
actual case ID and input configuration, including quantizer metadata hashes,
sparse payload identity, or tensor CTC lengths as applicable. CPU inputs remain
CPU inputs; only accelerator roles can match across CUDA and NPU. The source
must be a completed native collection from a clean, traceable FlagGems revision.

Only the reference phase imports latency. The candidate always runs through a
real host wall-clock timer, even when its callable equals the reference callable.
Sparse packing additionally synchronizes the target accelerator. Reports retain
both source and target provenance and the raw source/target latency ratio. There
is no GPU normalization, GPU FLOPS/bandwidth metric, backward benchmark, or
cudagraph interpretation. Formal timing requires `--mode operator`;
`--reference-only` can import a baseline without running a candidate.

FBGEMM availability differs by architecture. In particular the FP16 packed-weight
operand is a process-local opaque native handle. It must be created by the real
native packer in the process using it; its tensor bytes cannot be copied to ARM
or replaced with raw weights. Collecting its x86 latency does not establish an
executable ARM candidate ABI. Missing target input support remains a visible
failure and is not blanket-skipped by this plugin.
