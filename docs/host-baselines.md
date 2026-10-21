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
| `is_set_to` | CUDA-dispatch host storage-view predicate, with native Bool validation | separate artifact; explicit target-case selection supported |

The bundled [H800 host artifact](../op_flaggems_h800_host_baseline.json) contains
all 300 cases, collected on 2026-10-20 with an Intel Xeon Platinum 8468V CPU,
NVIDIA H800, Torch `2.8.0a0+5228986c39.nv25.05` and CUDA 12.9. Its source revisions
are FlagGems `730cfdea` and this collector `08cddb82`; full revisions, file hashes,
actual environments and per-case measurement metadata are retained in the JSON.

Collect with the existing Torch/FlagGems runtime on an NVIDIA host. The FlagGems
checkout must be clean, identify a Git commit, and include the CPU-quantized-input
accessor builders. Put the output outside that checkout to keep it clean. The collector
uses the original complete `core` catalog, not a new shape grid:

```sh
python tools/collect_baseline_host.py \
  --flaggems /path/to/FlagGems --device 0 --threads 1 \
  --output /path/to/runs/native_host.json
```

For `is_set_to`, the native module is `ops/pytorch/is_set_to.py`. Its input
contract retains storage identity, object identity, offsets, strides, layouts
and lazy flags. The collector first validates the native Bool result and times
only ATen, never a portable oracle. The source FlagGems file can instantiate
`GenericBenchmark` directly; instrumentation stays in the isolated collector
process and does not edit the source checkout.

An explicit subset may be supplied as `--case-selection selection.json`, a
mapping such as `{"is_set_to": ["<source core case ID>", ...]}`. IDs must be
unique and present in the complete source catalog. The artifact records the
full catalog IDs, selected IDs, and `explicit_core_selection` scope; a completed
task proves completion of that selection. Operators absent from the selection
keep their full core catalog. This can collect exactly the target backend's
required inputs without allocating unrelated large source-only shapes.

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
python -m pytest benchmark/test_q_per_channel_scales.py \
  -p pytest_host_baseline --host-baseline /path/to/native_host.json \
  --level core --mode operator \
  --override q_per_channel_scales:/path/to/candidate.py:run
```

`FLAGGEMS_HOST_BASELINE` can supply the path instead of `--host-baseline`.
Use the collected level and descriptors: this collector records `core` cases.
Missing cases fail explicitly rather than falling back to a different baseline.
The replay plugin acts on operator-local benchmark classes. Case listing,
preflight and profiling do not read the artifact. Imported cases must match the
actual case ID and input configuration, including quantizer metadata hashes,
sparse payload identity, or tensor CTC lengths as applicable. CPU inputs remain
CPU inputs; only accelerator roles can match across CUDA and NPU. The source
must be a completed native collection from a clean, traceable FlagGems revision.

For `is_set_to`, source case ordinals can shift when a backend filters large
shapes. Matching therefore uses the original relation parameter and actual
input contract, and preserves both source and target IDs in provenance. Missing
or ambiguous matches fail. This path requires a validated native H800 host
record and a clean collector revision; CPU oracle timings are never imported
as the NVIDIA reference. Use an operator-local benchmark class on the target
so the opt-in replay plugin can adapt only that operator.

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

### Published `is_set_to` H800 snapshot

`op_pytorch_h800_is_set_to_host_baseline.json` contains 270 native ATen host
predicate measurements collected on H800 from clean FlagGems
`3fcf2de40def60f6718aa58e3ffaebe585cf3efd` with clean collector
`3bca1900f7b04e943278a9474c2d727705762606` (PyTorch
`2.8.0a0+5228986c39.nv25.05`, Python 3.12.3). The native Bool result was checked
before timing every case. The timing thread count is one, untimed input preparation
uses four threads, and `blocked_autorange` has a 0.05-second minimum per case.

The source core catalog has 559 cases. This snapshot deliberately records
`explicit_core_selection`: its 270 selected source cases match the complete
Kunlunxin core catalog by dtype, both operand shapes, and original relation.
The reader also checks actual strides, storage identities and geometry, dtype,
lazy flags, and device roles; source and target ordinal IDs differ. This is not
a full 559-case NVIDIA collection. It preserves the original FlagGems grids and
does not reinterpret absent records as zero latency.

Collection command (paths identify isolated checkouts, not required install paths):

```sh
python3 tools/collect_baseline_host.py --flaggems /path/to/clean/FlagGems \
  --operators is_set_to --device 7 --threads 1 --input-threads 4 \
  --min-run-time 0.05 --case-selection selection.json \
  --output /path/outside/checkouts/is_set_to_host.json
```

`selection.json` is the artifact's `_selection` object. The source catalog and
selected IDs are retained in `_tasks.is_set_to`. This API reads host metadata;
these values are host dispatch latencies, not GPU kernel/NCU measurements or
correctness golden outputs. Existing workloads were left running on the H800
host; these are one-run host observations, not isolated peak-performance claims.
