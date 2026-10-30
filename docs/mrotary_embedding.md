# Qwen3.6 mrotary_embedding on H800

The recipe `ops/sglang/mrotary_embedding.py` collects the **native SGLang
v0.5.11 Triton** implementation retained in FlagGems-sglang. It does not import a
potentially different installed SGLang version, and never times Torch fallback.

Set `FLAGGEMS_SGLANG_SOURCE` to a checkout containing the portable-reference and
native-collection feature. Workloads and the exact matching key are read from
that checkout, rather than maintained as another independent shape list:

```sh
export FLAGGEMS_SGLANG_SOURCE=/absolute/path/FlagGems-sglang
python tools/collect_baseline_nvidia.py --repo sglang --ops mrotary_embedding \
  --device 0 --reference-chip H800 --report-dir reports/mrotary \
  --output op_sglang_mrotary_h800.json
```

The six BF16 workloads cover 27B/35B prefill/decode and 35B graph-strided positions.
The additional axis-routing correctness case is not an extra performance case.
Each record contains full tensor geometry and scalar ABI, plus source file hashes.
The key hashes the complete geometry/ABI so that positions `[512,1]` cannot match
contiguous positions by accident, while keeping NCU report filenames short.

Retain actual GPU/software/source metadata and NCU reports. The resource-based
normalization requires measured bottleneck information and a corresponding
hardware factor; a latency-only run does not establish that factor. Do not
relabel old `mrope_fused` records: their shape/rotary mode and source differ.

The FlagGems-sglang benchmark consumes this JSON using `--base-data`, requires an
exact key, and measures the candidate on its local device. Its `vs Base` metric
is `(H800 latency / candidate latency) / target-to-H800 capability factor`.
This modeled comparison does not claim a measured native implementation on the
target chip. Both native collection and the shared benchmark timer repeat the
in-place callable on preallocated tensors; input/cache construction is excluded.

## Collected snapshot

`op_sglang_mrotary_h800_perf_baseline.json` contains the six actual H800 records
collected on 2026-10-29, including NCU per-kernel records, full input configuration,
source hashes and `_environment`. The first attempt's overlong report filenames
failed profiling; this snapshot is exclusively the final median-statistic attempt
using hashed v2 keys. No failed first-attempt records were merged into it.

Measured latencies span about 5.66–6.26 microseconds. NCU selects memory bandwidth
as the largest observed resource utilization, but utilization is only about
0.1–0.2% for these tiny cases. Consequently, peak-bandwidth scaling has limited
predictive value for this workload; retain the measured times and distinguish
raw ratios from the modeled `vs Base` comparison.

Six raw `.ncu-rep` reports are retained in the original run directory
`/data/tj/workspace/sglang-mrotary-median-20261029/ncu` on H800 and in the local
validation archive `ncu-median.tar.gz` (SHA256
`083e5f24df41e87cc2269cbbc5cd0fd718bf77dc1b92d5cbb1b0112dd6318fe7`).
The native function ASTs were verified unchanged from FlagGems-sglang PR #247.

This recipe sets `BENCH_RETURN_MODE="median"` to match the SGLang kernel timer.
The collector forwards this option only for recipes that declare it; existing
recipes retain their original default statistic. Timing metadata is recorded
with these six records. Earlier mean-statistic trials are retained as evidence,
but are not the published snapshot. The `mrotary-v2-` key version pins this
median comparison contract and cannot match the earlier trial keys.
