# Pinned Qwen3.6 SGLang native baselines

`ops/sglang_qwen36` adds seven native workload profiles with 28 documented BF16
cases: Gemma RMSNorm (12), Gemma fused-add RMSNorm (4), optional-out SiLU (4),
TopK selection (2), the MoE kernel sequence (2), chunk GDN (2), and packed GDN
decode (2). Existing `ops/sglang` recipes remain independent.

These profiles require the companion FlagGems-sglang workload checkout, including
its pinned SGLang v0.5.11 environment (upstream commit
`612785ffdcaf35552f1ed433a981d596ca9fe900`). Follow that checkout's
`docs/qwen36_workloads.md` for offline wheel setup and validation. Missing source,
wrong package versions, changed entrypoints or unavailable FlashInfer fail rather
than selecting a Torch substitute.

```sh
export FLAGGEMS_SGLANG_SOURCE=/workspace/sglang-suite
python tools/collect_baseline_nvidia.py --repo sglang_qwen36 --device 0 \
  --reference-chip H800 --output op_sglang_qwen36_h800_perf_baseline.json \
  --report-dir qwen36-ncu
```

The functions selected by the native adapter include CUDA/FlashInfer and Triton
implementations. TopK's native configuration is prepared outside the timer.
Each profile selects median `triton.testing.do_bench` timing, matching the CUDA
benchmark path. Other target backends retain their existing timers (Ascend uses
`do_bench_npu`). Keys hash full tensor geometry (including stride and storage offset),
scalar/configuration ABI and timing mode. GDN pools retain their documented full
sizes. Configuration records contain input geometry, upstream revision and the
shared workload source SHA256. Collection `_environment` describes the actual
host/software/run; examine it alongside per-case NCU records before comparing.

Pass the resulting JSON through target FlagGems-sglang benchmark `--base-data`.
Only equal configuration keys should match; locally timed Torch reference-as-
candidate results validate the pipeline, not an optimized kernel.

## Published H800 snapshot

`op_sglang_qwen36_h800_perf_baseline.json` contains all seven interfaces and 28
core records collected on H800 with median latency and NCU counters. Each record
has a positive kernel count and retained raw `.ncu-rep` evidence. `_provenance`
binds the raw collection hash, exact remote/local source file hashes and Git
revisions, original native source specification and golden dataset manifest.
The collector's Git fields remain null for the tar-staged remote checkout;
verified source provenance is recorded separately rather than inventing Git
state on that host.

The legacy collector summary excludes underscore-prefixed keys when counting
operators, so it prints six for this snapshot. The named
`_fused_moe_kernel_sequence` entry is a real seventh operator and is retained;
consumers must use the requested operator key. The companion benchmark checks
all 28 configuration keys directly.

Golden data for these core cases was generated on H800 and uploaded to the fresh
KS3 prefix `op-verify/sglang-qwen36/20261030-native-v1`. Ascend replay of those
same inputs passed 28/28 with no skips. This validates the portable Torch oracle;
it does not establish performance of a new backend candidate.
