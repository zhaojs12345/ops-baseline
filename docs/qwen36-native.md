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
Each profile selects median `triton.testing.do_bench` timing, matching the target
benchmark. Keys hash full tensor geometry (including stride and storage offset),
scalar/configuration ABI and timing mode. GDN pools retain their documented full
sizes. Configuration records contain input geometry, upstream revision and the
shared workload source SHA256. Collection `_environment` describes the actual
host/software/run; examine it alongside per-case NCU records before comparing.

Pass the resulting JSON through target FlagGems-sglang benchmark `--base-data`.
Only equal configuration keys should match; locally timed Torch reference-as-
candidate results validate the pipeline, not an optimized kernel.
