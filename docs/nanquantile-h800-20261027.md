# Native H800 nanquantile core snapshot

This adds `nanquantile` and `nanquantile_scalar_out` native PyTorch collection modules and a latency-only static snapshot. Both reproduce the original FlagGems `ac9a1f703757537a3d54cc1ac1febeca410316c0` core builders from `benchmark/test_nanquantile.py`: seven shapes `(1,)`, `(17,)`, `(1024,)`, `(1025,)`, `(128,128)`, `(512,512)`, `(1024,1024)`, all float32, seed 0, `randn` input with flat positions `[::5]` set to NaN. Tensor q is `[0.1,0.5,0.9]` with dimension -1; scalar-out q is 0.73, dimension -1, keepdim False, linear interpolation and a separate contiguous output buffer. No comprehensive cases are substituted.

The actual clean collection source is `f6e5662ff5efddfaa1234170041821116f8c9783`, based on `7d43707757d04910585b0b97f9bdcb56faa1e56e`. All 1148 tracked source files matched before and after execution and Git status stayed clean. Collection ran in an isolated Git checkout using the unchanged official collector:

```sh
/usr/bin/python3 tools/collect_baseline_nvidia.py --repo pytorch \
  --ops nanquantile,nanquantile_scalar_out --devices 2 --no-ncu \
  --op-timeout 3600 --output <isolated-run>/reports/native-h800.json
```

Physical GPU2 was an idle NVIDIA H800, UUID `f71a6457-0dba-3807-5973-e98752701911`. Torch was `2.8.0a0+5228986c39.nv25.5`, CUDA 12.9, Triton 3.6.0, Python 3.12.3. Fourteen prior native CUDA-versus-CPU controls checked values with NaNs, output dtype/device/shape, original out alias, unchanged inputs and exact typed input configurations. Both collector tasks completed with exit 0, recording seven positive finite timings each.

The public snapshot preserves raw `config`, `T_us` and `latency_ms` values exactly. T_us is microseconds; latency_ms is milliseconds. It omits all unmeasured profiling placeholders and marks only latency as measured. No NCU metrics were collected. Each record retains its actual worker environment and task provenance; this matters because two sequential operators share physical GPU2 and the collector's global per-device environment reflects the last worker.

The adjacent provenance file includes source hashes, actual software/hardware, commands, control results, raw SHA256 and every latency. Complete original raw worker reports, source proofs and bundle are preserved under `context/suiyuan-failed-resume-20261027/b08/h800/` in the task workspace. The normalized artifact is a static comparison snapshot, not a collector resume input.

Backend RNG payloads are not claimed identical: the builders, distributions, NaN locations and constant q recipes match. The collector uses its native `do_bench` warmup/repetition; target FlagGems retains its original timer and fresh-input settings. Enflame currently needs a CPU oracle for the configured native Long-sort gap. A locally timed CPU-oracle candidate is functional harness evidence and must not be presented as an optimized GPU implementation or substituted for the H800 native baseline.
