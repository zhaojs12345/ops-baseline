# Histogram CPU dispatch and transfer baseline

`op_flaggems_h800_histogram_host_baseline.json` contains all 18 original core
cases for `_histogramdd_from_bin_cts`. The original FlagGems benchmark reference
copies its input and optional weight to CPU, calls the native CPU ATen histogram,
and returns the result to the input's accelerator device. These records measure
that complete host dispatch and transfer path; they are not GPU kernel timings.

The source benchmark is unchanged from FlagGems
`ac9a1f703757537a3d54cc1ac1febeca410316c0`. Its original six input shapes,
Float32 dtype, plain/weighted/density modes and input builder are preserved.
The builder resets the random seed to zero. The collector's separate global
seed setting is 20261028. The out overload is not included.

The actual collection source was ops-baseline
`b541d58f14a6b6031b953e861a11b70b9d9e8bad`, with clean Git checkouts before and
after the run. Subsequent changes to the replay consumer do not change that
collection revision. The native CPU wrapper is accepted only by its reviewed
AST identity and public module identity; arbitrary Python algorithms are not
eligible. The artifact preserves the collector/plugin hashes, native API,
wrapper identity, complete typed input configuration and host environment.

Collection used physical H800 GPU 7, UUID
`GPU-555633f8-c39c-bb8a-d549-791cb8451f44`, an Intel Xeon Platinum 8468V host,
Torch `2.8.0a0+5228986c39.nv25.05`, CUDA 12.9 and Python 3.12.3. Inputs were built
with four host threads and measured with one thread. The timer is
`torch.utils.benchmark.Timer`, with a minimum 0.05 second measurement window and
one CUDA synchronization before measurement and synchronization after each
timed complete reference call. There are
no NCU metrics. Idle GPU occupancy was sampled before independent controls,
before collection and after collection; this is not a continuous exclusivity
claim. Independent controls checked all 18 original builders, native CPU
dispatch, values, output metadata and unchanged inputs.

The actual collector command in the isolated H800 container was:

```sh
/usr/bin/python3 /data/tj/workspace/hist-native-host-h800-v1-20261028-b541d58f14/ops-baseline/tools/collect_baseline_host.py \
  --flaggems /data/tj/workspace/hist-native-host-h800-v1-20261028-b541d58f14/FlagGems \
  --output /data/tj/workspace/hist-native-host-h800-v1-20261028-b541d58f14/runs/histogram_host.json \
  --operators _histogramdd_from_bin_cts --device 7 --threads 1 --input-threads 4 \
  --seed 20261028 --min-run-time 0.05 --op-timeout 600
```

It completed all 18 cases, exit 0, in 40.893 seconds. The published artifact is
the exact collected file, SHA256
`f718031bc5bde9cc0c3164c3ba06871b2823cd655e1b7b4440da79dd90f42f24`.

To opt into this static reference when running an injected candidate, add the
tools directory to `PYTHONPATH` and pass `-p pytest_host_baseline --host-baseline
<artifact>` to the existing marker-selected core benchmark command. The consumer
matches the original case ID and complete input contract, maps accelerator roles
across machines, imports only the static reference latency and times the candidate
locally with synchronized host wall time. The shared benchmark class's out
overload retains its original behavior. Ratios are raw source host latency divided
by target host latency; they include transfers, host/software differences and
bootstrap overhead, with no GPU scaling or correctness-golden claim. Target
execution and its environment must be validated separately.
