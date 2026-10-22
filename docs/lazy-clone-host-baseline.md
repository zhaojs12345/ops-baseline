# Native `_lazy_clone` host metadata baseline

`aten::_lazy_clone` creates copy-on-write storage metadata for a CUDA tensor.
It does not copy the tensor payload during the call. CUDA-event elapsed time
and NCU memory traffic therefore do not measure this API's host dispatch cost.

Use the dedicated host collector from a clean ops-baseline feature checkout:

```sh
python tools/collect_lazy_clone_host.py --device 7 --threads 1 \
  --min-run-time 0.05 --output /outside/the/checkout/lazy-clone-host.json
```

The producer under `ops/pytorch/_lazy_clone.py` preserves all 60 original
Kunlunxin core cases: six shape/layout plans and ten dtypes. Inputs use the
original `torch.empty` recipe, with the original transposed layouts. No tensor
values or storage pointers are read. No output materialization or device
synchronization is included in the timer. Native dispatch and destruction of
the returned metadata object are included. Preparation and metadata checks
are outside the timer.

The artifact must have `_status == "complete"` and 60 unique original case
IDs. Each record contains `T_host_us`, input geometry, source/config hashes,
and explicit host measurement metadata; the environment identifies both the
H800 CUDA input device and the host CPU. Failed/partial runs stay incomplete.
There are no placeholder GPU compute, bandwidth or NCU metrics. Positive
CUDA-event results from the earlier diagnostic route must not be relabeled
as this host baseline.

A target-backend consumer must validate the original complete input identity
and time its candidate with a host wall-clock timer as well. Any CPU-oracle
transfer/materialization required by that candidate is part of its call cost;
the NVIDIA reference still measures the original native lazy API. Such a ratio
compares API call costs, not GPU kernel throughput or copy bandwidth. CPU
self-comparison verifies harness execution, not an independent generated kernel.

Do not skip every underscore-prefixed operator key when looking up legacy GPU
diagnostics: `_lazy_clone` is an API name, distinct from reserved metadata keys.
This host artifact uses an explicit `operator` field and `records` list.
