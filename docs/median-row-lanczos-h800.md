# Original median, row-stack and Lanczos core reference

`op_pytorch_h800_median_row_lanczos_union_perf_baseline.json` contains 114 latency records collected with the official PyTorch collector on an NVIDIA H800. The source used for collection is clean Git revision `5b2ca5c0c317f1ea937997edde011509bae84cd6`; the artifact's `_environment` preserves actual source, hardware, software and command metadata. Packaging revisions after collection do not change that source identity.

| Operator | Records | Original core coverage |
| --- | ---: | --- |
| median | 70 | Union of nine NVIDIA and five Enflame shapes × five dtypes |
| median_dim | 20 | Four shape/dim/keepdim bindings × five dtypes |
| row_stack_out | 9 | Three shapes × three dtypes, three input tensors and real out buffer |
| upsample_lanczos2d_aa_vec | 15 | Five shapes × three dtypes and the original scale recipe |

The union preserves the original FlagGems `ac9a1f703757537a3d54cc1ac1febeca410316c0` core grids. NVIDIA and Enflame median core grids differ; neither replaces the other. Existing row-stack and Lanczos modules retain their original recipe provenance. Cross-device random realizations are not claimed bit-identical; shape, dtype, layout, seed/recipe and native call configuration match.

Lanczos measures the original **PyTorch CUDA composition**, not a same-name opaque ATen API. Its weight-matrix, float32 intermediates, casts and scale selection are the original benchmark implementation. Independent CPU math controls were compared against that CUDA composition separately.

Collection used physical GPU5 after fresh occupancy samples, standard `triton.testing.do_bench` budgets (warmup25ms/rep100ms), and `--no-ncu`. Occupancy samples are boundary observations, not proof of continuous exclusive ownership. All unmeasured profiling fields are omitted from this public latency-only artifact. The immutable original collector output remains preserved separately; `_artifact_processing` records its SHA256 and the exact omitted fields. All114 retained latency/config records and the original `_environment` are byte-equivalent under canonical JSON serialization. Use `T_us` for latency comparisons.

The immutable artifact SHA256 is `be758c4f292c1d8ae7b25789b06d2733f23716fde65e798c80c62bc403d16cfb`. Target FlagGems benchmarks can consume exact records for references unavailable on their target, while timing injected candidates locally. CPU-oracle latency must not be presented as an NVIDIA baseline. This data does not prove target-backend correctness or performance.

The separate `op_pytorch_h800_median_dim_aten_current_core_perf_baseline.json` is the exact original median_dim ATen reference: `torch.ops.aten.median.dim(inp, dim, keepdim)`, preserving positional arguments and its plain tuple return. It has20 independently collected H800 records from clean source `9899c7d86491ed071449c8a9502e4fd0d87439ce`. Its published SHA256 is `6633a06619aaeddde7bcf813f8aae2f203da894d5481f897d7ff0915051a744a`, with a separate provenance sidecar. All20 original builder/RNG and native tuple/Long8/readonly/CPU-value controls passed. Profiling placeholders are strictly omitted; actual raw output and normalization history remain distinct.

The20 median_dim records in the earlier114 union measure the public `torch.median` dim frontend, which returns a namedtuple. They remain honestly labeled historical public-API measurements. Use the separate20-record ATen artifact for the original benchmark's exact callable; do not relabel those old20 records as an ATen frontend collection. The median and Lanczos union records retain their original actual source5b provenance.
