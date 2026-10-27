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

Collection used physical GPU5 after fresh occupancy samples, standard `triton.testing.do_bench` budgets (warmup25ms/rep100ms), and `--no-ncu`. Occupancy samples are boundary observations, not proof of continuous exclusive ownership. Profiling fields populated with zeros by the stock latency-only collector mean **not collected**, not measured zero utilization or traffic. Use `T_us` for this snapshot's latency comparisons.

The immutable artifact SHA256 is `7efb83703ec03252703e7f79b95e194a03b2c1f608594513fca1540457444990`. Target FlagGems benchmarks can consume exact records for references unavailable on their target, while timing injected candidates locally. CPU-oracle latency must not be presented as an NVIDIA baseline. This data does not prove target-backend correctness or performance.
