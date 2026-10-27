Native H800 full-API Flash baseline for the original Enflame core grid

Collected all18 original cases using the official ops-baseline collector: four generic shapes (1024,2048,4096,8192 sequence length) plus five explicit causal/non-square/scale/dropout/debug profiles, each Float16/BFloat16. No case or input configuration was reduced. The callable is native `aten::_scaled_dot_product_flash_attention.default`, returning its complete nine-output API.

Clean source `e66b1d176ddbeb0536e95d73e43091a893ce12a4` and all1147 tracked files were checked before/after. All18 actual H800 controls independently matched exact original ac9 builders and native output/LSE tensors bit-for-bit. The artifact retains hardware/software/source/run and actual device0 provenance, typed case IDs and full input descriptions.

The initial precondition observed foreign workloads on all cards and prevented collection. Those raw observations remain preserved. Collection began only after three genuinely idle GPU0 observations; no foreign process was stopped.

NCU was disabled. Published records preserve config, T_us and latency_ms exactly and omit unmeasured zero/unavailable NCU placeholders. Original raw data SHA256 is bound in provenance. These are native H800 latencies; CPU-oracle candidate execution on another backend is separate and is not presented as optimized target performance.
