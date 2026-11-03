# Consolidated native PyTorch / ATen H800 records

`op_pytorch_h800_perf_baseline.json` now also contains the direct native CUDA
records from 42 open baseline PRs: 49 source artifacts, 67 operator keys and
1,422 record observations. These are observations from separate collections,
not 1,422 deduplicated workloads or newly measured results. No GPU was used for
this consolidation. Public `torch.*` entrypoints that call the native operator
are included; a native operation may itself launch multiple kernels.

The original main artifact at commit
`546f12986bcd78cc1c681b6123ebf52a9a55db01` is retained without changing any
historical operator data or collection metadata. Original PR branches, producer
code, standalone snapshots, and FlagGems consumers are not modified. This is a
data-only consolidation; it does not merge or certify the source PRs, install
their producers, rerun numerical validation, or change existing consumers to
read a new artifact. Consumers pinned to the old standalone snapshot digests
continue to use those snapshots until explicitly migrated.

## Record layout and selection

Operator entries retain the existing `{native_api, shapes: {key: {dtype: record}}}`
format. Imported records use source-qualified shape keys:

```text
pr<number>:<original artifact stem>:<original shape key>
```

This prevents a different seed, stride, scalar, overload, state-restoration
protocol, or collection run from overwriting a record with the same shape and
dtype. Original operator keys are preserved, including existing producer aliases
such as `linalg_lu_solve_current_core`; aliases are not silently equated to other
native APIs. Every imported record dictionary, configuration and latency remains
exactly equal to its source. Historical, unprefixed shape keys are unchanged.

`_aten_consolidation.sources` maps the original PR, exact source head, artifact
Git blob/SHA256, operator, and original shape key to its new key. Each source
retains its own complete original top-level metadata and per-entry metadata.
Do not assign the primary file's historical `_environment` or the operator's
historical `native_api` to an imported record; use that source's metadata.
Original collection headers can describe excluded entries from a mixed source;
only the operators explicitly listed in the source index were imported.

Select a known source and exact original case, then perform the same full input,
dtype, overload and timing-contract checks as the original consumer. Do not pick
the newest record, match shape alone, or strip the prefix and combine records:

```python
import json

with open("op_pytorch_h800_perf_baseline.json") as stream:
    baseline = json.load(stream)

source_id = "pr24:op_pytorch_h800_neg_view_copy_perf_baseline"
source = baseline["_aten_consolidation"]["sources"][source_id]
operator = "_neg_view_copy"
original_shape_key = "functional__64x64"
shape_key = source["operators"][operator]["shape_keys"][original_shape_key]
record = baseline[operator]["shapes"][shape_key]["torch.float16"]
assert record["config"]["variant"] == "functional"
```

Adjacent provenance sidecars are retained byte-for-byte in
`adjacent_provenance[filename].raw_json` with their original hashes. Keeping them
as strings preserves native numerical-control examples containing NaN/Infinity
without introducing nonstandard numeric values into this JSON. Imported timing
values are finite and positive; their original metric availability is unchanged.
No missing NCU counters are invented.

This consolidated file is a published data snapshot, not a collector `--resume`
checkpoint. A subsequent collector run must use a separate output path.

## Deliberately excluded

- Host metadata: PRs #23 (`is_set_to`), #25 (nested metadata), #26 (`_lazy_clone`).
  These are native ATen operations but their work uses a host clock, so they stay
  in their dedicated host artifacts.
- CPU reference plus device transfers: #51 (histogram), #57 (fused Adagrad).
- Handwritten or substituted references: #31 (RMS backward), #34 (GPU LSTM
  composition), #41 (floating LSTM for the quantized-LSTM test), #44 (portable
  packing), #67 (overrideable-attention composition), #69 (per-group matmul).
- Mixed PR #32: exclude `linalg_polar` and `linalg_polar_out` SVD compositions;
  import only its native ATen entries.
- Mixed PR #50: import only the direct `softplus_backward` profile; leave the
  prepared autograd profile unchanged.
- Mixed PR #56: exclude the Lanczos composition; import the native median and
  row-stack entries.

## Verification

The offline checks reconstruct every selected original operator entry from the
consolidated file and compare it with its pinned original Git blob. They also
subtract all indexed imports and compare the entire remainder with the original
main artifact. Both checks pass, including historical metadata and all record
fields. All 1,422 imported timings are positive, finite and unit-consistent;
shape destinations are unique; all 39 adjacent provenance embeddings preserve
their original bytes. Strict JSON parsing rejects duplicate keys and nonstandard
numeric constants. The resulting file remains below GitHub's 100 MiB limit.

## Included source PRs

The following table counts source observations, including repeated measurements
or overlapping cases from separate sources.

| PR | Imported operator keys | Observations |
|---|---|---:|
| [#24](https://github.com/zhaojs12345/ops-baseline/pull/24) | `_neg_view_copy` | 18 |
| [#28](https://github.com/zhaojs12345/ops-baseline/pull/28) | `_sparse_csr_prod` | 54 |
| [#29](https://github.com/zhaojs12345/ops-baseline/pull/29) | `_sparse_softmax`, `_sparse_softmax_backward_data` | 24 |
| [#30](https://github.com/zhaojs12345/ops-baseline/pull/30) | `_sparse_log_softmax`, `_sparse_log_softmax_backward_data` | 40 |
| [#32](https://github.com/zhaojs12345/ops-baseline/pull/32) | `_flash_attention_forward`, `_scaled_dot_product_flash_attention`, `addmm_`, `baddbmm_`, `pack_padded_sequence`, `quantized_max_pool2d`, `quantized_max_pool2d_out`, `scaled_dot_product_flash_attention` | 196 |
| [#33](https://github.com/zhaojs12345/ops-baseline/pull/33) | `nuclear_norm` | 6 |
| [#35](https://github.com/zhaojs12345/ops-baseline/pull/35) | `cudnn_batch_norm_backward` | 4 |
| [#36](https://github.com/zhaojs12345/ops-baseline/pull/36) | `cholesky_inverse` | 8 |
| [#37](https://github.com/zhaojs12345/ops-baseline/pull/37) | `ldexp`, `ldexp_out` | 54 |
| [#38](https://github.com/zhaojs12345/ops-baseline/pull/38) | `embedding_bag_dense_backward` | 21 |
| [#39](https://github.com/zhaojs12345/ops-baseline/pull/39) | `nanquantile`, `nanquantile_scalar_out` | 14 |
| [#40](https://github.com/zhaojs12345/ops-baseline/pull/40) | `float_power_scalar_tensor_out`, `float_power_tensor_scalar_out`, `float_power_tensor_tensor_out` | 45 |
| [#42](https://github.com/zhaojs12345/ops-baseline/pull/42) | `masked_scale` | 5 |
| [#43](https://github.com/zhaojs12345/ops-baseline/pull/43) | `conv_depthwise2d_current_core` | 12 |
| [#45](https://github.com/zhaojs12345/ops-baseline/pull/45) | `cudnn_batch_norm_backward_current_core` | 4 |
| [#46](https://github.com/zhaojs12345/ops-baseline/pull/46) | `_scaled_dot_product_flash_attention` | 18 |
| [#47](https://github.com/zhaojs12345/ops-baseline/pull/47) | `_adaptive_avg_pool3d_backward` | 12 |
| [#48](https://github.com/zhaojs12345/ops-baseline/pull/48) | `thnn_fused_gru_cell_backward_current_core`, `thnn_fused_lstm_cell_backward_current_core`, `thnn_fused_lstm_cell_backward_impl_current_core` | 51 |
| [#49](https://github.com/zhaojs12345/ops-baseline/pull/49) | `median` | 25 |
| [#50](https://github.com/zhaojs12345/ops-baseline/pull/50) | `softplus_backward` | 15 |
| [#52](https://github.com/zhaojs12345/ops-baseline/pull/52) | `kaiser_window_current_core`, `ldexp_inplace_current_core` | 30 |
| [#53](https://github.com/zhaojs12345/ops-baseline/pull/53) | `_aminmax` | 30 |
| [#54](https://github.com/zhaojs12345/ops-baseline/pull/54) | `special_round` | 15 |
| [#55](https://github.com/zhaojs12345/ops-baseline/pull/55) | `lu_with_info_current_core` | 5 |
| [#56](https://github.com/zhaojs12345/ops-baseline/pull/56) | `median`, `median_dim`, `median_dim_aten_current_core`, `row_stack_out` | 119 |
| [#58](https://github.com/zhaojs12345/ops-baseline/pull/58) | `b03_activation`, `b03_activation_legacy`, `b03_activation_out`, `b03_activation_out_legacy`, `b03_addbmm` | 180 |
| [#59](https://github.com/zhaojs12345/ops-baseline/pull/59) | `ldexp_out` | 15 |
| [#61](https://github.com/zhaojs12345/ops-baseline/pull/61) | `sgn_out` | 30 |
| [#64](https://github.com/zhaojs12345/ops-baseline/pull/64) | `_linalg_solve_ex` | 72 |
| [#65](https://github.com/zhaojs12345/ops-baseline/pull/65) | `_lu_with_info` | 5 |
| [#66](https://github.com/zhaojs12345/ops-baseline/pull/66) | `_masked_scale` | 5 |
| [#70](https://github.com/zhaojs12345/ops-baseline/pull/70) | `linalg_ldl_factor_current_core` | 10 |
| [#71](https://github.com/zhaojs12345/ops-baseline/pull/71) | `addmm_activation_current_core`, `addmm_activation_out_current_core` | 150 |
| [#72](https://github.com/zhaojs12345/ops-baseline/pull/72) | `linalg_cond_current_core`, `linalg_svd_current_core`, `linalg_svdvals_current_core`, `pinverse_current_core` | 25 |
| [#73](https://github.com/zhaojs12345/ops-baseline/pull/73) | `weight_norm_backward_current_extra_core` | 4 |
| [#74](https://github.com/zhaojs12345/ops-baseline/pull/74) | `nextafter_inplace_legacy_core`, `nextafter_inplace_own_core` | 35 |
| [#75](https://github.com/zhaojs12345/ops-baseline/pull/75) | `linalg_lu_solve_current_core` | 10 |
| [#76](https://github.com/zhaojs12345/ops-baseline/pull/76) | `linalg_tensorinv_out_current_core` | 5 |
| [#77](https://github.com/zhaojs12345/ops-baseline/pull/77) | `orgqr_current_core` | 10 |
| [#78](https://github.com/zhaojs12345/ops-baseline/pull/78) | `cholesky_solve_helper_rowmajor_core` | 5 |
| [#79](https://github.com/zhaojs12345/ops-baseline/pull/79) | `float_power_inplace_kunlun_core` | 12 |
| [#80](https://github.com/zhaojs12345/ops-baseline/pull/80) | `fractional_max_pool2d_backward` | 24 |
