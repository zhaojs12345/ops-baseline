# Sparse softmax native H800 baselines (preparation)

This feature prepares `_sparse_softmax` and `_sparse_softmax_backward_data` only.
No H800 collection or published artifact is included yet. The forward module is
new; the backward module replaces its old random-input producer. Existing main
contains 12 completed backward records, but their payloads have no canonical
identity and they are **not** the new recipe's records.

## Original cases

Source: FlagGems `38fd6d1ba0374f40f9c60b6291d951e24e52a4c0`, with its
unmodified two own benchmark files. Each operator has six descriptors and
`float32`, plus `float64` on NVIDIA: 12 records/operator, 24 in total. Target
backends retain their original capability-selected dtype subset. The exact
source benchmark SHA256 and original canonical case ID are in each config.
There are no changed shapes, requested nnz, normalization dimensions or
sparse/dense dimension splits.

## Input interface and identity

The private `_sparse_softmax_inputs/` package is a directory, so the collector's
root `ops/pytorch/*.py` scan does not discover it as a third operator.

The copyable CPU-only interface is:

- `forward_cpu(shape, requested_nnz, dim, dtype)` returns a coalesced COO.
- `backward_cpu(shape, requested_nnz, sparse_dim, dim, dtype)` returns independent
  `(grad, output, self_tensor)`. The first two have equal index contents but
  independent index/value storage and remain uncoalesced, as in the original
  builder; `self_tensor` has shape `(1,)`. Output values are independent normal
  samples, **not** a newly computed softmax.
- `seed_hex(...)` binds the full descriptor, operator role, dtype and recipe.
- `input_identity(named_inputs)` records complete defined component bytes,
  dtype/shape/stride/offset/extent/lazy flags, sparse shape/dim/nnz/coalesced
  state, autograd metadata, and every component storage-alias pair.
- `to_device(...)` copies physical bytes into independent target storage, then
  reconstructs the original dtype and checks actual device, dtype, shape and
  complete identity. It is input preparation, never the timed callable.

Copy this interface into each **own** FlagGems benchmark. A target consumer must
not import the Ops repository at runtime or modify shared FlagGems files. It
must check its actual input identity against the pinned H800 config before
importing a static reference latency; the candidate still executes its local
real timer through a direct `flag_gems` callable, without `use_gems`.

The recipe uses a local NumPy PCG64 generator, uniform per-axis coordinates
with replacement and normal values. Float32 is drawn as normal32 and Float64
as normal64, preserving actual double precision. Forward support is stably
sorted and duplicate values are accumulated in original draw order with typed,
unbuffered addition; stored zero sums are kept. Coalescing is outside timing.
This is a **new canonical input recipe**, not bit reproduction of old Torch
random tensors. Neither Torch/Python/NumPy global RNG states are changed.

CPU normal generators can differ across platform/library builds. Config hash
and actual identity equality are authoritative; do not promise arbitrary
platform bit equality. Before formal collection/replay, compare all 24 configs
in the actual H800 and intended target environments. A mismatch is an error,
not permission to drop the identity or substitute a CPU latency.

## Verification and future collection

Focused tests exercise actual CPU ATen on independent literal cases, complete
catalog and genuine F64, duplicate/stored-zero semantics, RNG isolation,
physical transport and bad payload/geometry/alias detection. The legacy
backward manifest member is retained: its reconstruction test now checks every
original descriptor/dtype on real CPU inputs and against its original ABI
metadata, plus the new full identity, rather than skipping it or treating
undefined meta tensors as payload proof. These are CPU controls, not H800 data.

After source reviews and consumer recipe agreement, create an isolated clean
checkout in `flaggems_test_tj` on the designated H800. Reuse its existing runtime;
do not upgrade packages. Record Git/source hashes, full software/hardware and
input configs, per-operator worker results, controller exit and actual GPU
occupancy/PID provenance. A candidate command (not executed here) is:

```sh
python tools/collect_baseline_nvidia.py --repo pytorch \
  --ops _sparse_softmax,_sparse_softmax_backward_data --device 7 \
  --no-ncu --op-timeout 3600 \
  --output /data/tj/workspace/<owned-run>/sparse-softmax-h800.json
```

The official collector times native ATen CUDA with its normal GPU timer. CPU
fixtures/identity observation happen in `build_inputs`/`config`, outside that
measurement. No CPU oracle is the native timing target. Require all 24 records
finite and positive, both task states completed and matching actual source/input
identity before publishing. Preserve `_sparse_*` operator keys during lookup;
underscore prefixes are not sufficient to classify a key as metadata.
