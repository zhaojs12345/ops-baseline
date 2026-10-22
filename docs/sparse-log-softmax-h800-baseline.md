# Sparse log-softmax native baseline preparation

This source prepares two native PyTorch GPU modules for the original297 suite:
`_sparse_log_softmax` and `_sparse_log_softmax_backward_data`. It does not collect
or publish timings yet. Existing historical backward records are not this new
input recipe and must not be relabelled.

The exact case source is FlagGems `f30a563479449016490ef674b3670ede6c101988`.
The original own benchmark hashes are embedded in each module. Forward has nine
core descriptor variants, backward eleven; both retain float32 and float64 on
H800, so a complete collection contains **18 + 22 = 40 records**. Their original
pytest case IDs, including the backward `_benchmark` suffix, are exported in
`config`. This is not the previous sparse-softmax pair's six-plus-six grid.

Forward uses all four original shapes and every original `(sparse_dim, dim)`
pair, including dense-block reduction axes. Its deterministic grouped support
is sorted and unique, `is_coalesced=True`, and stored count is the original
`min(sparse_extent,8192)` rule. Values have the full trailing dense block.

Backward preserves the eleven descriptors, including duplicate coordinates at
shape4096/nnz8192, dense blocks, scalar nnz1 and nnz0, and empty nnz0. **grad,
output and self are all complete sparse COO tensors** of the descriptor shape,
with equal coordinate content but independent indices/values storage and
independently drawn normal values. Output is not replaced by probabilities or a
computed forward result. Self is not a dense `[1]` placeholder. Original stored
order and automatic empty/singleton coalesced metadata are retained.

## Canonical input recipe

`coo-log-softmax-pcg64-native-width-compact-v1` uses a local NumPy PCG64 generator
seeded by recipe/kind/shape/nnz/sparse_dim/dim/dtype. It changes random payloads
from historical Torch RNG samples. It preserves descriptor/support/independent
normal-value semantics, not historical tensor bits. Float64 draws are generated
at native width, never through float32. No global Python/NumPy/Torch RNG is set.

`ops/pytorch/_sparse_log_softmax_inputs/__init__.py` is a private package, not a
third operator module. Consumers should copy the necessary fixture/interface
code into their two own FlagGems benchmarks; target runtime must not import the
Ops checkout. The public fixture interface consists of `seed_hex`,
`forward_cpu`, `backward_cpu`, `input_identity` and `to_device`.

Input identities cover every defined byte of each compact indices/values
component, dtype, shape, stride, offset, storage extent, lazy flags, sparse/dense
dimensions, stored count, coalesced status, gradient flags and all component
storage-alias pairs. Absolute addresses and device ordinals are intentionally
not cross-machine keys. Transport independently checks the exact requested
physical device and full identity, using byte-preserving copies. Meta tensors
cannot establish physical identity and are not a supported collection target.
All native timing must execute direct CUDA ATen on the actual target inputs;
CPU construction/identity calculation is preparation, **not a timed CPU oracle**.

Bitwise reproducibility across arbitrary NumPy/OS combinations is not assumed.
Future target replay must compare actual complete input identity to the H800
config and reject a mismatch. No H800 result or target GPU replay is claimed by
CPU tests. No NCU metrics, GPU latency or performance speedup exists yet.

## Review and collection plan (not executed)

After independent source review and a clean pinned producer commit, use an
isolated H800 checkout and the existing authorized runtime, with no upgrades.
The official tool command is:

```sh
python tools/collect_baseline_nvidia.py --repo pytorch \
  --ops _sparse_log_softmax,_sparse_log_softmax_backward_data \
  --device 7 --no-ncu --output /data/tj/workspace/OWNED_RUN/native.json
```

The hardware/preflight/ownership gate, clean source and before/after hashes,
per-op worker environment, all forty case configs, finite positive `T_us`,
units and completed tasks must be independently reviewed before an artifact is
accepted. Preserve underscore operator keys explicitly when reading the JSON;
metadata filtering must not drop these two real names. A partial/error task is
not completed native data and absent data is never zero latency. Ops PRs remain
unmerged unless separately authorized.

CPU checks: `python -m unittest discover -s tests -p
test_sparse_log_softmax_canonical.py -v`. They exercise all40 config/builder
identities, native CPU literal formulas and readonly inputs, duplicate/scalar/
empty cases, private RNG, exact width/storage/alias checks and corrupted-copy
rejection. Legacy backward manifest membership and all11 descriptors/dtypes
remain tested; its physical canonical inputs are validated on CPU instead of
comparing real bytes to undefined meta tensors. A separate external audit runs
the pinned original coordinate helpers and compares all20 support descriptors.
