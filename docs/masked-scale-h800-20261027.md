# Native H800 masked-scale core baseline

`op_pytorch_h800_masked_scale_core_baseline.json` contains exactly five measured Float32 records for FlagGems `_masked_scale`, collected through the `masked_scale` PyTorch module and native `aten::_masked_scale.default`. Original ac9 core shapes, including both distinct Gi-element geometries, Gaussian seed0 input, subsequent uint8 randint(0,2) mask and scale2.0 are preserved.

Actual clean collection source: `335d9a9179c04034d5a1e77c291666857984a922`,1147 tracked source files verified before/after in a real Git checkout. Physical GPU3: NVIDIA H800 UUID `7d545def-d5ba-4424-b859-2067806337c2`. The existing official collector ran `--repo pytorch --ops masked_scale --devices 3 --no-ncu --op-timeout 3600`, writing outside source. Per-record origins retain actual worker environment, task and raw artifact/worker hashes; provenance also distinguishes coordinator metadata from the measured worker.

Native H20 and H800 controls demonstrate numerical multiplication by the uint8 mask, including2/255, NaN/Inf with zero masks and signed zeros, plus empty/noncontiguous ABI, nonaliasing and unchanged inputs. A prior torch.where counterexample is preserved separately. NaN classification agrees; NaN payload bit identity across CPU/CUDA is not claimed. The portable CPU oracle is a correctness/bootstrap route, never the performance baseline.

Public records retain raw config,T_us,latency_ms exactly, with positive finite measured latency and explicit measured validity. Uncollected NCU throughput/utilization placeholders are omitted. Full raw metadata remains in the evidence. A first precollection attempt passed native controls but observed620MiB immediately after their process exited, so its idle guard prevented starting the collector. A new isolated attempt recorded bounded quiescence observations and started only at0MiB/0%.

This is a static latency snapshot, not a resume input. Backend RNG payloads are not claimed byte-identical. No shared collector or existing large snapshot changes.
