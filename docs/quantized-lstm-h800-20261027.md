Native H800 latency for the original FlagGems quantized LSTM benchmark

All three original shapes × float16/float32/bfloat16 were collected through `tools/collect_baseline_nvidia.py --repo pytorch --ops quantized_lstm --devices 0 --no-ncu` in an isolated clean checkout at `f629b9aa704ec6c57a46b1336ad72e86c75f9a50`. The artifact records actual H800 hardware/software/source/run metadata.

The timed function is the benchmark's original floating `torch._VF.lstm` reference, consuming weights recovered via linear prepack/unpack and dequantization. It is not the numerically different dynamic quantized ATen LSTM. CPU ScriptObjects are not native inputs because the original floating reference discards them; their recipe is recorded. All original value, hidden state, dequantized-weight, dtype, shape and flag configurations remain.

Nine controls compared the reconstructed input tensors bit-for-bit with the exact original benchmark builder and compared native outputs bit-for-bit, on the actual H800. The full shape/dtype case IDs and input recipe are attached to every latency record. CPU/GPU random streams are seeded as the original builder; cross-machine CUDA random payload identity is not claimed.

NCU was disabled. The publication removes uncollected zero/unavailable NCU placeholders and binds the original raw output SHA256 in provenance; only actual latency is published.
