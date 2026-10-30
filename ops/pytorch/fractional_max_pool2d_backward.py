"""Exact shared/split FlagGems backward core recipes; native CUDA backward.

Pinned FG b976f497: shared factory keeps caller RNG, split factory resets seed0.
The actual input and original FP32 random samples produce flat int64 indices
through real CPU Double forward, matching the ASC fixture's interval precision.
Fixture construction is outside the native backward latency clock. CUDA tensor
bytes are freshly sampled; no cross-device random-byte identity is claimed.
"""

import torch
from _kernelgen import generate_tensor_input

OP_NAME = "fractional_max_pool2d_backward"
DTYPES = [torch.float16, torch.float32, torch.bfloat16]
IS_INPLACE = False


def native():
    return torch.ops.aten.fractional_max_pool2d_backward


def grid():
    return [
        dict(
            source=source,
            shape=list(shape),
            kernel_size=2 if source == "shared" else [2, 2],
            output_size=[shape[-2] // 2, shape[-1] // 2],
        )
        for source, shapes in [
            (
                "shared",
                [
                    (4, 3, 224, 224),
                    (16, 64, 56, 56),
                    (32, 128, 28, 28),
                    (64, 256, 14, 14),
                    (128, 512, 7, 7),
                ],
            ),
            ("split", [(4, 3, 32, 32), (8, 16, 28, 28), (2, 8, 16, 20)]),
        ]
        for shape in shapes
    ]


def key_shape(binding):
    return binding["source"] + "_" + "x".join(map(str, binding["shape"]))


def build_inputs(binding, dtype, device):
    if binding["source"] == "split":
        torch.manual_seed(0)
    inp = generate_tensor_input(tuple(binding["shape"]), dtype, device)
    if binding["source"] == "shared":
        inp.requires_grad_(True)
    samples = torch.rand(
        inp.shape[0], inp.shape[1], 2, dtype=torch.float32, device=device
    )
    output, indices = torch.nn.functional.fractional_max_pool2d(
        inp.detach().to(device="cpu", dtype=torch.float64),
        kernel_size=binding["kernel_size"],
        output_size=binding["output_size"],
        return_indices=True,
        _random_samples=samples.to(device="cpu", dtype=torch.float64),
    )
    output = output.to(device=device, dtype=dtype)
    indices = indices.to(device=device)
    grad_output = torch.randn_like(output)
    return (
        grad_output,
        inp,
        binding["kernel_size"],
        binding["output_size"],
        indices,
    ), {}
