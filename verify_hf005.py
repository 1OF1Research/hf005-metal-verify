#!/usr/bin/env python3
"""One-command verification for transformers #49337 (MetalLinear 3D inputs).

Run this on an Apple Silicon Mac with torch (MPS), transformers and the
`kernels` package installed:

    python3 verify_hf005.py

Stage 1 runs the issue's repro against your installed transformers and
reports whether the fault is present (every batch row after the first
wrong / non-finite on 3D inputs). Stage 2 applies the flatten/restore
fix from hf005_metal_linear_3d_fix.patch in-process (your install is
not modified) and re-runs: every row must agree with the per-row
reference and be finite. Paste the SUMMARY block back into
https://github.com/huggingface/transformers/issues/49337 either way —
a FAULT PRESENT + FIX CONFIRMED pair is the completed verification;
different numbers are new evidence.
"""
import sys

import torch

if not torch.backends.mps.is_available():
    print("MPS is not available on this machine — this check needs an "
          "Apple Silicon Mac with a Metal-capable torch build.")
    sys.exit(2)

from transformers.integrations.metal_quantization import (
    MetalLinear,
    _affine_quantize_tensor,
)

RTOL, ATOL = 0.02, 0.125  # the issue repro's own tolerances


def run_case(bits):
    torch.manual_seed(42)
    K, N, GROUP = 512, 256, 64
    w, s, b = _affine_quantize_tensor(torch.randn(N, K), GROUP, bits)
    layer = MetalLinear(K, N, bits=bits, group_size=GROUP).to("mps")
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.scales.copy_(s)
        layer.qbiases.copy_(b)
    x = torch.randn(4, 3, K, device="mps", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = layer(x)
        flat = layer(x.reshape(-1, K)).reshape(4, 3, N)
        rows = torch.stack([layer(row) for row in x])
    torch.mps.synchronize()
    finite_by_batch = torch.isfinite(actual).all(-1).all(-1).tolist()
    err_3d = (actual.float() - rows.float()).abs().amax((1, 2)).tolist()
    err_flat = (flat.float() - rows.float()).abs().max().item()
    ok = all(finite_by_batch) and all(e <= ATOL + RTOL * 1.0 for e in err_3d)
    return finite_by_batch, err_3d, err_flat, ok


def patched_forward(self, input):
    if self.weight.dtype != torch.uint32:
        return torch.nn.functional.linear(input, self.weight, self.bias)
    from transformers.integrations.metal_quantization import _get_metal_kernel

    kernel = _get_metal_kernel()
    input_shape = input.shape
    flat_input = input.reshape(-1, input_shape[-1])
    output = kernel.affine_qmm_t(
        flat_input,
        self.weight,
        self.scales.to(input.dtype),
        self.qbiases.to(input.dtype),
        self.group_size,
        self.bits,
    )
    output = output.reshape(*input_shape[:-1], output.shape[-1])
    if self.bias is not None:
        output = output + self.bias
    return output


print("=== Stage 1: installed transformers, as-is ===")
fault_bits = []
for bits in (8, 4):
    finite, err3d, errflat, ok = run_case(bits)
    print(f"bits={bits}  3D finite by batch: {finite}")
    print(f"bits={bits}  3D vs per-row max abs error by batch: "
          f"{[round(e, 4) for e in err3d]}")
    print(f"bits={bits}  flattened vs per-row max abs error: {errflat:.6f}")
    print(f"bits={bits}  -> {'OK (no fault)' if ok else 'FAULT PRESENT'}")
    if not ok:
        fault_bits.append(bits)

print("\n=== Stage 2: with the flatten/restore fix applied in-process ===")
MetalLinear.forward = patched_forward
fix_ok = True
for bits in (8, 4):
    finite, err3d, errflat, ok = run_case(bits)
    print(f"bits={bits}  3D finite by batch: {finite}")
    print(f"bits={bits}  3D vs per-row max abs error by batch: "
          f"{[round(e, 6) for e in err3d]}")
    print(f"bits={bits}  -> {'FIX CONFIRMED' if ok else 'FIX FAILED'}")
    fix_ok = fix_ok and ok

print("\n=== SUMMARY (paste this into issue #49337) ===")
print(f"fault present on installed transformers: "
      f"{'yes, bits ' + str(fault_bits) if fault_bits else 'no'}")
print(f"flatten/restore fix confirmed on this hardware: "
      f"{'yes' if fix_ok else 'NO — see stage 2 numbers above'}")
sys.exit(0 if fix_ok else 1)
