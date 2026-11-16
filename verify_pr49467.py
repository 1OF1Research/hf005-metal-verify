#!/usr/bin/env python3
"""Stage-1-only check of transformers PR #49467 (the maintainer's
kernel-side fix for #49337) against the issue's repro.

Install transformers from the PR branch, then run:

    python3 verify_pr49467.py

Unlike verify_hf005.py this script applies NO in-process fix: it runs
MetalLinear exactly as the installed (PR-branch) transformers ships it,
on 3D inputs, at 8 and 4 bits, and compares every batch row against
per-row calls. If the kernel-side route fixes the fault, the 3D output
agrees with the per-row reference at both widths with no
transformers-side flatten anywhere.
"""
import sys

import torch

if not torch.backends.mps.is_available():
    print("MPS is not available on this machine — this check needs an "
          "Apple Silicon Mac with a Metal-capable torch build.")
    sys.exit(2)

import transformers

try:
    from transformers.integrations.metal_quantization import affine_quantize as _quantize
    _quantize_is_mx = True
except ImportError:
    from transformers.integrations.metal_quantization import _affine_quantize_tensor as _quantize
    _quantize_is_mx = False

from transformers.integrations.metal_quantization import MetalLinear

print(f"transformers: {transformers.__version__} ({transformers.__file__})")
print(f"quantize entry point: {'affine_quantize' if _quantize_is_mx else '_affine_quantize_tensor'}")

RTOL, ATOL = 0.02, 0.125  # the issue repro's own tolerances


def run_case(bits):
    torch.manual_seed(42)
    K, N, GROUP = 512, 256, 64
    w, s, b = _quantize(torch.randn(N, K), GROUP, bits)
    layer = MetalLinear(K, N, bits=bits, group_size=GROUP).to("mps")
    with torch.no_grad():
        layer.weight.copy_(w)
        layer.scales.copy_(s)
        layer.qbiases.copy_(b)
    x = torch.randn(4, 3, K, device="mps", dtype=torch.bfloat16)
    with torch.inference_mode():
        actual = layer(x)
        rows = torch.stack([layer(row) for row in x])
    torch.mps.synchronize()
    finite_by_batch = torch.isfinite(actual).all(-1).all(-1).tolist()
    err_3d = (actual.float() - rows.float()).abs().amax((1, 2)).tolist()
    ok = all(finite_by_batch) and all(e <= ATOL + RTOL * 1.0 for e in err_3d)
    return finite_by_batch, err_3d, ok


all_ok = True
for bits in (8, 4):
    finite, err3d, ok = run_case(bits)
    print(f"bits={bits}  3D finite by batch: {finite}")
    print(f"bits={bits}  3D vs per-row max abs error by batch: {[round(e, 6) for e in err3d]}")
    print(f"bits={bits}  -> {'OK (no fault)' if ok else 'FAULT PRESENT'}")
    all_ok = all_ok and ok

print(f"\nPR #49467 as-is, no transformers-side fix: {'FAULT GONE at both widths' if all_ok else 'FAULT STILL PRESENT'}")
sys.exit(0 if all_ok else 1)
