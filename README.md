# HF-005 — MetalLinear 3D verification on real Apple Silicon

Verification home for the fix to
[huggingface/transformers#49337](https://github.com/huggingface/transformers/issues/49337):
`MetalLinear` returns wrong outputs for every batch row after the
first on 3D inputs, because the wrapper hands N-D tensors to the 2D
Metal kernel `affine_qmm_t` unflattened.

The fix (flatten leading dimensions before the kernel call, restore
them after) and its story live in the gift gist:
https://gist.github.com/1OF1Research/00c08569638eb94d3b3560d9abbd4e90

The authors of the fix have no Apple hardware. This repository
exists to close that gap: its GitHub Actions workflow runs the
verification on a `macos-15` runner (Apple Silicon; macos-14 was also tried — see the run history) — Apple Silicon, the real Metal
path — so the last step of the verification is a public,
re-runnable log instead of a promise.

## What runs

`verify_hf005.py` here is byte-identical to the gist's file
(md5 `21fc9079af6d5ca93c28fb3c1104a85a`). One run:

- **Stage 1** reproduces the issue's repro against the installed
  transformers (current `main`, torch 2.15.0): 3D vs per-row outputs
  for 8-bit and 4-bit, finite-by-batch and max errors.
- **Stage 2** applies the flatten/restore fix in-process (the
  install is not modified) and re-runs: every row must be finite
  and agree with the per-row reference.

## Run it

Actions → **verify-hf005** → Run workflow. The job prints the MPS
probe (`torch.backends.mps.is_available()`), runs the script with
full output, and uploads the complete log as the `verify-log`
artifact. Anyone can also run `python3 verify_hf005.py` on any
Apple Silicon Mac with torch, transformers and `kernels`
installed.

## Result of the cloud runs (2026-11-14)

The door was tried, on three runner labels:

- `macos-14` — the platform cancelled both dispatches within
  seconds; no runner was ever assigned.
- `macos-15` — ran end to end
  ([log](https://github.com/1OF1Research/hf005-metal-verify/actions/runs/43670519020)).
- `macos-26-arm64` — ran end to end
  ([log](https://github.com/1OF1Research/hf005-metal-verify/actions/runs/43671084913)).

On both completed runs the probe reads the same, in torch's own
words: `mps built: True`, `mps available: False` — *"Skipping
device Apple Paravirtual device that does not support Metal 2.0"*.
GitHub's macOS runners are virtualized, and the paravirtual GPU
they expose does not support Metal 2.0, so MPS — and with it the
Metal kernel this verification exists to run — cannot start
inside them. The script prints its plain message and stops, as
designed.

So the workflow stands ready and proves everything up to the
Metal call itself, in public, in about two minutes. The last step
still needs a physical Mac — or a future runner image whose
virtual device supports Metal. Re-dispatch the workflow after
any runner-image change and it retests the wall on its own.
