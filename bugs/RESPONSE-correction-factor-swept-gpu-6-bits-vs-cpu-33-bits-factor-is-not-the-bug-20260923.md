# Correction Factor Swept: GPU 6 bits vs CPU 33 bits; Factor Is Not the Bug

## The test

`test_bootstrap_correction_factor.py` sweeps correction factor 0/7/9/10/11/12/13.
One engine per factor, one bootstrap of `uniform(0.0814, 0.9812, seed=23)`, report
max and RMS error vs input.

## Results

| Factor | CPU max err | CPU RMS | CPU bits | GPU max err | GPU RMS | GPU bits |
|--------|------------|---------|----------|------------|---------|----------|
| 0      | 9.29e-10   | 1.06e-10| 33.1     | 0.066      | 0.0156  | 6.0      |
| 7      | 9.72e-10   | 1.07e-10| 33.1     | 0.072      | 0.0154  | 6.0      |
| 9      | 7.43e-10   | 1.07e-10| 33.1     | 0.266      | 0.062   | 4.0      |
| 10     | 7.19e-10   | 1.07e-10| 33.1     | 0.653      | 0.124   | 3.0      |
| 11     | 8.54e-10   | 1.07e-10| 33.1     | 1.030      | 0.247   | 2.0 (FAIL)|
| 12     | 9.30e-10   | 1.08e-10| 33.1     | 1.059      | 0.249   | 2.0 (FAIL)|
| 13     | 8.74e-10   | 1.08e-10| 33.1     | —          | —       | CRASH    |

## Interpretation

### Correction factor is not the bug

CPU is 33.1 bits at every factor — completely flat. The factor has no effect on
CPU precision, so it is not the missing parameter.

### GPU is 27 bits below CPU at every factor

At factor 0 (OpenFHE's default = 7):
- CPU: 33.1 bits (RMS 1.06e-10)
- GPU: 6.0 bits (RMS 0.0156)

The GPU RMS 0.0156 matches the "sigma = 0.0156" floor reported in
`RESPONSE-bootstrap-floor-noise-20260916.md`. That is not a noise floor — it is a
**27-bit precision deficit** vs CPU on the same input, same parameters, same
correction factor.

### GPU degrades sharply above factor 7

CPU stays at 33 bits through factor 13. GPU drops from 6 bits (factor 0/7) to
2 bits (factor 11/12) and crashes at factor 13. The correction factor amplifies
the signal 2^k on each side of ModRaise, so larger k amplifies whatever the GPU
is doing wrong. This is consistent with a structural error in the GPU bootstrap
that is independent of the correction factor but is worsened by it.

### The 6-bit GPU floor is the bootstrap bug

The original slot-0 corruption (13000x at the multiply) was measured on
bootstrapped ciphertexts. A bootstrap that only achieves 6 bits of precision
produces ciphertexts with RMS 0.0156 error, which is large enough to cause
catastrophic failures in the Goldschmidt iteration (which needs 2e-4 precision
on the denominator). The slot-0 corruption and the 6-bit bootstrap floor may
be the same underlying bug at different magnifications.

## What this rules out

- Correction factor: CPU is flat at 33 bits across all values
- CtS/StC diagonal encoding: confirmed by Part23's upstream comparison, and CPU
  achieves 33 bits with the same diagonals
- `broadcastLimb0_` ISU64 guard: confirmed by `b4ffc8a` that `constants.type`
  is all-ones, guard always taken

## What this points to

The GPU bootstrap computes a different result than CPU with a 27-bit precision
gap. The computation is deterministic (confirmed by
`test_stage_stability_one_ciphertext.py` — bit-identical across runs on the same
ciphertext), so this is a **systematic numerical error**, not a race or
uninitialized memory.

The most likely candidates for a 27-bit systematic error:
1. **NTT/INTT correctness** — if the GPU NTT uses different roots or a different
   algorithm than CPU, the error would be deterministic and structural
2. **Rescale/moddown precision** — FIXEDMANUAL rescale drops a prime; if the GPU
   rescale rounds differently, errors accumulate through 37 levels
3. **Key switch precision** — the fused key switch path may lose precision vs
   CPU's direct path

## Test

`python/tests/test_bootstrap_correction_factor.py`. Run with
`PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_bootstrap_correction_factor.py`.
Note: factor 13 crashes the GPU (segfault).
