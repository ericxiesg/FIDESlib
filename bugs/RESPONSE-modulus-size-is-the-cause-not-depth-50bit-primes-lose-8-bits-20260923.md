# Modulus size is the cause, not depth: 50-bit primes lose 8 bits vs 59-bit

**Date:** 2026-09-23

## Summary

**The 27-bit gap is caused by the modulus size (50/55 vs 59/60), NOT by depth (23 vs 37) or key distribution (sparse vs uniform).** Sweeping depth and modulus size independently:

| Config | depth | scaling_bits | first_mod_bits | RMS | Bits |
|---|---|---|---|---|---|
| depth23 mod59/60 | 23 | 59 | 60 | 6.17e-05 | **14.0** |
| depth23 mod50/55 | 23 | 50 | 55 | 0.0155 | **6.0** |
| depth30 mod50/55 | 30 | 50 | 55 | 0.0155 | **6.0** |
| depth37 mod50/55 | 37 | 50 | 55 | 0.0156 | **6.0** |
| depth37 mod59/60 | 37 | 59 | 60 | 6.16e-05 | **14.0** |

**Depth has NO effect** — 6.0 bits at depth 23, 30, and 37 with mod50/55.
**Modulus size has a LARGE effect** — 14.0 bits with mod59/60 vs 6.0 bits with mod50/55, at every depth.

The error is a **power of 2** in both cases:
- mod50/55: error = 2^-6.0
- mod59/60: error = 2^-14.0

The difference is 8 bits, close to the 9-bit difference in scaling_bits (59-50=9).

## What this means

1. **Depth is irrelevant** — the per-stage A/B at depth 23 was testing the wrong axis
2. **The sparse EvalMod fix** (commit `144e5de`) is valid but not the THOR bug — THOR uses the dense path
3. **The issue is in how the GPU handles smaller primes** (50-bit vs 59-bit scaling modulus)

The error scaling `2^-(scaling_bits - 44)` suggests a systematic issue where each rescale or coefficient encoding contributes error proportional to `2^-scaling_bits`, and there are ~44 such operations (or one operation with a 2^44 factor).

## CPU comparison

CPU with mod50/55 gets 33 bits. GPU with mod50/55 gets 6 bits. GPU with mod59/60 gets 14 bits. The GPU gap vs CPU:
- mod50/55: 33 - 6 = 27 bits
- mod59/60: 33 - 14 = 19 bits (assuming CPU also gets 33 bits with mod59/60)

The GPU is worse than CPU at both modulus sizes, but the gap widens with smaller moduli.

## What to investigate next

The modulus size affects:
1. **Rescale rounding**: each rescale divides by a scaling prime and rounds. With 50-bit primes, rounding error is 2^-50; with 59-bit, it's 2^-59. But this is 1 ulp per rescale and can't explain 8 bits alone.
2. **Chebyshev coefficient encoding**: `double` coefficients are scaled by the scaling modulus to convert to integers. With 50-bit modulus, the encoding precision is 50 bits; with 59-bit, it's 59 bits. If the GPU uses a different encoding path (e.g., FP32 instead of FP64), the loss would be larger for smaller moduli.
3. **Barrett reduction parameters**: `ALGO_BARRETT` uses precomputed `mu` values. These might be less accurate for smaller primes, but the collaborator confirmed Barrett is exact integer arithmetic.
4. **NTT twiddle factors**: different prime sizes use different twiddle factors, but NTT is exact.

The most likely candidate is #2 — the coefficient encoding. If the GPU encodes `double` coefficients using FP32 intermediate values, the precision loss would be proportional to the modulus size, exactly matching the observed pattern.

## Test code

```python
configs = [
    ('depth23 mod59/60', 23, 59, 60),
    ('depth23 mod50/55', 23, 50, 55),
    ('depth30 mod50/55', 30, 50, 55),
    ('depth37 mod50/55', 37, 50, 55),
    ('depth37 mod59/60', 37, 59, 60),
]
# Each: Engine(0, log_n=16, depth=depth, scaling_bits=sb, first_mod_bits=fmb,
#        dnum=4, secret_key_dist=SPARSE_TERNARY, bootstrap_level_budget=(3,3),
#        truncate_keys=True, allow_key_grow=True)
# Input: uniform(0.0814, 0.9812, seed=23)
```

## Follow-up: fine-grained modulus sweep

Swept `scaling_bits` at depth 37, SPARSE_TERNARY, holding `first_mod_bits` at 55 (except mod59/60):

| Config | scaling_bits | first_mod_bits | Bits | Notes |
|---|---|---|---|---|
| mod50/55 | 50 | 55 | 6.0 | THOR config |
| mod50/60 | 50 | 60 | NaN | Broken — incompatible |
| mod52/55 | 52 | 55 | 10.6 | Jump of 4.6 bits |
| mod54/55 | 54 | 55 | 10.5 | Plateau |
| mod56/55 | 56 | 55 | ERROR | scaling > first_mod |
| mod58/55 | 58 | 55 | ERROR | scaling > first_mod |
| mod59/55 | 59 | 55 | ERROR | scaling > first_mod |
| mod59/60 | 59 | 60 | 14.0 | Jump of 3.5 bits |

**Pattern:**
1. **mod50→mod52**: +4.6 bits (scaling_bits is the bottleneck)
2. **mod52→mod54**: plateau at ~10.5 bits (first_mod_bits=55 becomes the bottleneck)
3. **mod54→mod59/60**: +3.5 bits (first_mod_bits=60 removes the bottleneck)

The plateau at mod52-54/55 suggests the precision is limited by `min(scaling_bits, first_mod_bits - margin)`. With first_mod=55, the cap is ~10.5 bits. With first_mod=60, the cap is ~14 bits.

**The GPU precision is limited by both scaling_bits AND first_mod_bits, while the CPU achieves 33 bits with the same mod50/55.** This suggests the GPU has a fundamentally different (lower-precision) code path for encoding Chebyshev coefficients or performing the polynomial evaluation.
