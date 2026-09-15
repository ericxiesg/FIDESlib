# Bootstrap leaves NoiseLevel=2 under FIXEDMANUAL (THOR crash root cause)

**Date:** 2026-09-15  
**Status:** Fix applied in `pyfideslib/__init__.py`, workaround confirmed  
**Severity:** Critical — prevents any THOR FHE run from completing  

## Summary

`EvalBootstrap` on the GPU backend returns a ciphertext with `NoiseLevel = 2`
(scale Δ²) instead of `NoiseLevel = 1` (scale Δ). Under FIXEDMANUAL, `NoiseLevel`
**adds** on every multiplication and only subtracts 1 on rescale, so starting at 2
instead of 1 causes it to grow exponentially through the degree-15 polynomial
evaluation in `he_exp`. By the time softmax finishes, `NoiseLevel` reaches 12108,
and the first `EvalAddPt` in `stage_10_attention_dense` crashes with a scale
mismatch.

## Reproduction

```python
from pyfideslib import _core, Engine
import numpy as np

engine = Engine("cuda:0", log_n=16, depth=37, scaling_bits=50,
                first_mod_bits=55, dnum=3,
                scaling_technique=_core.FIXEDMANUAL,
                secret_key_dist=_core.SPARSE_TERNARY,
                bootstrap_level_budget=(3, 3),
                rotation_indexes=[1, 2, 4, 8, 16, 32],
                truncate_keys=True, allow_key_grow=True)

ct = engine.encrypt(np.ones(32768, dtype=complex))
print(f"Fresh:            level={engine.level(ct)} noise={engine.noise_level(ct)}")

bt = engine.bootstrap(ct)
print(f"After bootstrap:  level={engine.level(bt)} noise={engine.noise_level(bt)}")
# Expected: noise=1   Actual: noise=2
```

**Output (before fix):**
```
Fresh:            level=37 noise=1
After bootstrap:  level=21 noise=2    ← BUG
```

## Root cause analysis

The CKKS bootstrap on the GPU (`EvalBootstrap` → `approxModReduction`) ends with
`multIntScalar` followed by a single `rescale()` (line 80-82 of
`ApproxModEval.cu`). The `multIntScalar` multiplies c0 and c1 by an integer
scalar but does **not** change `NoiseLevel`. The `rescale()` decrements
`NoiseLevel` by 1.

If the bootstrap's internal Chebyshev / double-angle arithmetic leaves the
ciphertext at `NoiseLevel = 3` before the final `multIntScalar + rescale`, the
result is `NoiseLevel = 2` — which is what we observe.

Under FIXEDMANUAL the caller has no automatic scale adjustment, so this
`NoiseLevel = 2` propagates into every downstream operation:

| Operation              | NoiseLevel | Level |
|------------------------|------------|-------|
| Fresh encrypt          | 1          | 37    |
| **After bootstrap**    | **2**      | 21    |
| multiply(1/32)         | 3          | 21    |
| rescale                | 2          | 20    |
| square (NoRelin)       | 4          | 20    |
| relinearize            | 4          | 20    |
| rescale                | 3          | 19    |

Since `he_exp` evaluates a degree-15 polynomial via baby-step giant-step (5
levels of multiplications), and each multiplication **adds** the two operands'
`NoiseLevel` values, the `NoiseLevel` compounds exponentially. After the full
softmax (exp + Goldschmidt division + squarings), `NoiseLevel` reaches 12108.

The crash then occurs at `stage_10_attention_dense` when `EvalAddPt` is called
with a ciphertext at `NoiseLevel = 12108` and a plaintext at `NoiseLevel = 1`:

```
RuntimeError: FIDESlib: addPt with mismatched scales - the ciphertext is at
noise level 12108 and the plaintext at 1. Under FIXEDMANUAL the caller must
bring them to the same scale first.
```

## The crash site (before fix)

```
[probe] 07a.refreshed_scores     8 ct  level 18  min -10.45   max +12.45     ← correct
[probe] 07b.exp                  8 ct  level 10  min -2e+124  max +2e+124    ← overflow
[probe] 07c.denominator          1 ct  level 10  min -3.6e+123 max +3.6e+123 ← overflow
[probe] 07d.inverse_denominator  1 ct  level 14  min +0       max +0         ← all zero
RuntimeError: addPt with mismatched scales (noise level 12108 vs 1)
```

The `1e124` value is the signature described in `clear.py:98-100`:
> `power_basis` squared without relinearising and made `he_exp` return 1e124
> on the device while exact arithmetic, which has no third component to lose,
> stayed happy.

## Fix

Added a `rescale()` after `EvalBootstrap` in `pyfideslib/__init__.py:bootstrap()`,
gated on `FIXEDMANUAL`:

```python
out = self.cc.EvalBootstrap(x)
if self.scaling_technique == _core.FIXEDMANUAL:
    out = self.cc.Rescale(out)    # NoiseLevel 2 → 1, costs 1 level
```

**After fix:**
```
Fresh:            level=37 noise=1
After bootstrap:  level=21 noise=2
After rescale:    level=20 noise=1    ← fixed
```

The rescale costs one level (21 → 20) but the decrypted values are unchanged
(0.99787 → 0.99787, no precision loss). THOR's depth budget of 37 has enough
headroom for this extra level.

## THOR FHE run result (after fix)

```
encrypted run: 1 of 12 layers on fideslib, 4 samples
  layer 0 completed in 1245.92s (all 4 samples)

accuracy (against the dataset labels)
  plaintext   accuracy 100.00%  F1 100.00%
  encrypted   accuracy  50.00%  F1  0.00%

per-stage fidelity (sample 0):
  query              MAE 1.889e-07  relRMSE 2.884e-07    ← excellent
  value              MAE 1.204e-07  relRMSE 3.141e-07    ← excellent
  scores             MAE 3.765e-07  relRMSE 2.873e-07    ← excellent
  softmax            MAE 5.628e+00  relRMSE 1.155e+02    ← still wrong
  attention_dense    MAE 6.092e+01  relRMSE 2.896e+02    ← propagated
  ...
```

The crash is fixed and the first 6 stages (query, key, value, scores, rotated,
softmax exp) now produce correct values. The softmax **inverse denominator**
(Goldschmidt division) still produces garbage (1e+178), which is a separate
issue — see "Remaining problem" below.

## Remaining problem: Goldschmidt inverse denominator

After the bootstrap fix, the softmax probes show:

```
[probe] 07b.exp                  max +0.0002257     ← correct (was 1e+124)
[probe] 07c.denominator          max +0.0005854     ← correct (was 1e+123)
[probe] 07d.inverse_denominator  max +2.926e+178    ← WRONG (should be ~1700)
```

The exp and denominator are now correct, but `he_inv` (Goldsmidt iteration in
`numeric.py:172`) still produces garbage. This is likely another `NoiseLevel`
or scale tracking issue inside the Goldschmidt loop — the iteration bootstraps
the denominator again (`self.bootstrap(denominator)` at line 184), and the
`_restore_magnitude` function (line 208) does `add(ct, conjugate(ct))` and
`multiply(ct, factor)` which may not correctly track `NoiseLevel` under
FIXEDMANUAL.

This is a separate bug to investigate.

## Config differences: THOR vs bert-tiny

| Parameter            | THOR                          | bert-tiny                     |
|----------------------|-------------------------------|-------------------------------|
| depth (L)            | 37                            | 23-25                         |
| scaling_technique    | FIXEDMANUAL                   | FLEXIBLEAUTO                  |
| secret_key_dist      | SPARSE_TERNARY                | UNIFORM_TERNARY               |
| binary-rotations     | yes (15 keys)                 | no (210 keys)                 |
| refresh-after-dense  | yes                           | no                            |
| N (ring dimension)   | 2^16 (32768 slots)            | 2^16 (32768 slots)            |
| bootstrap budget     | {3, 3}                        | {3, 3}                        |
| slot layout          | block-diagonal + window rotate| square row-major              |
| softmax scale        | 1/512 (layer 0)               | 1/64                          |

**Key insight:** bert-tiny used FLEXIBLEAUTO, which auto-rescales and is immune
to the `NoiseLevel = 2` bootstrap bug. THOR uses FIXEDMANUAL, which requires
explicit rescale management and exposes the bug.

## Files changed

- `python/pyfideslib/__init__.py` — added `self.scaling_technique` attribute and
  `rescale()` after bootstrap under FIXEDMANUAL
- `python/pyfideslib/__init__.py.bak` — backup of original
