# BUG: multScalar under FIXEDMANUAL corrupts he_exp in softmax — root cause of poor GPU fidelity

## Summary

Per-stage probes inside the softmax (v11, commit 824e9f1) show that **he_exp (the exponential
polynomial evaluation) produces values at 1e+124** — astronomically larger than the expected
~1e0.  The bootstrap and score refresh are correct; the corruption begins at the polynomial
evaluation, which uses `EvalMultScalar` → `multScalar(double)`.

Additionally, commit `43e1c56` fixed `EvalNegate` by replacing `multScalar(-1.0)` with
`negate()`, but **four other call sites** in `CryptoContext.cpp` still use `multScalar(-1.0)`,
which increments `NoiseLevel` and multiplies `NoiseFactor` by `Delta` under FIXEDMANUAL.
These affect `EvalSub(double, ct)` used by `he_inv` in the softmax, but he_exp is broken
first.

## Softmax probe data (v11)

```
  [probe] 07a.refreshed_scores     8 ct  level 18  min -10.43  max +12.48  |x| med 0.9272    ✅ CORRECT
  [probe] 07b.exp                  8 ct  level 10  min -1.96e+124  max +1.836e+124  |x| med 2.907e+123  ❌ BROKEN
  [probe] 07c.denominator          1 ct  level 10  min -3.613e+123  max +3.573e+123  |x| med 7.162e+122  ❌ (sum of broken exp)
  [probe] 07d.inverse_denominator  1 ct  level 14  min +0  max +0  all zero                ❌ (he_inv on garbage)
```

The refreshed scores (07a, after bootstrap + split) are correct — values ~10, which is 2x
the attention scores.  But he_exp (07b) produces 1e+124, which is wrong by ~Delta^8
(if Delta ≈ 2^50).  The polynomial evaluation consumes 8 levels (18 → 10), suggesting the
scale error accumulates across the polynomial's baby-step/giant-step evaluation.

## Per-stage fidelity (v10/v11, identical)

```
  query     scale 2.0000  relRMSE 2.884e-07  ✅
  value     scale 2.0000  relRMSE 3.141e-07  ✅
  scores    scale 0.5000  relRMSE 2.873e-07  ✅
  softmax   scale 0.0000  relRMSE 1.000e+00  ❌ FIRST DIVERGENCE
  attention_dense  scale 0.0000  relRMSE 1.000e+00  ❌ downstream
  norm_1    scale 0.0000  relRMSE 1.000e+00  ❌ downstream
  intermediate+  CRASH: SetLevel multiplicative depth [37] insufficient
```

## Per-stage proof (v10, commit 5fa4c2a)

```
  query                 scale   2.0000   relRMSE 2.884e-07    ✅ PERFECT
  value                 scale   2.0000   relRMSE 3.141e-07    ✅ PERFECT
  scores                scale   0.5000   relRMSE 2.873e-07    ✅ PERFECT
  softmax               scale   0.0000   relRMSE 1.000e+00    ❌ BROKEN — first divergence
  attention_dense       scale   0.0000   relRMSE 1.000e+00    ❌ downstream garbage
  norm_1                scale   0.0000   relRMSE 1.000e+00    ❌ downstream garbage
  intermediate+         CRASH: SetLevel: multiplicative depth [37] is insufficient
```

Commit `43e1c56` hypothesised that `norm_1` would be the first bad stage (from the
`EvalNegate` → `he_invsqrt` → LayerNorm chain).  The per-stage data disproves this:
**softmax is the first bad stage**, because `he_inv` (not `he_invsqrt`) is the first
function to call `subtract(float, ct)` → `EvalSub(double, ct)` → `multScalar(-1.0)`.

## The bug

`Ciphertext::multScalar(double c)` (Ciphertext.cpp:886) does:
- `NoiseLevel += 1`
- `NoiseFactor *= Delta`
- Rescale is guarded by `(rescale && FIXEDMANUAL) && FIXEDAUTO` — **never true**

So `multScalar(-1.0)` leaves the ciphertext at `Delta^2` while looking like `Delta^1`.
The value decrypts correctly (NoiseFactor is tracked), but the next `addScalar` or
`addPt` that assumes canonical `Delta^1` is wrong by a factor of `Delta`.

`Ciphertext::negate()` (Ciphertext.cpp:901) multiplies each limb by `q_i - 1` (integer
negation), which touches neither NoiseLevel nor NoiseFactor.  This is the correct fix.

## Remaining multScalar(-1.0) calls (api/CryptoContext.cpp)

### 1. `EvalSub(ct, pt)` — line 1073

```cpp
res_gpu->multScalar(-1.0);   // BUG: NoiseLevel++
res_gpu->addPt(*pt_gpu);     // -ct + pt = pt - ct (also wrong sign!)
```

**Should be:** `res_gpu->subPt(*pt_gpu);`  (computes `ct - pt`, correct sign + scale)

Note: This function also has a **sign error** — it computes `pt - ct` instead of `ct - pt`.
The Python wrapper's `subtract(ct, numpy)` calls this via `EvalSubPt`, expecting `ct - pt`.

### 2. `EvalSub(double scalar, ct)` — line 1122  ← **ROOT CAUSE**

```cpp
res_gpu->multScalar(-1.0);   // BUG: NoiseLevel++
res_gpu->addScalar(scalar);  // -ct + scalar = scalar - ct (correct sign, wrong scale)
```

**Should be:**
```cpp
res_gpu->negate();           // scale-neutral: -ct
res_gpu->addScalar(scalar);  // -ct + scalar = scalar - ct (correct sign + scale)
```

This is called by `subtract(float, ct)` → `EvalScalarSub(float, ct)` in `he_inv`:
```python
correction = self.subtract(2 / k * b.delta, b.ciphertext)
```

### 3. `EvalSubInPlace(double scalar, ct1)` — lines 1188, 1190

```cpp
res_gpu->multScalar(-1.0);   // BUG: NoiseLevel++
res_gpu->addScalar(scalar);  // -ct + scalar = scalar - ct
res_gpu->multScalar(-1.0);   // BUG: NoiseLevel++ again → NoiseLevel += 2 total
```

**Should be:** `res_gpu->addScalar(-scalar);`  (computes `ct - scalar`, scale-aware, no negate needed)

Note: This function may also have a sign error — it computes `ct - scalar` but the function
name suggests `scalar - ct`.

### 4. Line 1073 was already in `EvalSub(ct, pt)` — see #1 above.

## Why the scale mismatch check (4a2023d) didn't catch this

The check in commit `4a2023d` only guards `addPt` and `subPt`.  `EvalSub(double, ct)` uses
`multScalar` + `addScalar`, neither of which is checked.  `addScalar` is scale-aware (it
reads the ciphertext's `NoiseLevel`), so it silently uses the wrong `NoiseLevel` without
throwing.

## Why v6 and v8 results are identical

The subtract fix (commit `113e854`) changed the **numpy** branch of `subtract()`:
```python
# Before: EvalAddPt(EvalNegate(y), pt)  — broken because EvalNegate was broken
# After:  EvalNegate(EvalSubPt(y, pt))  — works because EvalSubPt is canonical
```

But `he_inv` uses the **float** branch: `subtract(float, ct)` → `EvalScalarSub(float, ct)`
→ `EvalSub(double, ct)` → `multScalar(-1.0)`.  This branch was never changed, so v6 and v8
produce identical results.

## Fix

Replace all `multScalar(-1.0)` in `api/CryptoContext.cpp` with `negate()` or `subPt`/
`addScalar(-scalar)` as appropriate:

| Function | Line(s) | Current | Fix |
|---|---|---|---|
| `EvalSub(ct, pt)` | 1073-1074 | `multScalar(-1.0); addPt(pt)` | `subPt(pt)` |
| `EvalSub(double, ct)` | 1122-1123 | `multScalar(-1.0); addScalar(s)` | `negate(); addScalar(s)` |
| `EvalSubInPlace(double, ct)` | 1188-1190 | `multScalar(-1.0); addScalar(s); multScalar(-1.0)` | `addScalar(-s)` |

## Reproduction

```bash
export PYTHONPATH=/home/zhiyuan/workspace/THOR-v2/FIDESlib/python:/home/zhiyuan/bench-run
export LD_LIBRARY_PATH=/home/zhiyuan/workspace/THOR-FIDE/openfhe-install/lib:/usr/local/cuda-12.9/lib64
python3 -u -m thorfhe.bench fhe \
    --engine fideslib --device cuda:0 \
    --depth 37 --dnum 4 --bootstrap-level-budget 3,3 \
    --binary-rotations --refresh-after-dense \
    --layers 1 --limit 1 \
    --per-stage --device-memory --offline
```

Per-stage log: `/home/zhiyuan/bench-run/gpu-depth37-v10.log`
Probe log: `/home/zhiyuan/bench-run/gpu-depth37-v11.log`

## Additional finding: multScalar(double) may corrupt he_exp under FIXEDMANUAL

The he_exp polynomial evaluation uses `multiply(ct, float)` → `EvalMultScalar` →
`multScalar(double)`, which does `NoiseLevel += 1` and `NoiseFactor *= ScalingFactor`.
The Python code follows each multiply with `rescale()`, which should restore NoiseLevel.

However, the probe shows the he_exp output is wrong by ~Delta^8 (8 levels consumed),
suggesting the NoiseLevel/NoiseFactor tracking drifts across the polynomial's
baby-step/giant-step evaluation.  The `level_down(merged, 3)` call before he_exp
(softmax.py:175) may also interfere: after dropping 3 limbs, the scaling factor at
the new level may not match what `multScalar` and `rescale` expect.

The `multScalar(double)` function (Ciphertext.cpp:886) has `assert(NoiseLevel == 1)`
which is compiled out in release builds, so any NoiseLevel drift is silent.

## Two separate bugs

1. **he_exp corruption** — `multScalar(double)` under FIXEDMANUAL may not correctly
   track NoiseLevel/NoiseFactor across the polynomial evaluation, especially after
   `level_down`.  This is the FIRST corruption (probe 07b).

2. **multScalar(-1.0) in EvalSub** — the `EvalSub(double, ct)` and related functions
   still use `multScalar(-1.0)` instead of `negate()`.  This would corrupt `he_inv`
   (probe 07d), but he_exp is broken first so he_inv never gets valid input.

Both need to be fixed for the softmax to work correctly.
