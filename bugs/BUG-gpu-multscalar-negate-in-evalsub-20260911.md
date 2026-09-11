# BUG: multScalar(-1.0) in EvalSub/EvalSubInPlace still increments NoiseLevel — root cause of softmax failure

## Summary

Commit `43e1c56` fixed `EvalNegate` by replacing `multScalar(-1.0)` with the scale-neutral
`negate()` method.  However, **four other call sites** in `CryptoContext.cpp` still use
`multScalar(-1.0)`, which increments `NoiseLevel` and multiplies `NoiseFactor` by `Delta`
under FIXEDMANUAL.  The most critical is `EvalSub(double scalar, ct)`, which is called by
`subtract(float, ct)` in `he_inv` (the softmax's Goldschmidt division).  This corrupts the
softmax output, which is the root cause of the poor GPU benchmark fidelity (accuracy 50%,
relRMSE 1.0).

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
