# BUG: power_basis relinearize-before-square produces degree-2 ciphertexts, corrupting he_exp

**Date:** 2025-09-11  
**Severity:** Critical — he_exp outputs 1e+124, softmax fails, full benchmark relRMSE 1.0  
**File:** `python/thorfhe/numeric.py:49`  
**Status:** Fix applied, pending benchmark verification

## Root cause

`power_basis` computes even powers as:

```python
basis[k] = self.rescale(self.square(self.relinearize(basis[half])))
```

The `relinearize` is applied to `basis[half]` **before** `square`, not to the result of `square` **after**. Since `basis[half]` is already degree-1 (it was relinearized when created), the `relinearize` is a no-op. The `square` then produces a **degree-2 ciphertext** (with c2 component) that is never relinearized.

Every other squaring in the codebase uses the correct order `rescale(relinearize(square(...)))`:
- `he_exp` line 116: `self.rescale(self.relinearize(self.square(result)))`  
- `he_invsqrt` line 276: `self.rescale(self.relinearize(self.square(correction)))`  
- `evaluate_polynomial` giant step line 100: `self.rescale(self.relinearize(self.multiply(high, step)))`  
- `power_basis` odd branch line 53: `self.rescale(self.relinearize(self.multiply(left, right)))`  

Only the even-power branch of `power_basis` has the wrong order.

## How it corrupts he_exp

`evaluate_polynomial` calls `power_basis(x, [2, 3] + giants)`. For a degree-16 polynomial, the powers are [2, 3, 4, 8]:

1. `basis[2]` = `rescale(square(relinearize(basis[1])))` — relinearize is no-op (basis[1] already degree-1), square produces **degree-2**, rescale keeps it degree-2.

2. `basis[3]` = `rescale(relinearize(multiply(basis[2], basis[1])))` — `multiply` calls `EvalMultNoRelin(degree-2, degree-1)`. In `Ciphertext::multNoRelin` (Ciphertext.cpp:741):
   ```cpp
   assert(!c2 && !b.c2 && "multNoRelin: operands must be degree-1 ciphertexts");
   ```
   This assert is **disabled** in Release builds (`-DNDEBUG`), so it silently proceeds. The `binomialMult` computation assumes degree-1 × degree-1 and ignores the c2 component of the degree-2 operand, computing:
   - c0 = a0*b0  (correct)
   - c1 = a0*b1 + a1*b0  (correct)
   - c2 = a1*b1  **WRONG** — should be a1*b1 + a2*b0, missing the a2*b0 term
   - c3 = a2*b1  **MISSING** — no c3 component exists

   The resulting ciphertext is mathematically incorrect. The error is proportional to a2*b0, which is the c2 component of basis[2] times b0 of basis[1].

3. The corrupted basis[3] propagates through the baby-step and giant-step evaluation, producing the 1e+124 values seen in probe 07b.

## Why v12 (negate fix) didn't help

The `negate()` fix (commit `ac891e7`) replaced `multScalar(-1.0)` with `negate()` in `EvalSub`/`EvalSubInPlace`. This fixed the NoiseLevel corruption in subtraction, but the he_exp corruption is a **completely separate bug** — it's in the polynomial evaluation's power basis construction, not in subtraction.

## Probe data (v11 and v12, identical)

```
[probe] 07a.refreshed_scores    8 ct  level 18  min -10.42  max +12.41  |x| med 0.9272    ← normal
[probe] 07b.exp                 8 ct  level 10  min -1.924e+124  max +1.888e+124          ← GARBAGE
[probe] 07c.denominator         1 ct  level 10  min -3.735e+123  max +3.557e+123          ← garbage from 07b
[probe] 07d.inverse_denominator 1 ct  level 14  min +0  max +0  all zero                  ← he_inv on garbage
```

Stages 01-06 are perfect (relRMSE ~3e-7). Softmax is the first diverging stage. he_exp (07b) is the first broken sub-step within softmax.

## Impact

This bug affects **all** polynomial evaluations that use even powers > 1:
- `he_exp` (softmax) — confirmed broken via probes
- `he_tanh_for_pooler` (pooler) — uses `evaluate_polynomial`, likely broken
- `he_tanh_for_gelu` (GELU) — uses `evaluate_polynomial`, likely broken

## Fix

```python
# Before (buggy):
basis[k] = self.rescale(self.square(self.relinearize(basis[half])))

# After (fixed):
basis[k] = self.rescale(self.relinearize(self.square(basis[half])))
```

This produces: square (degree-2) → relinearize (degree-1) → rescale (degree-1, NoiseLevel=1). The result is a canonical degree-1 ciphertext, matching what `multNoRelin` expects.
