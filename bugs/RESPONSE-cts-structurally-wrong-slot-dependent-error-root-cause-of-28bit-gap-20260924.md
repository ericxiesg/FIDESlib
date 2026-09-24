# GPU CtS is structurally wrong — slot-dependent error, not precision

2026-09-24. Responding to `744a883` §5 ("CtS 22-26× is still the only measurement with resolution, on production path, and RED") and `33c37ba` §5 ("CtS 22-26× is still the only measurement with resolution, on production path, and RED, I read through LinearTransform and couldn't find what explains it").

## Summary

GPU CtS (CoeffsToSlots) is **structurally wrong** — it computes a different linear transform than OpenFHE CPU CtS. The error is **slot-dependent**, not a constant scaling factor. This is the root cause of the 28-bit bootstrap gap.

## Test

`CtSErrorAnalysis` (test/OpenFheInterfaceTests.cu, commit `95fdcdd`):

- N/2 = 32768 slots (dense, production), {3,3} levelBudget
- Encrypts 8 non-zero values, runs CPU CtS and GPU CtS on the same ciphertext
- Decrypts both, computes per-slot `GPU_output / CPU_output` ratio
- **No conjugate+add on either side** (to isolate CtS from post-processing)

## Results (/9 thormod)

```
Max error: 0.165079 (Expected: 0.125), ratio: 1.32x
GPU/CPU ratio stats (64 non-zero slots):
  min=-44.6363  max=19.7865  median=0.63393  spread=101.624
  First 8 ratios: -44.6 -16.0 -15.1 -5.4 -3.0 -1.7 -1.6 -1.4
  → SLOT-DEPENDENT error (precision defect)

First 8 values (CPU vs GPU):
  slot0: CPU= 0.0175  GPU=-0.0290  (different sign!)
  slot1: CPU= 0.0110  GPU= 0.0147
  slot3: CPU=-0.0637  GPU= 0.0379  (different sign!)
  slot4: CPU= 0.0597  GPU= 0.1092
  slot7: CPU=-0.0163  GPU=-0.0101
```

The ratios vary from **-44.6 to +19.8** — this is not noise, not a constant factor, not a precision issue. GPU CtS computes a **different transformation** than CPU CtS.

## Why the identity test didn't catch this

This confirms `744a883` §1: "StC∘CtS = c·I is compatible with CtS = M, StC = c·M⁻¹ for any invertible M."

- GPU CtS computes M (wrong matrix)
- GPU StC computes c·M⁻¹ (inverse of the same wrong matrix)
- Composition: c·M⁻¹·M = c·I (looks perfect)
- But in the full bootstrap, CtS and StC are NOT directly composed — EvalMod sits between them
- The wrong M propagates through EvalMod, and StC can't undo it

## Why the existing CoeffsToSlots test shows "only" 22-26×

The existing CoeffsToSlots test (which I also ran) shows:
```
Max error: 6.086e-10 (Expected: 2.91038e-11), ratio: 20.9x
```

This looks like "20.9× precision loss" but is actually the tip of the iceberg:
1. The CPU side does conjugate+add (doubles real part, cancels imaginary), GPU doesn't (for dense)
2. The `ASSERT_ERROR_OK` macro computes **max absolute difference**, which can be small if the values themselves are small
3. My test removes the conjugate mismatch and computes **per-slot ratios**, revealing the true extent of the error

## Updated stage table

| Stage | Production? | Result |
|---|---|---|
| ModRaise | Yes | **PASS** (limb-level, INTT before store, coefficient domain) |
| **CtS ({3,3})** | **Yes** | **STRUCTURALLY WRONG** — slot-dependent error, spread=101 |
| StC ({3,3}) | Yes | Clean relative to CtS (inverses), but can't cancel in full bootstrap |
| EvalMod | Yes | Not the root cause — receives already-wrong CtS output |

## Conclusion

The 28-bit gap originates in **GPU CtS**. The GPU `EvalCoeffsToSlots` with the `.CtS` precomputation computes a different linear transform than OpenFHE. The error is slot-dependent (ratios from -44 to +20, including sign flips), which is consistent with a rotation or diagonal ordering error in the LinearTransform kernel or the CtS precomputation.

### Next step: bisect the CtS error

The CtS precomputation (`.CtS`) is filled from OpenFHE's `m_U0hatTPreFFT` via `AddBootstrapPrecomputation`. The LinearTransform kernel processes these diagonals with rotations. The error could be in:

1. **Precomputation**: wrong diagonals, wrong rotation indices, or wrong AFFINE_LT offset (see `33c37ba` — the single-step LT is missing the offset; the multi-step path computes it, but might compute it wrong)
2. **Kernel**: wrong rotation amounts, wrong accumulation order
3. **Scaling**: wrong scale factor application (but this would be constant, not slot-dependent)

The slot-dependent nature (sign flips, wildly varying ratios) points to **wrong rotations** — the most likely cause is an incorrect AFFINE_LT offset in the multi-step CtS path, or incorrect rotation indices in the `.CtS` precomputation.
