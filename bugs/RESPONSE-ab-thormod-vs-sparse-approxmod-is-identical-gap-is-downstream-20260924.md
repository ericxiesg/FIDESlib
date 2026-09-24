# A/B thormod vs sparse: ApproxModEval is identical, gap is downstream

**Date:** 2026-09-24
**Collaborator commit:** `b8829e2` (tparams64_13_4_thormod)

## Setup

Ran per-stage A/B with `tparams64_13_4_thormod` (mod50/55, depth 23, should reproduce 6-bit failure) vs `tparams64_13_4_sparse` (mod59/60, depth 23, should pass). Both FIXEDMANUAL, SPARSE_TERNARY, dnum=4.

GPU: Quadro GV100 32GB, `FIDESLIB_TRACE_MODEVAL=1`

## Results

### ApproxModEval — identical at both moduli

| Index | Config | Result | Max error | Expected |
|---|---|---|---|---|
| 8 | sparse (59/60) | **OK** | 0.00730358 | 0.00390625 |
| 9 | thormod (50/55) | **OK** | 0.00683724 | 0.00390625 |

**Both pass with ~7 bits.** The ApproxModEval stage produces the same precision regardless of modulus size. The 27-bit gap is NOT in ApproxModEval.

Trace (sparse): `level 18→15→15, scale degree 2→2→2` (same as before).

### CoeffsToSlots — both fail (pre-existing LinearTransform issue)

| Index | Config | Result | Max error | Expected |
|---|---|---|---|---|
| 8 | sparse (59/60) | **FAIL** | 1.51e-12 | 5.68e-14 |
| 9 | thormod (50/55) | **FAIL** | 6.45e-10 | 2.91e-11 |

Both fail — this is the pre-existing LinearTransform failure that affects all scaling techniques, not the modulus-dependent gap.

### SlotsToCoeffs — crashed

SlotsToCoeffs/8 started but the process crashed with:
```
terminate called after throwing an instance of 'lbcrypto::OpenFHEException'
  what(): poly.h:274:operator*=(): Modulus mismatch
```

### CPU bootstrap at mod59/60 — NOT YET OBTAINED

The standalone CPU test crashed the server (depth 37, ring dim 65536, full key gen is very memory-intensive on CPU). Need to run with reduced parameters or on a machine with more RAM.

## Interpretation

**The modulus-dependent 27-bit gap is NOT in the ApproxModEval stage.** The per-stage test shows identical precision (~7 bits) at both mod50/55 and mod59/60. The gap must be in:

1. **The combination of stages** — individual stages may pass, but error accumulation across the full pipeline differs by modulus
2. **CoeffsToSlots or SlotsToCoeffs** — but both CoeffsToSlots variants fail, so this is a separate issue
3. **ModRaise** — not tested separately (no ModRaise test exists)

The full Python bootstrap shows 6 bits at mod50/55 and 14 bits at mod59/60, but the individual ApproxModEval stage shows ~7 bits at both. The extra precision loss (14→6) must happen outside ApproxModEval.

## What still needs to be done

1. **CPU bootstrap at mod59/60** — the collaborator specifically asked for this. It determines whether the gap is constant (CKKS behavior) or modulus-dependent (GPU defect). The standalone CPU test crashed the server; need a lighter approach.
2. **CoeffsToSlots/SlotsToCoeffs at mod50/55 vs mod59/60** — the current test crashes before completing. Need to fix the crash first.
3. **Full bootstrap A/B** — run the complete bootstrap (not individual stages) at both moduli in the gtest framework, to see where the 14→6 bit drop happens.
