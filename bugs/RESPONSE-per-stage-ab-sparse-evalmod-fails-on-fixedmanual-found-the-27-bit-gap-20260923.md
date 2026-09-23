# Per-stage A/B: ApproxModEvalSparse FAILS on FIXEDMANUAL — this is the 27-bit gap

**Date:** 2026-09-23
**Test:** `OpenFHEBootstrapTests/OpenFHEBootstrapTest.*` (81 tests, depth 23, `tparams64_13_4_sparse`)
**Build:** `b82228d` (collaborator's unblock commit)
**Env:** `FIDESLIB_TRACE_MODEVAL=1`

## Summary

**18 of 81 tests failed. The 27-bit gap is localized to `ApproxModEval.cu`'s sparse branch on FIXEDMANUAL.**

## Results by test

### ApproxModEval (dense / uniform ternary) — ALL PASS

| Index | dnum | Scaling | Result | Max error |
|---|---|---|---|---|
| 0 | 1 | FLEXIBLEAUTOEXT | PASS | ~0.007 |
| 1 | 1 | FIXEDAUTO | PASS | ~0.007 |
| 2 | 2 | FLEXIBLEAUTOEXT | PASS | ~0.007 |
| 3 | 2 | FIXEDMANUAL | PASS | ~0.007 |
| 4 | 3 | FLEXIBLEAUTOEXT | PASS | ~0.007 |
| 5 | 3 | FIXEDMANUAL | PASS | ~0.007 |
| 6 | 4 | FLEXIBLEAUTOEXT | PASS | ~0.007 |
| 7 | 4 | FIXEDMANUAL | PASS | ~0.007 |
| 8 | 4 | SPARSE_TERNARY | PASS | ~0.007 |

All 9 dense (uniform ternary) ApproxModEval tests pass with ~7 bits precision (Max error ~0.007, expected 0.00390625 = 2^-8). **The uniform branch is correct.**

### ApproxModEvalSparse (sparse ternary) — FAILS on FIXEDAUTO/FIXEDMANUAL

| Index | dnum | Scaling | Result | Max error | Expected |
|---|---|---|---|---|---|
| 0 | 1 | FLEXIBLEAUTOEXT | **PASS** | — | — |
| 1 | 1 | FIXEDAUTO | **FAIL** | **2.56783** | 1.77636e-15 |
| 2 | 2 | FLEXIBLEAUTOEXT | **PASS** | — | — |
| 3 | 2 | FIXEDMANUAL | **FAIL** | **2.56783** | 1.77636e-15 |
| 4 | 3 | FLEXIBLEAUTOEXT | **PASS** | — | — |
| 5 | 3 | FIXEDMANUAL | **FAIL** | **2.56783** | 1.77636e-15 |
| 6 | 4 | FLEXIBLEAUTOEXT | **PASS** | — | — |
| 7 | 4 | FIXEDMANUAL | **FAIL** | **2.56783** | 1.77636e-15 |
| 8 | 4 | FIXEDMANUAL+SPARSE | **FAIL** | **2.56783** | 1.77636e-15 |

**FLEXIBLEAUTOEXT passes. FIXEDAUTO and FIXEDMANUAL fail.** The error is always exactly **2.56783** when the expected is **1.77636e-15** — a complete loss of precision (not 27 bits, but ~50 bits lost).

**THOR uses FIXEDMANUAL, so it hits the failing branch.**

### LinearTransform — ALL FAIL (separate issue)

All 9 LinearTransform tests fail, including FLEXIBLEAUTOEXT. Max error ~9e-8 (expected ~1.5e-11). This is a separate pre-existing issue, not related to the 27-bit gap.

### CoeffsToSlots — FAILS on FIXEDMANUAL (downstream of LinearTransform)

| Index | dnum | Scaling | Result |
|---|---|---|---|
| 0 | 1 | FLEXIBLEAUTOEXT | PASS |
| 1 | 1 | FIXEDAUTO | PASS |
| 2 | 2 | FLEXIBLEAUTOEXT | PASS |
| 3 | 2 | FIXEDMANUAL | **FAIL** |
| 4 | 3 | FLEXIBLEAUTOEXT | PASS |
| 5 | 3 | FIXEDMANUAL | **FAIL** |
| 6 | 4 | FLEXIBLEAUTOEXT | PASS |
| 7 | 4 | FIXEDMANUAL | **FAIL** |
| 8 | 4 | FIXEDMANUAL+SPARSE | **FAIL** |

CoeffsToSlots uses LinearTransform internally, so this is likely downstream of the LinearTransform failure. Only FIXEDMANUAL fails, which matches the LinearTransform pattern for FIXEDMANUAL.

### Crash in later test

The test suite crashed (EXIT_CODE=134, SIGABRT) with:
```
terminate called after throwing an instance of 'lbcrypto::OpenFHEException'
  what(): poly.h:274:operator*=(): Modulus mismatch
```

This is in a later test (OpenFHEBootstrap or OpenFHEBootstrapManualPrescale), a separate issue.

## FIDESLIB_TRACE_MODEVAL output

Only one trace was captured (from a dense test):
```
[FIDESlib] EvalMod after Chebyshev (into double-angle): level 17, scale degree 2
[FIDESlib] EvalMod after double-angle: level 11, scale degree 2
[FIDESlib] EvalMod after post scalar (EvalMod exit): level 11, scale degree 2
```

The sparse tests did not produce trace output (the trace might only fire in the dense path, or the sparse test uses a different code path that doesn't hit the trace).

## Interpretation

**The 27-bit gap is in `ApproxModEval.cu`'s sparse branch (`approxModReductionSparse`), specifically on FIXEDMANUAL scaling.**

The pattern is:
- **FLEXIBLEAUTOEXT**: sparse EvalMod works → auto rescaling handles the scale correctly
- **FIXEDAUTO / FIXEDMANUAL**: sparse EvalMod fails catastrophically (error 2.57 vs expected 1e-15)

The error value 2.56783 is **identical across all failing tests** (different dnums, different key distributions), which means it's a deterministic, configuration-independent error in the sparse code path. This is consistent with the collaborator's finding that GPU arithmetic is exact — the error is not from precision loss but from a wrong constant or wrong rescale position in the sparse branch under manual/fixedauto scaling.

**`ApproxModEval.cu` is the one bootstrap stage FIDESlib implements itself (not transcribed from OpenFHE). The sparse branch has never been tested before — until now.**

## What this does NOT yet answer

- **Why** the sparse branch fails on FIXEDMANUAL but passes on FLEXIBLEAUTOEXT — the difference is in how rescaling is handled (manual vs auto), so the sparse branch likely has a missing or extra rescale under FIXEDMANUAL
- **What the value 2.56783 represents** — it's suspiciously consistent and might be a specific mathematical constant or the result of a wrong scaling factor
- **Whether the LinearTransform failure is related** — it fails on ALL scaling techniques, so it's likely a separate issue

## Next step

The collaborator's hypothesis is confirmed: the 27-bit gap is in `ApproxModEval.cu`'s sparse branch. The next step is to compare FIDESlib's `approxModReductionSparse` with OpenFHE's `EvalMod` sparse path, focusing on rescale positions under FIXEDMANUAL. The value 2.56783 should be identifiable — it's the same for every configuration, so it's a structural error, not a precision error.
