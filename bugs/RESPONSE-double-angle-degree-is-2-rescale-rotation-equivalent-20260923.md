# RESPONSE: applyDoubleAngleIterations entry degree is 2 — rescale rotation is equivalent, NOT the bug

**Date:** 2026-09-23
**Hypothesis tested:** FIDESlib rescales at the **top** of each `applyDoubleAngleIterations` iteration where OpenFHE rescales at the **bottom**. If the degree entering is not 2, the rotated rescale is inequivalent and loses precision.
**Trace instrumented by:** collaborator commit `f1c2717` (`FIDESLIB_TRACE_MODEVAL` env var in `src/CKKS/ApproxModEval.cu`)

## Setup

- THOR BERT-tiny config: log_n=16 (N=65536), slots=32768, depth=37, dnum=4, FIXEDMANUAL
- GPU: Quadro GV100 32GB (sm_70), CUDA 12.9
- Bootstrap with `bench_params()` (correction factor 0 = OpenFHE auto-selects 7)
- Input: `uniform(0.0814, 0.9812, seed=23)`
- Env: `FIDESLIB_TRACE_MODEVAL=1`

## Trace output

```
[FIDESlib] EvalMod after Chebyshev (into double-angle): level 27, scale degree 2
[FIDESlib] EvalMod after double-angle: level 24, scale degree 2
[FIDESlib] EvalMod after post scalar (EvalMod exit): level 24, scale degree 2
[FIDESlib] bootstrap returned scale degree 2, rescaled 1 time(s) to reach the declared 1. Each one costs a level.
max 0.0677169  rms 0.0153622
```

## Interpretation

| Checkpoint | Level | Scale degree |
|---|---|---|
| Into double-angle (after Chebyshev) | 27 | **2** |
| After double-angle | 24 | 2 |
| EvalMod exit (after post-scalar) | 24 | 2 |

The collaborator's hypothesis was:

> "Expected under FIXEDMANUAL if the rotated rescale is equivalent: degree 2 in, degree 2 out. **Degree 1 on entry would be a finding.**"

The degree on entry is **2**, not 1. The rescale rotation (top vs bottom of loop) is therefore **equivalent** — it produces the same result as OpenFHE's bottom-of-loop rescale. This is **not the bug**.

## Precision

Bootstrap result: `rms 0.0153622 = 2^-6.0` → **6 bits**, consistent with all previous measurements.

## Eliminated hypotheses (cumulative)

1. ~~Nondeterminism / race / uninitialized memory~~ — GPU is bit-identical across repeated runs on the same ciphertext
2. ~~`broadcastLimb0_` ISU64 guard skipping uint32 primes~~ — `Parameters::adaptTo` marks all Q primes as U64, guard always executes
3. ~~Correction factor~~ — CPU is 33.1 bits at every factor (0,7,9,10,11,12); GPU is 6 bits at 0/7, degrades at ≥9
4. ~~CtS/StC diagonal encoding error~~ — FIDESlib transcribes `m_U0hatTPreFFT` / `m_U0PreFFT` from OpenFHE, does not encode diagonals itself
5. ~~Weight cache~~ — `bench_params()` builds a fresh Engine, bootstrap diagonals use `MakeCKKSPackedPlaintext`, not the light-plaintext cache path
6. ~~Slot 0 specific corruption~~ — it's a 27-bit global precision deficit, not slot-specific
7. ~~`applyDoubleAngleIterations` rescale position (top vs bottom)~~ — degree on entry is 2, rotation is equivalent

## Remaining suspects

The 27-bit systematic GPU deficit (6 bits vs 33 bits) is **not** explained by any of the above. Remaining areas to investigate:

- **Chebyshev series evaluation itself** (`evalChebyshevSeries`): the degree is 2 going into double-angle, but is the Chebyshev computation itself losing 27 bits? The trace only checks the degree, not the numerical values.
- **ModRaise**: does `grow` + `generateSpecialLimbs(false, true)` + `broadcastLimb0` + NTT introduce noise? The special limb path returns early when `zero_out=false`, potentially retaining stale data.
- **CtS / StC LinearTransform**: the hoisted rotation + fused `DotProductPtInternal` path — is the dot product kernel accumulating correctly on GPU?
- **Key switching**: truncated keys (34 of 48 truncated) — is the truncation introducing extra noise on GPU vs CPU?

## Per-stage A/B still blocked

The OpenFheInterfaceTests per-stage A/B comparison (which would isolate which stage loses the 27 bits) remains blocked by OOM: the test uses full untruncated keys at depth 37 / ring dim 65536, exceeding 32 GB GPU memory.
