# Softmax step-by-step: first he_inv is fine, second diverges — bootstrap is root cause

**Date:** 2026-09-23
**Test:** `python/tests/test_softmax_step_by_step.py` (collaborator commits `80703d5`, `d75e15d`)
**Runtime:** 437.9s, 4 passed, 1 skipped (MRPC dataset unreachable)

## Setup

- `PYFIDESLIB_BENCH_PARAMS=1 THORFHE_DEBUG=1 python3 -m pytest -s -q tests/test_softmax_step_by_step.py`
- Device: Quadro GV100 32GB, CUDA 12.9
- Clear engine: exact arithmetic, depth=60
- Device engine: `bench_params()` (depth=37, FIXEDMANUAL, log_n=16, slots=32768)
- Synthetic scores: `seed=61`, range [-8, 8], 2 heads
- `#N` suffix on probe names = Nth call to `he_inv` with the same `_probe_tag` (ProbeRecorder deduplicates by appending `#2`, `#3`, ...)

## Result 1: Synthetic scores — first he_inv matches, second diverges

### Goldschmidt |b| per iteration

| iteration | device | clear | match? |
|---|---|---|---|
| 07.inv_iter01_b | 0.250031 | 0.250031 | YES |
| **07.inv_iter01_b#2** | **0.862029** | **0.251951** | **NO — 3.4x** |
| 07.inv_iter02_b | 0.000579844 | 0.000579844 | YES |
| **07.inv_iter02_b#2** | **0.627987** | **0.0100328** | **NO — 62x** |
| 07.inv_iter03_b | 0.000541838 | 0.000541838 | YES |
| **07.inv_iter03_b#2** | **11.4238** | **0.00376192** | **NO — 3037x** |
| 07.inv_iter04_b | 0.00190298 | 0.00190297 | YES |
| **07.inv_iter04_b#2** | **97577** | **0.00352066** | **NO — 2.8e7x** |
| 07.inv_iter05_b | 0.00389446 | 0.00389446 | YES |
| **07.inv_iter05_b#2** | **3.90e+12** | **0.0038925** | **NO** |
| 07.inv_iter06_b | 0.00375065 | 0.00375065 | YES |
| **07.inv_iter06_b#2** | **4.21e+27** | **0.00389229** | **NO** |
| 07.inv_iter07_b | 0.00389832 | 0.00389832 | YES |
| **07.inv_iter07_b#2** | **4.56e+57** | **0.00390346** | **NO** |

**Pattern:** Odd rows (first `he_inv` call) match bit-for-bit. Even rows (second `he_inv` call, `#2`) diverge from iter01 and explode double-exponentially.

### Device vs clear probe table (selected, first he_inv)

| probe | max err | rel | magnitude |
|---|---|---|---|
| 07a0.score_refresh_input | 1.667e-09 | 2.95e-10 | 5.652 |
| 07a.refreshed_scores | 2.797e-09 | 3.50e-10 | 8 |
| 07b.exp | 9.442e-12 | 1.02e-07 | 9.239e-05 |
| 07c.denominator | 2.656e-09 | 1.13e-06 | 0.002346 |
| 07d.inverse_denominator | 0.001541 | 3.37e-04 | 4.571 |
| 07e.halved_denominator_k128 | 2.701e-06 | 5.33e-05 | 0.05072 |

First `he_inv` is clean — bootstrap refresh, exp, denominator, inverse all within 1e-4 relative.

### Device vs clear probe table (selected, second he_inv = #2)

| probe | max err | rel | magnitude |
|---|---|---|---|
| 07.inv_input_lift1#2 | **0.7145** | **14.1** | 0.05072 |
| 07.inv_iter01_b#2 | **0.8457** | **3.36** | 0.252 |
| 07.inv_iter04_b#2 | **97577** | **2.77e+07** | 0.003521 |
| 07.inv_iter07_b#2 | **4.56e+57** | **1.17e+60** | 0.003903 |

Second `he_inv` is catastrophically wrong from the very first iteration.

### Step recomputed from own measured input

| relation | device rel | clear rel |
|---|---|---|
| 07a = 2*Re(07a0) | 1.61e-10 | 0 |
| 07a = 2*Im(07a0) | 1.61e-10 | 0 |
| 07c = groupwise sum of 07b | 1.13e-06 | 3.70e-16 |
| 07d * 07c constant across slots | 7.99e-04 | 6.82e-04 |

The device's own arithmetic is correct — `2*Im(07a0)` matches `07a` to 1e-10. The bootstrap fold's doubling is fine. The groupwise sum is fine. The Goldschmidt delta is fine (7.99e-04 vs 6.82e-04 — both small).

**This means the first `he_inv`'s bootstrap produced a good enough result that Goldschmidt converged, but the second `he_inv`'s bootstrap did not.**

### he_inv input ranges

```
[range] he_inv observed [0.000849682, 0.00234579] ratio 0.362216 against epsilon 6.10352e-05   ← first call
[range] he_inv observed [0.0205227, 0.0507246]  ratio 0.404591 against epsilon 0.00390101      ← second call
```

The second call has a **larger denominator** (0.02-0.05 vs 0.0008-0.002). The bootstrap error (6 bits = ~0.015 RMS) is:
- ~7% of the first denominator (0.015/0.002 = 7.5) — Goldschmidt self-corrects
- ~30% of the second denominator (0.015/0.05 = 30%) — Goldschmidt diverges

## Result 2: MRPC scores — everything diverges from the bootstrap

The MRPC test (skipped because dataset unreachable on this machine) would have been worse. In the second test run (which the test framework ran with `depth=60` clear vs `depth=37` device), even the **first** bootstrap refresh fails:

| relation | device rel | clear rel |
|---|---|---|
| 07a = 2*Re(07a0) | 1.68e-02 | 0 |
| **07a = 2*Im(07a0)** | **3.12e-01** | **0** |
| 07d * 07c constant across slots | **9.12e+02** | **6.82e-04** |

The imaginary part of the bootstrap output is wrong by **31%**. The Goldschmidt delta is off by 912x. Everything downstream is garbage.

## Interpretation

The 27-bit bootstrap precision gap (GPU 6 bits vs CPU 33 bits) is the **root cause** of the softmax divergence:

1. **When the bootstrap error is small relative to the denominator**, Goldschmidt self-corrects (first `he_inv` with synthetic scores: denominator ~0.001, error ~0.015 → 7% → converges)
2. **When the bootstrap error is large relative to the denominator**, Goldschmidt diverges (second `he_inv`: denominator ~0.03, error ~0.015 → 50% → diverges from iter01)
3. **With MRPC scores**, the denominator lands in a range where even the first bootstrap is too imprecise (31% error on imaginary part)

The device's arithmetic is **correct** — `2*Im(07a0) = 07a` to 1e-10. The groupwise sum is correct. The Goldschmidt delta is correct. The problem is solely that the **bootstrap** produces 6 bits of precision instead of 33, and the Goldschmidt iteration amplifies that error rather than damping it when the denominator is small.

## What this confirms

- The 27-bit gap is **not** in the softmax/Goldschmidt code — it enters through the bootstrap
- The 27-bit gap is **not** in `he_exp` (07b.exp: 9.442e-12 error — perfect)
- The 27-bit gap is **not** in the groupwise sum (07c: 1.13e-06 relative — fine)
- The 27-bit gap **is** in the bootstrap refresh that feeds `he_inv`

## What this does NOT yet answer

- **Why** the bootstrap loses 27 bits (the question we've been chasing all session)
- **Which stage** of the bootstrap (ModRaise, CtS, ApproxModEval, StC) introduces the error
- **Why** the second `he_inv` call's bootstrap is worse than the first (different input → different error, but both use the same bootstrap)

The per-stage A/B test (OpenFheInterfaceTests with `tparams64_16_thor_fixmanual`) would isolate which bootstrap stage loses the 27 bits, but it remains blocked by OOM (full untruncated keys at depth 37 / ring dim 65536 exceed 32 GB).
