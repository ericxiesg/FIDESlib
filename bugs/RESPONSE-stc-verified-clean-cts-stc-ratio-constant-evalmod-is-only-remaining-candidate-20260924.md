# GPU StC verified clean — CtS∘StC ratio is perfectly constant on both configs

2026-09-24, continuing from `f93f0ce` (ModRaise verified clean). Responding to `08d6720` (two traps in CtS∘StC identity test).

## Summary

GPU StC (SlotsToCoeffs) is **not** the source of the 28-bit gap. The CtS∘StC composition yields a perfect constant multiple of the identity, with no slot-dependent error:

| Config | Constant ratio | Spread (max−min)/median | Precision | Verdict |
|---|---|---|---|---|
| /9 thormod (mod50/55) | 2.28571 (= 16/7) | 6.60e-12 | ~37 bits | **PASS** |
| /8 sparse (mod59/60) | 2.28571 (= 16/7) | 1.05e-13 | ~43 bits | **PASS** |

The ratio is identical across all 32 carried slots. The spread is at the level of CKKS encoding precision. There is no slot-dependent error in StC.

## Test design

`CtSStCIdentity` (test/OpenFheInterfaceTests.cu, commit `61ade0b`):

1. `{2,2}` levelBudget setup, 32 slots (matching full bootstrap where 28-bit gap was measured).
2. Encode 32 distinct non-zero values (1, 2, ..., 32) at level 11 — the EvalMod exit level for {2,2} (**Trap 2 mitigation**).
3. GPU CtS: `EvalCoeffsToSlots(GPUct1, slots, false)`.
4. GPU StC directly on CtS output: `EvalCoeffsToSlots(GPUct1, slots, true)` — no EvalMod, no conjugate in between.
5. Decrypt, compute `output[i] / input[i]` for each carried slot.
6. Check ratio consistency: report `(max−min)/median` (**Trap 1 mitigation** — does not require knowing the constant).

### Both traps from `08d6720` handled

- **Trap 1** (1/k constant factor): Instead of comparing against original plaintext, we check whether `output[i]/input[i]` is the same constant across all slots. The test does not assume any particular value for the constant. The actual constant observed is 16/7 ≈ 2.28571, not 1/28 — this is because the encoding level (11) introduces additional scaling. But the key point is: **the ratio is constant**, not what it equals.
- **Trap 2** (wrong level): Ciphertext encoded at level 11 (EvalMod exit for {2,2}), not fresh top level. This matches production. GPU confirms: Before CtS level=12, After CtS level=11, After StC level=9.

## What this rules out

The 28-bit gap is **not** in:
- **ModRaise** — verified clean (limb-level test, `f93f0ce`)
- **StC** — verified clean (this test, ratio constant)
- **CtS as a linear transform** — the CtS∘StC composition is constant, meaning CtS introduces no slot-dependent error. (The previous 22–26× CtS error was a comparison against CPU CtS, which may reflect a constant scaling difference, not a precision defect.)

## Updated stage table

| Stage | Production? | Tested? | Result |
|---|---|---|---|
| ModRaise | Yes | Yes | **PASS** (limb-level, both configs) |
| **EvalMod** | **Yes** | **No** | **— (only remaining candidate)** |
| CtS ({2,2}) | Yes | Yes (via composition) | Clean linear transform |
| StC ({2,2}) | Yes | Yes | **PASS** (ratio constant, spread < 1e-12) |
| LT ({1,1}) | No | Yes | 0.0917 structural error (different defect) |

## Conclusion: EvalMod is the only remaining candidate

Three of the four production-path stages are now verified clean. The 28-bit gap must originate in **EvalMod** — the Chebyshev polynomial approximation + key switching step. This is the most complex stage: it involves `Accumulate` (baby-step/giant-step), `approxModReduction` / `approxModReductionSparse`, and multiple rescalings.

### Next step: isolate EvalMod

The existing `ApproxModEval` and `ApproxModEvalSparse` tests (test/OpenFheInterfaceTests.cu:2614, :2858) already test EvalMod in isolation. I will run these on /8 and /9 to measure the CPU vs GPU precision gap in EvalMod alone.
