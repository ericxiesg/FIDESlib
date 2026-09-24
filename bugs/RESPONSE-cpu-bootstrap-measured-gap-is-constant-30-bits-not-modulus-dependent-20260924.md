# CPU bootstrap measured: gap is constant ~30 bits, not modulus-dependent

**Date:** 2026-09-24
**Collaborator commit:** `d677347` (normalize by q0/Δ)

## The number that was missing

CPU-only OpenFHE bootstrap, 256 slots, depth 37, ring dim 65536, FIXEDMANUAL, SPARSE_TERNARY, dnum=4:

| Config | RMS | Raw bits | Normalized bits (q0/Δ) |
|---|---|---|---|
| CPU mod50/55 | 1.38e-11 | 36.1 | **41.1** |
| CPU mod59/60 | 2.51e-14 | 45.2 | **46.2** |

## The full picture (CPU + GPU, normalized)

| Config | q0/Δ | GPU raw | GPU norm | CPU raw | CPU norm | **Gap (norm)** |
|---|---|---|---|---|---|---|
| mod50/55 | 32 | 6.0 | 11.0 | 36.1 | 41.1 | **30.1** |
| mod59/60 | 2 | 14.0 | 15.0 | 45.2 | 46.2 | **31.2** |

## Conclusion

**The gap is constant at ~30 bits, not modulus-dependent.**

- CPU normalized: 41.1 (mod50/55) → 46.2 (mod59/60): **5.1 bit spread**
- GPU normalized: 11.0 (mod50/55) → 15.0 (mod59/60): **4.0 bit spread**
- Gap: 30.1 (mod50/55) → 31.2 (mod59/60): **roughly constant**

The collaborator's two hypotheses from `d677347`:
> - CPU 在 mod59/60 归一化后仍是 ~38 → 缺口从 27 收到 23，GPU 有轻微的模数依赖
> - CPU 在 mod59/60 归一化后更高 → 缺口恒定，模数依赖完全是正常行为，只需查那个常数缺口

**Answer: the second one.** CPU normalized precision does rise with modulus size (41→46), but the GPU rises by almost the same amount (11→15), so the gap stays at ~30 bits. The modulus dependence is mostly normal CKKS behavior (as the collaborator predicted). What remains is a **flat ~30-bit structural deficit**.

## What this means

The ~30-bit gap is:
- **Not** precision loss (GPU arithmetic is exact integers, `59f4d6c`)
- **Not** modulus-dependent (this report)
- **Not** depth-dependent (`82e9514`)
- **Not** in ApproxModEval alone (`9e1ae3a` — both moduli give ~7 bits)
- **Not** in the sparse branch (`144e5de` — THOR uses dense path)
- **Not** in correction factor (`cc698cf` — CPU is 33 bits at every factor)
- **Not** nondeterminism (`test_stage_stability_one_ciphertext.py` — bit-identical)

It is a **flat, deterministic, structural ~30-bit deficit** that appears somewhere in the full bootstrap pipeline but not in the individual ApproxModEval stage. The most likely location is:
1. **CoeffsToSlots / SlotsToCoeffs** (LinearTransform) — but those fail at both moduli, so it's hard to isolate
2. **ModRaise** — not tested separately
3. **The interaction between stages** — individual stages pass, but something in the composition loses 30 bits

## Per-stage A/B status

The `tparams64_13_4_thormod` vs `tparams64_13_4_sparse` A/B ran:
- **ApproxModEval**: identical at both moduli (~7 bits), both PASS
- **CoeffsToSlots**: both FAIL (pre-existing LinearTransform issue)
- **SlotsToCoeffs**: crashed with "Modulus mismatch" exception

The crash prevents completing the A/B. The CoeffsToSlots/LinearTransform failure is a separate bug that needs fixing before the per-stage comparison can isolate the 30-bit gap.
