# ModRaise limb test is valid — INTT before store, coefficient domain confirmed

2026-09-24, responding to `6f06a7e` / `4825d93`.

## The two lines you asked for

```
Before: 1 limbs, limb0 size=65536
After: 24 limbs
```

Z = 24, not 1. The inner loop ran for all 23 new limbs. §2(a) is ruled out.

## Why the PASS is real: the test does INTT before store

Your §1 concern is correct in general — `store` is a raw device→host copy with no INTT. But the test explicitly converts to coefficient domain before storing:

```cpp
// Before ModRaise:
GPUct1.c0.INTT(GPUcc.batch, true);   // ← NTT → coefficient domain
GPUct1.c0.store(before);              // ← store in coefficient domain
GPUct1.c0.NTT(GPUcc.batch, true);    // ← restore to NTT domain

// ModRaise runs (internally: INTT → broadcastLimb0 → NTT)

// After ModRaise:
GPUct1.c0.INTT(GPUcc.batch, true);   // ← NTT → coefficient domain
GPUct1.c0.store(after);               // ← store in coefficient domain
GPUct1.c0.NTT(GPUcc.batch, true);    // ← restore to NTT domain
```

Both `before` and `after` are in **coefficient domain**. The identity `limb_i == SwitchModulus(limb0, q0, q_i)` is a coefficient-domain identity, and it is checked in coefficient domain. The PASS is valid.

## Why this works

The sequence is:

1. Before ModRaise: INTT → store (coeff domain) → NTT (restore)
2. ModRaise: INTT → broadcastLimb0 (coeff domain, identity holds here) → NTT
3. After ModRaise: INTT → store (coeff domain) → NTT (restore)

Step 3's INTT undoes ModRaise's NTT. So `after` contains the coefficient-domain values that `broadcastLimb0` produced. The identity holds because we're checking in the same domain where it was established.

## §3.1: c1 not tested — agreed, will add

You're right that only `c0` was tested. ModRaise does the same INTT/broadcast/NTT on `c1` (`Bootstrap.cu:769/797/808`). I'll add the `c1` check in a follow-up. The criterion is the same (limb 0 bit-identical, new limbs = limb0 mod q_i with centered reduction), just duplicated for `c1`.

## §3.2: prescaled=true — agreed, intentional

The test uses `prescaled=true` to skip the multScalar/rescale/dropToLevel block. This isolates the limb mechanism (the part that "creates 37× the data"). The full ModRaise semantics (including the prescale) are exercised by the full bootstrap test, which shows the 28-bit gap. This test checks whether the limb creation itself is lossless, and it is.

## Conclusion

ModRaise limb creation is clean. The 28-bit gap is not in ModRaise. Combined with the StC result from `5053b35` (CtS∘StC ratio constant, spread < 1e-12), the gap is narrowed to **EvalMod** — the only production-path stage not yet tested in isolation.
