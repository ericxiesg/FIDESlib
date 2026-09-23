# Non-Determinism Starts at Stage 1 (ModRaise), Not Stage 2 (CoeffsToSlots)

## What was found

Ran `bootstrap_stage(encrypt(b0), stage)` 4 times on the same input for each of
the 4 stages, on both CPU and GPU.

### CPU: all 4 stages perfectly stable

```
stage 1 ModRaise     : 0/32768 unstable   slot0=[0.7058 0.7058 0.7058 0.7058]
stage 2 CoeffsToSlots: 0/32768 unstable   slot0=[0.7058 0.7058 0.7058 0.7058]
stage 3 EvalMod      : 0/32768 unstable   slot0=[0.7058 0.7058 0.7058 0.7058]
stage 4 SlotsToCoeffs: 0/32768 unstable   slot0=[0.7058 0.7058 0.7058 0.7058]
```

### GPU: non-determinism starts at stage 1 and decreases through stages

```
stage 1 ModRaise     : 32759/32768 unstable (99.97%)  slot0=[-0.0068 -0.0219 0.0179 0.0216]
stage 2 CoeffsToSlots: 32747/32768 unstable (99.94%)  slot0=[0.0536 0.0358 -0.1071 0.0358]
stage 3 EvalMod      : 14632/32768 unstable (44.65%)  slot0=[0.1323 0.1323 0.1323 0.1324]
stage 4 SlotsToCoeffs:  1670/32768 unstable  (5.10%)  slot0=[0.1784 0.1740 0.1737 0.1760]
```

The decreasing instability is expected: each stage applies more computation
(NTTs, key switches, Chebyshev polynomials) which smooths out the randomness,
and the final output is mostly stable (5% unstable). But the **source** is
stage 1.

## What this means

The previous report (`db84cca`) localized the non-determinism to stage 2
(CoeffsToSlots), but that was because we only tested stage 2. The per-stage
sweep shows it starts at **stage 1 (ModRaise)** — the very first operation.

Stage 1 is `ModRaise` in `Bootstrap.cu:226`, which calls:
1. `ctxt.multScalar(constantEvalMult, false)` — scale by a constant
2. `ctxt.keySwitch(btoa)` — key switch (only if `sparse_encaps`)
3. `Accumulate(...)` — hoisted rotations + additions + moddown

The `ModRaise` function (`Bootstrap.cu:423`) itself does:
1. `ctxt.rescale()` if NoiseLevel == 2
2. `ctxt.multScalar(adjustmentFactor)` + `ctxt.rescale()` + `ctxt.dropToLevel(0)`
3. `ctxt.c0.INTT()` + `ctxt.c0.grow(cc.L)` + `generateSpecialLimbs(false, true)`
4. `ctxt.c0.broadcastLimb0()` + `ctxt.c0.NTT()`
5. Same for `ctxt.c1`

The non-determinism is in one of these operations. The most likely candidates:
- **`grow(cc.L)`** — allocates new limbs and may not zero them
- **`generateSpecialLimbs(false, true)`** — `zero_out=false` means it reuses
  stale data (we tested `zero_out=true` in a fix attempt and it didn't help,
  but that was at the bootstrap exit, not here)
- **`broadcastLimb0()`** — copies limb 0 to all limbs, could have a race
- **`Accumulate`** — hoisted rotations with stream synchronization

## Connection to the slot-0 bug

The slot-0 corruption at the multiply is **deterministic** (always slot 0/9891/
20170), but the CtS non-determinism is **random**. These may be two separate
issues:

1. The non-determinism at ModRaise adds noise to every slot, but the Chebyshev
   compression in EvalMod smooths most of it out by stage 4.
2. The deterministic slot-0 corruption is a separate structural error that
   survives the smoothing.

Or they may be connected: if ModRaise's non-determinism leaves some state
uninitialized, and that state is read deterministically at a specific position
during the later multiply, it could produce a deterministic slot-0 error from
a non-deterministic source.

## Next step

Test stability of the components of stage 1 individually:
1. Is `encrypt(b0)` itself deterministic? (It should be — no randomness in CKKS
   encryption without a random nonce)
2. Is `multScalar` deterministic?
3. Is `grow` deterministic?
4. Is `Accumulate` deterministic?

The answer narrows the bug to a specific kernel.

## Test

`python/tests/test_stage_stability.py`. Run with
`PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_stage_stability.py`.
Takes ~5 minutes (4 stages × 4 runs on CPU + GPU).
