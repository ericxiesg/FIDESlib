# Encrypt Is Deterministic; ModRaise Introduces All Non-Determinism

## The result

```
encrypt(b0) on GPU:  0/32768 unstable   slot0=[0.7058 0.7058 0.7058 0.7058]
stage 1 on GPU:  32768/32768 unstable   slot0=[5.7e-05 -0.0053 0.00136 -0.00134]
-> encrypt is deterministic, ModRaise introduces non-determinism
```

`encrypt(b0)` is bit-identical across 4 runs on the GPU. After `bootstrap_stage(ct, 1)`
— which is ModRaise + constant scale + Accumulate — 100% of slots are non-deterministic.

## What this means

The non-determinism is introduced entirely by the ModRaise stage, not by
CoeffsToSlots or any later stage. ModRaise (`Bootstrap.cu:423`) does:

1. `ctxt.rescale()` if NoiseLevel == 2
2. `ctxt.multScalar(adjustmentFactor)` + `ctxt.rescale()` + `ctxt.dropToLevel(0)`
3. `ctxt.c0.INTT(cc.batch, true)`
4. `ctxt.c0.grow(cc.L)` — **grows from 1 limb to 38 limbs**
5. `ctxt.c0.generateSpecialLimbs(false, true)` — **zero_out=false**
6. `ctxt.c0.broadcastLimb0()` — copies limb 0 to all grown limbs
7. `ctxt.c0.NTT(cc.batch, true)`
8. Same for `ctxt.c1`

Then stage 1 also does:
9. `ctxt.multScalar(constantEvalMult, false)`
10. `Accumulate(...)` — hoisted rotations + additions + moddown

The non-determinism is in one of steps 3-10. The most likely candidates:

- **`grow(cc.L)`** (step 4): allocates 37 new limbs. If `cudaMallocAsync` returns
  uninitialized memory and `broadcastLimb0` doesn't cover all of it, leftover
  GPU memory becomes part of the ciphertext.
- **`generateSpecialLimbs(false, true)`** (step 5): `zero_out=false` explicitly
  reuses stale data. We tested `zero_out=true` at the bootstrap exit (fix attempt 2)
  and it didn't help, but that was a different call site — this one in ModRaise
  was never tested.
- **`Accumulate`** (step 10): hoisted rotations with stream synchronization.
  A missed `s.wait()` would race.

## Next step

Test the sub-components of stage 1 in isolation. The cleanest approach is to
add a `stopAfterStage` that stops *inside* ModRaise, before vs after `grow`/
`generateSpecialLimbs`/`broadcastLimb0`/`Accumulate`, and check stability at
each point.

Alternatively, add a `cudaMemsetAsync` after `grow` to zero all limbs before
`broadcastLimb0`, and see if the non-determinism disappears. If it does, `grow`
is returning uninitialized memory. If it doesn't, the bug is in `Accumulate`'s
stream synchronization.

## Test

`python/tests/test_modraise_components.py`. Run with
`PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_modraise_components.py`.
