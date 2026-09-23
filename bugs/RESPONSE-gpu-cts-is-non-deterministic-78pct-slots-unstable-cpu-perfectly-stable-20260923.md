# GPU CoeffsToSlots Is Non-Deterministic: 78.8% of Slots Unstable Across Repeats

## What was asked

The collaborator's `e339ff3` invalidated the CPU comparison (CPU ignored
`stopAfterStage` and ran the full bootstrap) and suggested: "run stage 2 twice on
the same input and diff per slot. Which slots are unstable across repeats is a map
that needs no CPU baseline and no known scale."

## What we found

Ran `bootstrap_stage(encrypt(b0), 2)` 6 times on the **same** input (seed 23,
same engine, no re-seed between runs) on both CPU and GPU.

### CPU: perfectly deterministic

All 6 runs produced bit-identical output. Median spread = 7.19e-11 (floating-point
noise). **0 slots** with CV > 1.0. Slot 0 = 0.705801 in every run.

### GPU: massively non-deterministic

**25809 out of 32768 slots (78.8%)** have CV > 1.0 — more variable across runs
than their own mean. Median spread = 0.0608, same order as the median value 0.0565.
Median CV = 1.115.

**Slot 0 across 6 GPU runs:**
```
[0.017931, 0.017931, -0.0356404, -0.0356404, 7.38595e-05, -0.0356404]
```
Three distinct values. The "7.39e-05" reading from the previous experiment was just
one of several possible outcomes.

**Known worst-5 slots across 6 GPU runs:**
```
slot     0: [0.0179, 0.0179, -0.0356, -0.0356, 7.4e-05, -0.0356]  — 3 values
slot  9891: [-0.0357, 0.0714, 0.0893, -0.0357, 0.1964, 0.0893]     — 4 values
slot 19782: [0.125, 0.0536, 0.0714, 0.0893, 0.0179, 0.0179]       — 4 values
slot 20170: [3.1e-08, -0.0357, 0.0536, 0.0714, 0.0536, 0.0536]    — 4 values
slot 16687: [0.0357, 2.4e-07, -0.0714, 2.4e-07, 2.4e-07, 0.0179]  — 3 values
```

All known slots are unstable. The values jump between different k/56 quantization
levels across runs.

### The top-20 most unstable slots (by CV) jump between ~0 and k/56

```
slot 11820: [-1.0e-07, 0.01786, -1.0e-07, -1.0e-07, -1.0e-07, -1.0e-07]  — 2 values
slot  9643: [-2.5e-07, -2.5e-07, -2.5e-07, -2.5e-07, -0.05357, -2.5e-07] — 2 values
slot 28542: [-0.125, 1.8e-08, 1.8e-08, 1.8e-08, 1.8e-08, 0.01786]        — 3 values
slot 23201: [1.7e-08, 1.7e-08, 1.7e-08, -0.01786, -0.1607, 1.7e-08]      — 3 values
```

These slots alternate between essentially zero (1e-7 to 1e-8) and meaningful k/56
values. The quantization the collaborator identified is real, but the slot-to-value
mapping is random across runs.

## Interpretation

This is not normal CKKS noise (which is deterministic given the same ciphertext).
The GPU CoeffsToSlots is using uninitialized memory or has a race condition in the
linear transform pipeline:

- **78.8% of slots unstable** rules out a single bad diagonal entry
- **CPU is perfectly deterministic** — the bug is GPU-specific
- **Values jump between k/56 levels** — the computation is mostly right but the
  accumulation/rotation is non-deterministic
- **The same slot reads differently on re-run** — this is a race or uninitialized
  read, not a wrong value

This connects to the slot-0 multiply bug: the multiply damage is deterministic
(always slot 0/9891/20170), but CtS is non-deterministic. Either:
1. The non-determinism at CtS is averaged out by later stages, leaving only a
   deterministic structural error, or
2. The multiply bug and the CtS instability are two separate GPU issues

The CoeffsToSlots code (`CoeffsToSlots.cu`) does hoisted rotations + a fused
dot-product (`DotProductPtInternal` → `RNSPoly::LTdotProductPtBatch`) with stream
synchronization. A missed synchronization or an uninitialized buffer in that path
would produce exactly this pattern: most values land near the right k/56 level but
some slots pick up wrong residues from a previous run or a concurrent kernel.

## Next step

The non-determinism is the strongest signal yet. The next step is to look at
`LTdotProductPtBatch` and the stream synchronization in `LinearTransform.cu:130-160`
— those `results[0]->c0.GPU[j].s.wait(...)` calls are the synchronization points,
and a missed wait or an unwaited stream would produce non-deterministic output.

## Test

`python/tests/test_stage2_stability.py`. Run with
`PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_stage2_stability.py`.
Takes ~3 minutes (6 bootstraps on CPU + 6 on GPU).
