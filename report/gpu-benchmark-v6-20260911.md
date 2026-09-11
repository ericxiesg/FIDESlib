# GPU Benchmark v6 — Full Layer Completion, Fidelity Investigation

**Date**: 2026-09-11
**Commit**: `7877aa6` (includes `dc2ef1d` memory leak fix, upstream LimbPartition/rotate_hoisted fixes, Python subtract/add numpy fixes)
**Hardware**: Quadro GV100 32 GiB

## Configuration

| Parameter | Value |
|-----------|-------|
| depth | 37 |
| dnum | 4 |
| bootstrap level budget | (3,3) |
| bootstrap depth | 17 (measured, includes 1 alignment level) |
| binary rotations | yes (15 keys) |
| refresh-after-dense | yes (stage 10 → bootstrap → stage 11) |
| layers | 1 |
| samples | 4 |
| --per-stage | no (causes OOM with trace mode) |

## What Changed Since v4

v4 (commit `7ac1f2d`) crashed at stage 11 `he_invsqrt` with `TypeError: EvalSub() received numpy array`.
Three fixes landed for v6:

1. **Python subtract/add fix** (`7877aa6`): `subtract(numpy_array, ciphertext)` now encodes the array
   as plaintext and computes `pt - y` via `EvalNegate(y) + EvalAddPt(neg_y, pt)`. `add()` gained the
   symmetric `x-is-scalar` and `x-is-numpy` branches.

2. **C++ memory leak fix** (`dc2ef1d`): "the dangling-reference fix was leaking every dropped context."
   Every freed ciphertext was leaking its context (polynomials + GPU memory). This is likely why v4
   took ~150 min for stages 01-10 and suffered chronic OOM — the leaked memory filled the card.

3. **C++ upstream fixes** (`fa97286`): `LimbPartition::multPt` now uses the top limb of the current
   level, not the physical top of storage. `rotate_hoisted` copies all metadata to results in the
   non-fused path.

## Results

### Timing

| Phase | Time | % |
|-------|------|---|
| layer 0 (encrypted) | 1413.19s | 98.3% |
| encode weights | 12.90s | 0.9% |
| encrypt | 5.70s | 0.4% |
| decrypt | 2.40s | 0.2% |
| **total** | **1437.73s** | |
| per sample | 359.43s | |

~6 min/sample, ~24 min total. Compare v4: ~150 min for stages 01-10 alone (then crashed).
The memory leak fix eliminated the OOM-related slowdown.

### Accuracy

| Model | Accuracy | F1 |
|-------|----------|----|
| plaintext | 100.00% | 100.00% |
| encrypted | 50.00% | 66.67% |

### Fidelity (against plaintext model)

| Metric | Value |
|--------|-------|
| hidden after layer 0 MAE | 4.199e-01 |
| hidden after layer 0 RMSE | 5.600e-01 |
| hidden after layer 0 max | 9.755e+00 |
| hidden after layer 0 relRMSE | 1.000e+00 |
| best-fit scale | 0.0000 |
| logits MAE | 1.616e+00 |
| logits RMSE | 1.705e+00 |
| logits relRMSE | 9.409e-01 |
| probabilities L1 mean | 8.178e-01 |
| label agreement | 50.00% |

## Analysis

**The benchmark completed** — all 15+ stages of one BERT encoder layer ran end-to-end on GPU
without crashes. This is the first full completion.

**Fidelity is poor**: `relRMSE 1.0` means the error is as large as the signal. `best-fit scale 0.0`
means the encrypted output is uncorrelated with the plaintext. The accuracy dropped to 50% (chance
for 2 classes).

The poor fidelity was not visible in v4 because v4 crashed at stage 11 before producing results.
The 148 passing pytest tests (including GPU tests) suggest the individual operations are functionally
correct, but the full pipeline may expose a correctness issue in one of the three C++ changes:

- **LimbPartition::multPt**: "use the top limb of the current level, not the physical top of storage"
  — could change plaintext-ciphertext multiplication results at specific levels.
- **rotate_hoisted metadata**: could affect rotation results if metadata (level, scale) is wrong.
- **Memory leak fix**: if leaked contexts were being reused with stale data, freeing them properly
  could expose a use-after-free or zeroed-memory issue.

## Next Steps

1. Run with `--per-stage` on CPU (`--engine clear`) to verify the Python pipeline is correct.
2. Revert the C++ changes one by one to isolate which fix introduced the regression:
   - Revert to `7ac1f2d` + Python fix only (no C++ changes) — but this will OOM without the leak fix
   - Revert only LimbPartition::multPt (keep leak fix + rotate_hoisted)
   - Revert only rotate_hoisted (keep leak fix + LimbPartition)
3. Run the GPU fidelity test (`test_gpu_fidelity.py`) to check individual operations.
4. Try `--per-stage` with `--device-memory` now that the leak fix frees memory — the `device_memory()`
   method is now implemented and trace mode might fit.
