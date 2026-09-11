# BUG: Softmax stage (stage 07) produces completely uncorrelated output — root cause of poor GPU benchmark fidelity

## Summary

Per-stage fidelity measurement on GPU (Quadro GV100, depth=37, dnum=4, binary-rotations,
refresh-after-dense, 1 layer, 1 sample) shows that **stages 01–06 (query, key, value, scores)
are perfect** (relRMSE ~3e-7, scale 0.5–2.0), but **stage 07 (softmax) is completely broken**
(scale=0.0000, relRMSE=1.0).  Every downstream stage inherits the garbage, and stages from
`intermediate` onward crash with `SetLevel: multiplicative depth [37] is insufficient`.

This means the poor fidelity reported in v6/v8 benchmarks (accuracy 50%, relRMSE 1.0) is
**not** caused by the `EvalNegate` scale bug or the `subtract(numpy, ct)` fix — those only
affect `he_invsqrt` in LayerNorm, which is downstream of the broken softmax.  The root cause
is in the softmax itself.

## Per-stage fidelity data (v10, commit 5fa4c2a, 2026-09-11)

```
per-stage fidelity, layer 0 (sample 0), each rescaled by its best fit
  query                 scale   2.0000   MAE 1.889e-07  RMSE 2.642e-07  max 2.545e-06  relRMSE 2.884e-07
  value                 scale   2.0000   MAE 1.204e-07  RMSE 1.673e-07  max 1.387e-06  relRMSE 3.141e-07
  scores                scale   0.5000   MAE 3.765e-07  RMSE 5.064e-07  max 3.717e-06  relRMSE 2.873e-07
  softmax               scale   0.0000   MAE 2.273e-02  RMSE 6.097e-02  max 9.442e-01  relRMSE 1.000e+00
  attention_dense       scale   0.0000   MAE 1.969e-01  RMSE 2.638e-01  max 2.596e+00  relRMSE 1.000e+00
  norm_1                scale   0.0000   MAE 7.993e-01  RMSE 1.288e+00  max 6.210e+01  relRMSE 1.000e+00
  intermediate          could not compare: SetLevel: multiplicative depth [37] is insufficient
  gelu                  could not compare: SetLevel: multiplicative depth [37] is insufficient
  output_dense          could not compare: SetLevel: multiplicative depth [37] is insufficient
  norm_2                could not compare: SetLevel: multiplicative depth [37] is insufficient
```

All 15 `[mem]` stage boundaries completed (rotated through norm_2), so the computation
runs end-to-end.  The crash is only in the per-stage decode of later stages and in the
final `decode_six_blocks`.

## What the softmax stage does

`stage_07_softmax` (softmax.py:159) performs:

1. **Merge** pairs of score ciphertexts (real + imaginary into complex)
2. **Bootstrap** each merged ciphertext (`self.bootstrap(merged)`) — this is the softmax's
   own bootstrap, separate from `--refresh-after-dense`
3. **Split** back into real and imaginary parts via add/conjugate/subtract
4. **he_softmax** on the refreshed scores:
   - Normalise by scale (32 or 64)
   - `he_exp` — polynomial approximation of exp()
   - Mask by attention mask
   - Sum over groups
   - `he_inv` — Goldschmidt division for 1/sum
   - Iterate `update_inv_D` to refine
   - `_broadcast_softmax` to spread the result

The scale=0.0000 means the encrypted softmax output has **zero linear correlation** with the
plaintext softmax — this is not a precision issue, it is a complete failure.

## What is NOT the cause

- **EvalNegate scale bug** (fixed in 43e1c56): only affects `he_invsqrt` in LayerNorm,
  which is downstream of the softmax.  v6 (broken) and v8 (fixed) produce identical
  benchmark results, confirming this.
- **subtract(numpy, ct) fix** (113e854): same — only affects `he_invsqrt` in LayerNorm.
- **addPt/subPt scale mismatch** (checked by 4a2023d): the v9 run with the scale mismatch
  check did NOT throw any error, so all addPt/subPt calls have matching NoiseLevels.

## Possible causes to investigate

1. **Softmax bootstrap** — The bootstrap inside `stage_07_softmax` (line 174) might be
   producing garbage at this level/configuration.  The `test_stage4_bootstrap.py` test
   passes, but it may not test the same level/metadata the softmax uses.

2. **he_exp polynomial evaluation** — The exponential approximation might be diverging
   at the score magnitudes used in the benchmark.  The polynomial is evaluated on
   `scores / scale`, and if the scores are larger than expected, the polynomial might
   not converge.

3. **he_inv (Goldschmidt division)** — The division might not be converging.  `he_inv`
   uses `subtract(float, ct)` which calls `EvalScalarSub` — this is a different code
   path from the numpy branch we fixed, and might have its own issues.

4. **Level management after bootstrap** — The `intermediate` stage crashes with
   "multiplicative depth [37] is insufficient", meaning a ciphertext is at level 37.
   The softmax bootstrap or the LayerNorm (norm_1) might be leaving the ciphertext at
   the wrong level.

5. **The "doubled scores" handling** — `stage_07_softmax` notes that "the pack-and-unpack
   around the bootstrap doubles the scores".  If this doubling is not correctly handled
   by `he_softmax`'s parameters, the exponential input would be wrong.

## Reproduction

```bash
export PYTHONPATH=/home/zhiyuan/workspace/THOR-v2/FIDESlib/python:/home/zhiyuan/bench-run
export LD_LIBRARY_PATH=/home/zhiyuan/workspace/THOR-FIDE/openfhe-install/lib:/usr/local/cuda-12.9/lib64
python3 -u -m thorfhe.bench fhe \
    --engine fideslib --device cuda:0 \
    --depth 37 --dnum 4 --bootstrap-level-budget 3,3 \
    --binary-rotations --refresh-after-dense \
    --layers 1 --limit 1 \
    --per-stage --device-memory --offline
```

Log: `/home/zhiyuan/bench-run/gpu-depth37-v10.log`

## Environment

- GPU: Quadro GV100 32 GiB
- FIDESlib commit: 5fa4c2a (includes negation fix 43e1c56, scale mismatch check 4a2023d)
- OpenFHE: FIXEDMANUAL scaling, depth=37, dnum=4, 32768 slots
- Benchmark: 1 BERT layer, 1 sample, MRPC dataset
