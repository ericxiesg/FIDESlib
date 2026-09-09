# Bootstrap works! Level budget off-by-one in stage 10 after alignment fix

Date: 2026-09-09. GPU: Quadro GV100 32 GiB. Commit: 176e466.

## Bootstrap is FIXED — no crash, ran to completion

The level alignment fix in `CoeffsToSlots.cu` works. The bootstrap completed
without any illegal memory access. The benchmark ran through stage 07 (softmax
bootstrap), stage 08, stage 09, and into stage 10 before hitting a level
budget issue.

**This is the first time a GPU bootstrap has completed in this project.**

## Bootstrap diagonals log line (requested by remote)

```
[FIDESlib] bootstrap diagonals: CtS layer 0 holds 34 limbs; a ciphertext at L=34 has 35 (scaling technique 1)
```

Confirms: off-by-one, FIXEDMANUAL (scaling technique 1), exactly as predicted.

## The new error: stage 10 `EvalLevelReduce would drop every RNS limb`

```
layer.forward → stage_10_attention_dense
  → level_down(wx[index], 1)
  → EvalLevelReduce(x, 1)
  → RuntimeError: EvalLevelReduce would drop every RNS limb
```

### Cause

The `dropToLevel` in `EvalCoeffsToSlots` (the alignment fix) consumes one
extra level from the ciphertext. The bootstrap output is now at a lower level
than before. When stage 10 calls `level_down(x, 1)` to reduce by 1 more level,
it tries to drop below 0 — "every RNS limb" gone.

### Evidence

The `test_stage4_bootstrap.py` failure confirms this:
```
assert e.level(out) == 10  →  assert 9 == 10
```
The bootstrap output is 1 level lower than expected. The alignment `dropToLevel`
cost one level.

### Impact on level budget

With depth=34 and bootstrap depth=16 (levelBudget 3,3), the post-bootstrap
level was 18 (34-16). After the alignment fix, it's 17. The THOR layer
needs all 17 remaining levels for stages 08-16 (especially GELU = 13 levels).
Losing one more level at stage 10's `level_down` pushes it below 0.

## pytest results

- 102 passed, 1 skipped
- `test_stage4_bootstrap::test_bootstrap_complex_and_keep_levels[cuda:0]`:
  **FAILED** — `assert 9 == 10` (bootstrap output level is 1 lower than expected)
- Bootstrap precision: max err 1.74e-5 — **good** (the fix doesn't hurt precision)

## Two options for the remote

1. **Adjust `keep_levels` / post-bootstrap level expectation**: The alignment
   drop is correct and necessary. The Python side should account for the extra
   level consumed. `test_stage4_bootstrap` should expect `keep_levels - 1`.

2. **Compensate in `EvalCoeffsToSlots`**: After the linear transform, grow
   the ciphertext back by one level (or skip the drop on the last step).
   This would preserve the original level budget but may not be semantically
   correct.

3. **Increase depth by 1**: depth=35 instead of 34. But 35+12=47 > MAXP=64
   is fine (47 ≤ 64). Budget: 28.0 GiB, headroom 4.0 GiB. This gives one
   extra level to absorb the alignment cost.

## Key metrics from this run

- **First successful GPU bootstrap**: ~120 min CPU for stages 01-10
- Key memory: 8536 MiB, 16 truncated, **0 grown at runtime**
- GPU resident: 23,172 MiB, 5.0 GiB headroom
- No CUDA errors, no illegal memory access
