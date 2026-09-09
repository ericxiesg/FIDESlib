# ROOT CAUSE FOUND: pt[0] has 34 limbs, kernel reads 35 — level mismatch in EvalCoeffsToSlots

Date: 2026-09-09. GPU: Quadro GV100 32 GiB. Commit: e414f3d.

## The diagnostic

The limb-count bounds check in `LTdotProductPtBatch` finally caught the root cause:

```
FIDESlib: LTdotProductPtBatch would read 35 limbs from pt[0], which holds 34.
The operands are at different levels.
```

**The bootstrap's CoeffsToSlots linear transform multiplies a plaintext diagonal
(pt[0]) that has 34 limbs (level 33) against a ciphertext that needs 35 limbs
(level 34).** The kernel indexes `pt_partition[blockIdx.y]` with `blockIdx.y`
ranging to 35, but `pt[0]` only has 34 limb entries — off-by-one → illegal
memory access.

## Configuration

```
depth=34, dnum=4, levelBudget=(3,3), binary_rotations=True
refresh_after_dense=True
49 rotation keys, 16 truncated, 0 grown at runtime
resident 8536 MiB (would be 8600 MiB untruncated)
```

The "0 grown at runtime" confirms the key level plan is now correct — the
`AddRotationKeys` fix works.

## Timeline

The run survived ~100 minutes of CPU time (22 bootstraps) before hitting this
in a late CtS layer. Previous runs crashed at the same point but the error was
asynchronous and unattributable.

## Root cause analysis

In `EvalCoeffsToSlots` → `LinearTransform` → `DotProductPtInternal`:

1. `out[0]->getLimbSize(*out[0]->level)` computes `limbsize = 35` from the
   ciphertext's current level (34, which means 35 limbs including the special
   modulus limb)
2. `grid.y = limbsize` — the kernel iterates `blockIdx.y` from 0 to 34
3. `pt[0]` was precomputed at level 33 (34 limbs) — one fewer than needed
4. When `blockIdx.y == 34`, `pt_partition[34]` is out of bounds → the kernel
   reads garbage → illegal memory access

### Why the plaintext has one fewer limb

The bootstrap's CtS plaintexts are precomputed at a specific level. The
ciphertext entering the bootstrap is at level 34 (depth-1 = 33 consumed levels
→ level 0 remaining, but ModRaise puts it back to level 34). The CtS
plaintexts should be encoded at the same level, but they're at level 33 —
off by one.

This could be:
- A ModRaise level calculation off-by-one in the bootstrap setup
- The plaintext encoding using `depth-1` instead of `depth` as the target level
- The `limbsize` calculation including a special-prime limb that the plaintext
  doesn't have

## What the remote should do

1. **Fix the level alignment** — the CtS plaintexts need to be at the same
   level as the post-ModRaise ciphertext. Check `EvalBootstrapSetup` and
   `EvalCoeffsToSlots` for the level used to encode the plaintext diagonals.

2. **Alternative: make the kernel tolerant** — if the plaintext has fewer
   limbs, skip the missing ones (treat as zero). This is less correct but
   would unblock testing.

3. **The key level plan is now correct** (0 grown at runtime, 16 truncated) —
   no changes needed there.
