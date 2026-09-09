# CUDA illegal memory access located: LTdotProductPtBatch in EvalCoeffsToSlots

Date: 2026-09-09. GPU: Quadro GV100 32 GiB. Commit: be7acf5.

## Precise crash location (CUDA_LAUNCH_BLOCKING=1)

With `CUDA_LAUNCH_BLOCKING=1`, the async error is pinned to its true source:

```
EvalBootstrap
  -> Bootstrap
  -> EvalCoeffsToSlots                    (CtS — Coeffs to Slots)
  -> LinearTransform
  -> DotProductPtInternal<Ciphertext*, Plaintext*>
  -> RNSPoly::LTdotProductPtBatch
  -> LimbPartition::LTdotProductPtBatch   ← crash here
  -> Stream::wait
  -> CudaUtils.cu:214: 'an illegal memory access was encountered'
```

**The failing kernel is inside `LimbPartition::LTdotProductPtBatch`**, which is
the batched plaintext-ciphertext dot product used by the bootstrap's CoeffsToSlots
linear transform. This is NOT `multMonomial` (the previous suspected location).

## What changed from previous runs

| Run | Keys | Crash location | Error |
|---|---|---|---|
| depth=51/(4,4), truncated | 74 keys | `Ciphertext::copy → RNSPoly::grow` | OOM |
| depth=34/(3,3), untruncated, commit 3f72760 | 49 keys | `multMonomial → ~LimbPartition → GPUfree` | illegal address |
| depth=34/(3,3), untruncated, commit be7acf5 | 49 keys | `LTdotProductPtBatch → Stream::wait` | illegal address |
| depth=34/(3,3), truncated, commit be7acf5 | 49 keys | `ensureLevel: key truncated to 28, used at 33` | key level error |
| depth=34/(3,3), --allow-key-grow, commit be7acf5 | 49 keys | `KeySwitchingKey.cu:87` (during key reload) | illegal address |
| compute-sanitizer (memcheck) | 49 keys | `GPUmalloc: out of memory` | sanitizer overhead |

## Analysis

The `LTdotProductPtBatch` kernel is the bootstrap's linear transform (CoeffsToSlots).
It performs a batched dot product of plaintext diagonals with ciphertext limbs.
The illegal memory access likely comes from:

1. **Limb index out of bounds**: The kernel indexes `limb[i]` where `i` ranges
   over the limb count derived from level, similar to the `multMonomial` issue
   that was bounded-checked in this commit. But the bounds check was only added
   to `multMonomial`, not to `LTdotProductPtBatch`.

2. **Plaintext-ciphertext level mismatch**: The CtS linear transform multiplies
   precomputed plaintexts by the ciphertext. If the plaintext was encoded at a
   different level than the ciphertext's current level, the limb counts won't
   match, and the kernel will index past the end of one of the limb arrays.

3. **Auxiliary poly pool corruption**: `LTdotProductPtBatch` creates temporary
   `RNSPoly` objects from the aux pool. If a pooled poly has a stale `level`
   field (as hypothesized in the multMonomial case), the kernel will use wrong
   limb counts.

## Key level plan issue (separate bug)

The `--allow-key-grow` run shows the per-layer key plan is still wrong:
- 15 of 49 keys needed to be reloaded (grown) at runtime
- Keys truncated to level 28-29 were used at level 31-34
- The `std::map::emplace` dedup fix (keeping the higher-level key) did not
  resolve this — the keys are still being truncated too aggressively

The key memory report says "26 truncated (4072 MiB)" — but many of those
truncated keys are needed at higher levels by the bootstrap. The per-layer
plan in `GetBootstrapKeyLevelPlan` is underestimating the required levels.

## What the remote should do

1. **Add bounds checks to `LTdotProductPtBatch`** — same pattern as the
   `multMonomial` fix: verify `limb_size <= g.limb.size()` before the kernel
   loops. This will turn the illegal address into a diagnostic message.

2. **Fix the key level plan** — the per-layer bootstrap key plan is truncating
   keys too aggressively. With `--allow-key-grow`, 15 keys needed reloading,
   meaning the plan underestimated their required level. The plan should
   account for the ciphertext's level at each CtS/StC layer.

3. **Try `compute-sanitizer --tool memcheck` with reduced memory** — the
   sanitizer needs ~2 GiB extra GPU memory. Running with `--light-plaintext-cache 0`
   or reducing depth might free enough. Alternatively, `compute-sanitizer
   --tool memcheck --force-blocking-launches` with the regular configuration.
