# GPU OOM at stage 06 EvalMultLightPt — memory pool fragmentation with depth=37

Date: 2026-09-09. GPU: Quadro GV100 32 GiB. Commit: b72cc34.

## Summary

depth=37/dnum=4/(3,3) passes the level wall (bootstrap completes, stage 10
no longer hits the floor), but crashes at stage 06 `attention_score` with a
**GPU memory pool fragmentation OOM**: 512 KiB slab can't be allocated because
only 16 MiB is free and it's held by other size classes.

**The bootstrap completed successfully** — stages 07-10 ran without any CUDA
errors. The crash is at stage 06, which runs *before* the first bootstrap.
The issue is that depth=37's larger key/plaintext footprint (28.3 GiB vs 27.0
for depth=34) leaves only 3.7 GiB headroom, and the memory pool's
never-return-slab design fragments the remaining space.

## Configuration

```
depth=37, dnum=4, levelBudget=(3,3), binary_rotations, refresh_after_dense
49 rotation keys, 16 truncated, 0 grown at runtime
resident 9336 MiB keys, 7560 MiB plaintexts
total GPU: 25,244 MiB, headroom 3.7 GiB
```

## Crash

```
stage_06_attention_score → _accumulate_product → multiply
  → EvalMultLightPt(x, y)
  → RuntimeError: GPU memory pool could not obtain a slab of 524288 bytes
    for size class 524288 (device 0, 16 of 32491 MiB free).
    Note the pool never returns slabs to the driver and is keyed by exact
    allocation size, so free memory reported by the driver may already be
    held by other size classes.
```

### What happened

1. Keygen + bootstrap precomputation: 25,244 MiB resident, 3.7 GiB free
2. Stage 01-05: run fine (light plaintext multiplications, small allocations)
3. Stage 06 `attention_score`: calls `EvalMultLightPt` which needs a 512 KiB
   slab. The pool has 16 MiB free but it's all in other size classes — the
   512 KiB class has no free blocks and no room for a new slab.

### This is the slab allocator's "never return" design

The memory pool allocates slabs (1 GiB or smaller with adaptive sizing) and
never returns them to the driver. Free blocks go to per-size-class free lists.
When a new size class needs a slab but the free memory is held by other size
classes, the allocation fails — even though `nvidia-smi` reports 16 MiB free.

## What works

- **Bootstrap: COMPLETE** — no CUDA errors, no illegal memory access
- **Stages 01-05: PASS**
- **Key plan: CORRECT** — 16 truncated, 0 grown at runtime
- **Level budget: CORRECT** — depth=37 gives post-bootstrap level 20, enough
  for the full layer (stages 08-16 including GELU)
- **Diagonal log**: `CtS layer 0 holds 37 limbs; a ciphertext at L=37 has 38`

## The problem is purely memory pressure

| depth | Total | Headroom | Bootstrap | Stage 06 | Stage 10 |
|---|---|---|---|---|---|
| 34 | 27.0 GiB | 5.0 GiB | works | works | FAIL (level floor) |
| 37 | 28.3 GiB | 3.7 GiB | works | **OOM** | would work |

depth=34 has enough memory but not enough levels. depth=37 has enough levels
but not enough memory. The window is extremely narrow.

## Options

1. **Slab return to driver** (the long-promised fix) — when a size class's
   free list grows large enough, return a whole slab to `cudaFree`. This would
   free the 16 MiB held by other size classes and let the 512 KiB slab be
   allocated.

2. **Reduce plaintext cache** — `--light-plaintext-cache 4` instead of 8
   might save enough for stage 06 to pass.

3. **Reduce key truncation** — the 16 truncated keys save 2944 MiB. Less
   truncation costs more memory but avoids runtime grows. More truncation
   saves more but risks ensureLevel errors. Current 16 is already optimal.

4. **Depth=35 or 36** — gives post-bootstrap level 18-19, which might be
   enough for the layer if the alignment cost is absorbed. Need to recheck
   the level budget.
