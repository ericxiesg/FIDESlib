# Stage 06 OOM persists — slab reclamation ran but found nothing to free

Date: 2026-09-09. GPU: Quadro GV100 32 GiB. Commit: e8e29d0.

## Result

Both fixes are in effect:
- **Slab reclamation**: ran (error message changed to "wholly free slabs were
  already returned to the driver before this failed")
- **Lazy alignment**: in effect (line numbers match: `attention.py:337/313`)
- **`--light-plaintext-cache 4`**: in effect

But stage 06 still OOMs at `EvalMultLightPt` — same 512 KiB slab, same 16 MiB free.

## Key observations

1. **No "memory pool returned" message** — the reclamation code executed but
   found zero wholly-free slabs. All 3.7 GiB headroom is genuinely in use by
   active allocations from other size classes.

2. **Lazy alignment helped**: the previous crash was at 13 min CPU; this run
   reached 85 min CPU before crashing. The peak was reduced, but the
   `pieces` array in `_accumulate_product` still accumulates enough ciphertexts
   to exhaust the pool.

3. **Same location**: `_accumulate_product` → `multiply` → `EvalMultLightPt`.
   The accumulation loop builds up results across iterations, and at some
   point a 512 KiB allocation can't be satisfied.

4. **Compile fix**: `CudaUtils.cu` had a missing `extern FIDESlib::Stream s[]`
  forward declaration in `ReclaimFreeSlabs`. Fixed locally.

## What's left to try

From the remote's response, these were listed as "not yet done":
1. **Auxiliary poly pool trimming** (`trimAuxilarPoly`) — with slab reclamation
   now in place, draining the poly pool at stage boundaries could free whole
   slabs. This is the most promising remaining option.
2. **`levelBudget={4,4}`** — plaintexts drop from 7.6 to ~6.9 GiB, but depth
   must increase to 38 (bootstrap depth 18), which means even more key memory.

## Memory breakdown at crash

| Component | MiB |
|---|---:|
| Bootstrap keys | 9,336 |
| Bootstrap plaintexts | 7,560 |
| Rotation keys (15) | ~2,500 |
| Everything else | ~5,800 |
| **Total resident** | **~25,200** |
| **Free** | **~3,700** |
| Free at crash | **16** |

The 3.7 GiB headroom was consumed by stage 06's working set. The accumulation
loop creates ~128 ciphertext results plus intermediate light plaintext
expansions, each needing ~38 MiB at depth=37.
