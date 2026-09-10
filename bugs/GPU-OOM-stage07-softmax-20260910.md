# Stage 06 passes! OOM moves to stage 07 softmax EvalRelinearize

Date: 2026-09-10. GPU: Quadro GV100 32 GiB. Commit: 560e54d.

## Progress: stage 06 PASSES

The trim + consume fixes work. The benchmark ran through stage 06 (attention_score)
without OOM — previously crashed at 13 min (v1) or 85 min (v2). Now reaches
stage 07 softmax before OOM.

## New crash: stage 07 `_broadcast_softmax` → `EvalRelinearize`

```
stage_07_softmax → he_softmax → _broadcast_softmax
  → _times → rescale(relinearize(multiply(x, y)))
  → EvalRelinearize(x)
  → RuntimeError: GPU memory pool could not obtain a slab of 524288 bytes
    for size class 524288 (device 0, 16 of 32491 MiB free).
    wholly free slabs were already returned to the driver before this failed.
```

### Location in softmax

`_broadcast_softmax` multiplies `left` (exp_u) by `right` (inv_D) across multiple
broadcast iterations. Each `_times` call does multiply → relinearize → rescale,
creating temporary ciphertexts. The `left` array has ~128 ciphertexts (the
softmax broadcast pattern), and each multiply creates a new ciphertext that
gets relinearized.

### Same 512 KiB slab / 16 MiB free pattern

The trim reduced the aux poly pool enough for stage 06 to pass, but stage 07's
softmax broadcast creates a new working set that exceeds the remaining headroom.

### Timeline

| Version | First OOM stage | CPU time | Fix that got past it |
|---|---|---|---|
| v1 (slab reclamation only) | stage 06 `EvalMultLightPt` | 13 min | — |
| v2 (slab + lazy alignment) | stage 06 `EvalMultLightPt` | 85 min | — |
| **v3 (slab + lazy + trim + consume)** | **stage 07 `EvalRelinearize`** | **~90 min** | **trim + consume** |

### No "memory pool returned" message

The slab reclamation ran ("wholly free slabs were already returned") but found
nothing to free — same as v2. The trim is keeping the aux pool low, but the
softmax's working set is the new bottleneck.

## What's needed

The softmax broadcast (`_broadcast_softmax`) creates ~128 temporary ciphertexts
from `left` × `right` multiplications. Each ciphertext at depth=37 is ~38 MiB.
128 × 38 MiB ≈ 4.9 GiB — more than the 3.7 GiB headroom.

Options:
1. **Apply `consume` to softmax's `left` array** — same pattern as stage 06/08,
   use each element then set to None immediately.
2. **Reduce `--light-plaintext-cache` to 1** — saves ~60 MiB (4 items × 15 MiB).
3. **Try `levelBudget={4,4}`** — plaintexts 7.6→~6.9 GiB, saves 700 MiB, but
   depth must increase to 38 (bootstrap depth 18), key memory increases.
4. **Stream the softmax broadcast** — process elements in smaller batches
   rather than holding all 128 simultaneously.
