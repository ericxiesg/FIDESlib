# levelBudget=(4,4) measured + runtime grow OOM

Date: 2026-09-08. GPU: Quadro GV100 32 GiB.

## Summary

levelBudget=(4,4) with depth=51/dnum=4 **passes the level wall** (ClearEngine
simulation OK, bl=33) and **fits in MAXP** (52+12=64=MAXP exact). Keygen
succeeds with 24,334 MiB resident. But the benchmark crashes at runtime with
CUDA OOM during `CiphertextImpl::copy -> RNSPoly::grow -> generateLimbToLevel
-> GPUmalloc`. This is a **new** failure mode — not keygen OOM, not MAXP
overflow — the budget model has no line item for it.

## Measured data for MEASURED_BOOTSTRAP

```
levelBudget=(4,4), depth=51, dnum=4, log_n=16, binary rotations
  Plaintexts loaded: 248 ~ 6882MB
  Bootstrap key level plan: 21 of 73 keys truncated to level 38
    (L=51, bootstrap depth 18, levelBudget {4,4}, margin 1)
    53 StC indexes, 53 CtS indexes, 122 in total
  Rotation keys loaded: 74 ~ 18648MB (untruncated estimate)
  Key memory resident: 17452 MiB (would be 18900 MiB untruncated)
    36 truncated (7624 MiB saved), 0 grown at runtime
```

So for `MEASURED_BOOTSTRAP`:
- `(4,4)`: 248 plaintexts, 6882 MiB, 74 keys, 17452 MiB resident

## Memory budget at depth=51/dnum=4/(4,4)

| Component | MiB |
|---|---:|
| Bootstrap plaintexts | 6,882 |
| Rotation keys (74, truncated) | 17,452 |
| **Total resident** | **24,334** |
| Free on 32 GiB card | **8,434** |
| Keygen scratch (transient) | ~3,000 |

Keygen succeeds (8,434 MiB > 3,000 MiB scratch).

## Runtime OOM

The benchmark crashes at the very first computation step ("layer 0 softmax
window: THOR's table"):

```
CiphertextImpl::copy (copy constructor)
  -> CryptoContextImpl::CopyDeviceCiphertext
  -> Ciphertext::copy
  -> RNSPoly::copy
  -> RNSPy::grow
  -> LimbPartition::generateLimbToLevel
  -> LimbPartition::generate
  -> GPUmalloc: 'out of memory'
```

### What we tried

| Variant | Result |
|---|---|
| `--light-plaintext-cache 8` (default) | OOM at `RNSPoly::grow` |
| `--light-plaintext-cache 2` | Same OOM (all 248 plaintexts still loaded) |
| `--lenient` | Same OOM |

### Analysis

The crash is in `CiphertextImpl::copy` — a **ciphertext copy** triggers
`RNSPoly::grow`, which calls `LimbPartition::generateLimbToLevel`. This is
not a key-switching operation; it's a copy that needs to grow a ciphertext
to a higher level.

The free GPU memory is 8,434 MiB. A single ciphertext at L=51 with N=32768
is only ~16 MiB (32768 * 63 * 8 bytes). The `LimbPartition::generate`
allocation must be requesting much more than a single ciphertext's worth —
possibly the full decomp/digit structure.

This is the same `LimbPartition::generate` code path as the keygen scratch
OOM, but at **runtime** instead of keygen. The budget model's
`KEYGEN_SCRATCH=3 GiB` only covers keygen-time scratch; runtime grow scratch
is unaccounted.

### Questions for the C++ side

1. **How much does `LimbPartition::generateLimbToLevel` allocate?** The
   `generate` call in the stack allocates `VectorGPU<uint64_t>` — is this
   just the limb data (~16 MiB) or the full decomp matrix?
2. **Why does `Ciphertext::copy` trigger `grow`?** A copy should be at the
   same level. Is the copy constructor trying to align to a target level?
3. **Can the grow scratch be freed after each operation?** If multiple
   grows are in flight simultaneously, the memory adds up.
4. **Would avoiding truncation help?** Untruncated keys (18,900 MiB vs
   17,452 MiB, +1,448 MiB) wouldn't need growing, but `--no-truncate` is
   only on the `budget` subcommand, not `fhe`.

## Parameter space exhausted

| Combo | MAXP | Keygen | Runtime | Level wall |
|---|---|---|---|---|
| depth=50, dnum=4, (4,4) | 51+12=63 OK | OK (24,274 MiB) | **not tested** (level wall fails) | FAIL (bl=32, level -1 at gelu) |
| depth=51, dnum=4, (4,4) | 52+12=64 OK | OK (24,334 MiB) | **OOM at grow** | OK (bl=33) |
| depth=52, dnum=4, (4,4) | 53+12=65 **EXCEEDS MAXP** | — | — | — |
| depth=50, dnum=5, (4,4) | OK | **OOM keygen** | — | — |
| depth=52, dnum=5, (4,4) | OK | **OOM keygen** | — | — |

**depth=51/dnum=4/(4,4) is the only parameter combo that passes both MAXP
and the level wall.** It fails at runtime due to grow OOM. This is a C++
memory management issue, not a parameter tuning issue.

## Recommendations

1. **Fix the runtime grow OOM** — this is the only blocking issue. The
   `LimbPartition::generate` allocation during `Ciphertext::copy` either
   needs to be smaller, pooled, or freed eagerly.
2. **Add `--no-truncate` to the `fhe` subcommand** — untruncated keys
   avoid runtime grows entirely, at the cost of 1,448 MiB more key memory
   (18,900 vs 17,452 MiB). With 8,434 MiB free, this might fit if the grow
   scratch was the problem.
3. **Add runtime grow scratch to the budget model** — currently
   `KEYGEN_SCRATCH=3 GiB` only covers keygen. Runtime needs its own line.
4. **Alternative: algorithm-level intermediate bootstrap** (option 2 from
   RESPONSE-gpu-keygen-20260908.md) — adding a bootstrap between stage_08
   and stage_13 would reduce the 37-level chain, lowering the minimum depth
   and allowing a smaller parameter set.
