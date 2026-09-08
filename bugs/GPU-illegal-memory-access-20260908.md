# CUDA illegal memory access in GPUfree during bootstrap multMonomial

Date: 2026-09-08. GPU: Quadro GV100 32 GiB. Commit: 3f72760 (slab allocator fix).

## Summary

depth=34/dnum=4/(3,3)/--refresh-after-dense/--binary-rotations with **untruncated
rotation keys** gets past the key level error and runs the first bootstrap, but
crashes during `Ciphertext::multMonomial` inside `EvalBootstrap` with
`cudaErrorIllegalAddress` in `GPUfree` (CudaUtils.cu:512).

This is a **new** failure mode — not OOM, not key level mismatch, not MAXP overflow.
It is likely a bug in the adaptive slab allocator introduced in commit 3f72760.

## Configuration

```
depth=34, dnum=4, levelBudget=(3,3), binary_rotations=True
refresh_after_dense=True (4 extra bootstraps, 22 total)
rotation keys: 49 keys, all at level 34 (untruncated), resident 8600 MiB
bootstrap plaintexts: 378, 6804 MiB
total GPU resident: 23,172 MiB (4.6 GiB headroom)
```

## Crash stack

```
EvalBootstrap
  -> Bootstrap
  -> Ciphertext::multMonomial
  -> ~LimbPartition (destructor)
  -> GPUfree
  -> Cuda failure CudaUtils.cu:512: 'an illegal memory access was encountered'
```

The crash happens when the `LimbPartition` destructor calls `GPUfree` to free
temporary limb data created during `multMonomial` (a rotation by a monomial,
used internally by the bootstrap).

## Analysis

The slab allocator fix in 3f72760 added adaptive slab sizing: when a full 1 GiB
slab can't be allocated, it halves the slab size and retries. But `GPUfree` puts
blocks back into the per-size-class free list without tracking which slab they
came from. If a size class has blocks from multiple slab sizes (e.g., some from
a 1 GiB slab and some from a 512 MiB slab), freeing a block might write past the
slab boundary or access a corrupted free list entry.

The crash at `GPUfree` (not `GPUmalloc`) suggests a **use-after-free or slab
corruption**: a previously freed block's metadata is invalid, and the free list
traversal hits an illegal address.

### Why it didn't crash before

Previous runs (depth=51/(4,4)) crashed at `GPUmalloc` (OOM) before any bootstrap
ran. The slab fix made it past `GPUmalloc`, but the first bootstrap triggers
many `multMonomial` calls with temporary limb allocations/deallocations,
exercising the `GPUfree` path heavily for the first time.

### Why untruncated keys

The original plan truncated 15 of 49 rotation keys to lower levels, which caused
`KeySwitchingKey::ensureLevel` to reject a key used at level 33 (truncated to 28).
Setting all keys to level 34 (untruncated) fixes the level mismatch at a cost of
only 356 MiB (8600 vs 8244 MiB). The `FIDESLIB_KEY_GROW=1` path also works but is
extremely slow (82+ minutes CPU, did not finish).

## What works

- Budget prediction: 27.4 GiB total, 4.6 GiB headroom — **correct**
- Keygen: succeeds, 8600 MiB keys, 0 truncated, 0 grown
- Engine initialization: succeeds, 23,172 MiB resident
- Stage 01-06: completes (gets to "layer 0 softmax window: THOR's table")
- First bootstrap starts: `EvalBootstrap` is called

## What fails

- Inside the first bootstrap, `multMonomial` creates temporary `LimbPartition`
  data, and the destructor's `GPUfree` hits an illegal memory address

## Questions for the C++ side

1. **Does `GPUfree` handle blocks from different slab sizes correctly?** The
   adaptive slab fix means a single size class can have blocks from slabs of
   different sizes (1 GiB, 512 MiB, 256 MiB, ...). If `GPUfree` assumes all
   blocks in a size class are from the same slab size, it could calculate
   incorrect offsets.
2. **Is there a race condition in the free list?** `multMonomial` might be
   called from multiple threads (OpenMP), and the free list might not be
   thread-safe.
3. **Does the slab base address tracking work with adaptive sizing?** The
   original code allocated 1 GiB slabs. With adaptive sizing, smaller slabs
   have different base addresses, and the slab-to-block mapping might break.

## Recommendation

1. **Fix the `GPUfree` slab corruption** — this is the blocking issue for the
   GPU benchmark. The adaptive slab sizing in `GPUmalloc` needs a matching fix
   in `GPUfree` to track which slab each block belongs to.
2. **Alternative: revert the adaptive slab sizing** and instead implement slab
   return to driver (the option 2 from RESPONSE-gpu-runtime-grow-20260908.md).
   This avoids the mixed-slab-size issue entirely.
3. **The Python-side `plan_rotation_keys` bug** (truncating bootstrap rotation
   keys too aggressively) should also be fixed — the plan should not truncate
   keys that are used by the bootstrap at the ciphertext's current level. The
   workaround of setting all keys to `depth` works but wastes 356 MiB.
