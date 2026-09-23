# Found It: broadcastLimb0_ Skips uint32 Primes; New Limbs Stay Uninitialized

## The mechanism

`ModRaise` in `Bootstrap.cu:662` calls:
1. `ctxt.c0.grow(cc.L)` — allocates 37 new limbs via `cudaMallocAsync` (uninitialized)
2. `ctxt.c0.broadcastLimb0()` — supposed to copy limb 0 to all new limbs

But the `broadcastLimb0_` kernel (`ElemenwiseBatchKernels.cu:140-148`) has a type guard:

```cpp
__global__ void broadcastLimb0_(void** a) {
    int idx          = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y + 1;
    if (ISU64(primeid) && ISU64(0)) {   // ← only copies if BOTH are uint64
        uint64_t in = ((uint64_t*)a[0])[idx];
        SwitchModulus(in, 0, primeid);
        ((uint64_t*)a[primeid])[idx] = in;
    }
    // if ISU64(primeid) is false (uint32 prime), this limb is NEVER written
}
```

`ISU64(x)` is `constants.type & (1 << x)` — true when primeid `x` is stored as uint64.

If any prime in the chain is stored as uint32, `broadcastLimb0_` skips that limb
entirely. The newly allocated (uninitialized) memory stays as-is, and its contents
are whatever `cudaMallocAsync` / the memory pool left there.

## Why this matches every observation

- **Non-deterministic**: `cudaMallocAsync` returns whatever the pool has, which
  varies between runs. The memory pool reuses freed blocks from previous
  operations, so the residual data is different each time.
- **100% of slots unstable at stage 1**: every slot depends on all N coefficients,
  and uninitialized limbs contribute random residues to every slot.
- **CPU is perfectly stable**: CPU's `ModRaise` doesn't use `broadcastLimb0_`.
- **Decreasing instability through stages**: later stages (Chebyshev, linear
  transforms) smooth out the noise, but some survives to the final output.
- **Slot 0 deterministic corruption**: if the uninitialized data happens to have
  a consistent pattern at certain positions (due to memory pool allocation order),
  it could produce a deterministic slot-0 error after the multiply compresses it.

## What needs verification

1. **Are any primes uint32 in the depth=37 / first_mod_bits=55 config?**
   - first_mod_bits=55 → primeid 0 is uint64 (55 > 32)
   - scaling_bits=50 → scaling primes are uint64 (50 > 32)
   - But small special primes or auxiliary primes could be uint32

2. **Is `broadcastLimb0_mgpu_` (the multi-GPU version) affected the same way?**
   - Same `ISU64` guard at `ElemenwiseBatchKernels.cu:153`

## The fix

The `broadcastLimb0_` kernel needs a uint32 path, or all primes should be forced
to uint64. The simplest fix is to add a uint32 branch:

```cpp
__global__ void broadcastLimb0_(void** a) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    const int primeid = blockIdx.y + 1;
    if (ISU64(primeid) && ISU64(0)) {
        uint64_t in = ((uint64_t*)a[0])[idx];
        SwitchModulus(in, 0, primeid);
        ((uint64_t*)a[primeid])[idx] = in;
    } else if (!ISU64(primeid) && !ISU64(0)) {
        uint32_t in = ((uint32_t*)a[0])[idx];
        SwitchModulus(in, 0, primeid);
        ((uint32_t*)a[primeid])[idx] = in;
    }
    // mixed u32/u64 would need a conversion path
}
```

Or, more defensively, zero the new limbs after `grow` and before `broadcastLimb0`:

```cpp
// In ModRaise, after grow:
for (auto& lp : ctxt.c0.GPU) {
    for (auto& l : lp.limb) {
        // zero the limb
    }
}
```

## Test

The existing `test_modraise_components.py` confirms stage 1 is 100%
non-deterministic. The fix should make stage 1 deterministic, then re-check
whether the slot-0 multiply corruption is also fixed.
