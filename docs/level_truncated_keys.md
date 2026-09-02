# Level-truncated key-switching keys

## Why
A hybrid key-switching key (`KeySwitchingKey`) is stored as, per digit `i` (dnum digits):

* `DECOMP[i]`: the Q-limbs of digit `i` (prime ids ascending),
* `DIGIT[i]` : every special limb, then every Q-limb *not* in digit `i` (ids ascending).

Device bytes of a complete key: `2 * dnum * (L+1+K) * N * 8`.
The key-switching kernels (`dotKSK`, `hoistedRotateDotKSKBatched___`, the fused variants) only read
the limbs with prime id `<= level(ciphertext)` plus the special limbs, and skip whole digits once the
digit start offset passes the ciphertext limb count. Therefore a key that is only ever applied to
ciphertexts of level `<= m` needs only limbs with `id > L` (special) or `id <= m`; in both record lists
this is a *prefix*, so the truncated key is simply "the same allocation stopped early".

Bytes of a key truncated to level `m`: `2 * ceil((m+1)/alpha) * (m+1+K) * N * 8`, `alpha = ceil((L+1)/dnum)`.

In a bootstrap, CoeffsToSlots/Accumulate/conjugation run right after ModRaise (level `L`), but
SlotsToCoeffs runs after EvalMod at level `L - bootDepth + levelBudget[1]`, i.e. ~10-15 limbs instead of
`L+1+K`. For `N=2^16, L+1=30, dnum=3, K=10` a complete key is 120 MiB and an StC key truncated to
level 12 is `2 * 2 * 23 * 0.5 MiB = 46 MiB`.

## What was changed
* `LimbPartition::generateAllDecompAndDigit(iskey, maxLevel)` / `generateAllDigitLimb(..., maxLevel)`:
  allocate only the needed prefix; unused pointer-table entries are zeroed so a bad access faults
  instead of reading garbage. `growDecompAndDigitToLevel(m)` extends a truncated key in place.
* `RNSPoly` wrappers of the above (single-GPU only; multi-GPU keys stay complete).
* `KeySwitchingKey`: `maxLevel`, `Initialize(rkk, maxLevel, reloader)`, `ensureLevel(level)`,
  `deviceBytes()`. `ensureLevel` grows a truncated key on demand (using `reloader`, which re-reads the
  key from the OpenFHE context: no extra host memory) and prints a warning: correctness never depends on
  the level plan being exact, only memory does.
* Every key consumer in `Ciphertext.cpp` (`rotate`, `conjugate`, both `rotate_hoisted` paths,
  `HoistedRotate`) and `LinearTransform.cu` calls `ensureLevel(level)` before using a key.
* `ContextData::truncateKeys` (default true), `keyLevelMargin` (default 1), `keyDeviceBytes()`,
  `grownKeyCount()`, `printKeyMemoryReport()`. Env overrides: `FIDESLIB_KEY_TRUNCATION=0`,
  `FIDESLIB_KEY_LEVEL_MARGIN=n`.
* `GetBootstrapKeyLevelPlan(cc, slots, GPUcc)` (openfhe-interface): StC keys ->
  `L - FHECKKSRNS::GetBootstrapDepth(levelBudget, dist) + levelBudget[1] + margin`; every index that is
  also used by CtS/Accumulate stays complete. `AddBootstrapKeys` applies the plan through the new
  `AddRotationKeys(publicKey, GPUcc, indexes, maxLevels)` overload.
* API: `CryptoContextImpl::truncate_keys`, `key_level_margin`, `GetKeyDeviceBytes()`, `GetGrownKeyCount()`.
* `examples/key-truncation`: runs the same bootstrap with and without truncation and checks precision,
  memory and that no key had to be grown.

## Things to verify on real hardware (could not be compiled here)
1. `lbcrypto::FHECKKSRNS::GetBootstrapDepth(const std::vector<uint32_t>&, SecretKeyDist)` is the
   signature in the patched OpenFHE (`fideslib-ref-v1.5.1.1`). If not, use the overload with
   `approxModDepth` or compute `levelBudget[0] + levelBudget[1] + GetModDepthInternal(dist)`.
2. Run `examples/key-truncation`; if `keys grown at runtime > 0`, raise `key_level_margin` and report
   which levels were requested (the warning prints from/to levels). Expected: 0 with margin 1.
3. `GetRotationKey(index, keyID, slots, actual_index)` may resolve a CtS request to an StC key through
   its "modulo-slot compatible" fallback; `ensureLevel` handles it (warning + reload), but if it happens
   systematically the plan should mark that index complete.
4. `fusedHoistRotateBatch` (LimbPartitionBatch.cu) uses `ksk->limbptr` for the decomp digit: the table
   is rebuilt in `loadDecompDigit` with `nullptr` for missing limbs, and the kernel grid only spans
   `limbsize + special`, so truncation is safe as long as `level <= maxLevel` (guaranteed by ensureLevel).
5. Multi-GPU contexts force `maxLevel = -1`.

## Next steps (same design, more savings)
* CtS layer `i` keys can be truncated to `L - i` (small gain).
* Seed-expanded `a` component (halves every key) - needs the patched OpenFHE key generation.
* On-demand generation of the CtS/StC diagonal plaintexts (currently ~2-3 GiB resident for N=2^16).
