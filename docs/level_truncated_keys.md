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
  instead of reading garbage. `resetDecompAndDigit()` releases them again and
  `decompDigitLevelCovered()` reports the highest ciphertext level the resident limbs can serve.
* `RNSPoly` wrappers of the above (single-GPU only; multi-GPU keys stay complete).
* `KeySwitchingKey`: `maxLevel`, `Initialize(rkk, maxLevel, reloader)`, `ensureLevel(level)`,
  `rebuildAtLevel(m)`, `coversLevel(level)`, `deviceBytes()`.
* Every key consumer in `Ciphertext.cpp` (`rotate`, `conjugate`, both `rotate_hoisted` paths,
  `relinearize`) and `LinearTransform.cu` calls `ensureLevel(level)` **before queueing any GPU work**
  for that operation: a reload frees and reallocates the key limbs and synchronises the device, which
  must not happen while the shared mod-up auxiliary polynomial is in flight.
* `ContextData::truncateKeys` (default true), `keyLevelMargin` (default 1), `allowKeyGrow`
  (default false), `keyDeviceBytes()`, `grownKeyCount()`, `printKeyMemoryReport()`. Env overrides:
  `FIDESLIB_KEY_TRUNCATION=0`, `FIDESLIB_KEY_LEVEL_MARGIN=n`, `FIDESLIB_KEY_GROW=1`.

## What happens when the level plan is wrong

`ensureLevel(level)` is a comparison when the key already covers `level`. When it does not, the
(delta -> level) table the caller passed to `SetRotationKeyLevels` is wrong, and by default that is
**reported**: `ensureLevel` throws naming the key, its truncation level and the level it was asked for.
Silently reloading would turn a one-line table mistake into a host->device key transfer on every call --
in THOR, hundreds of them per layer -- and hide it behind a warning nobody reads.

Set `allowKeyGrow` (API `allow_key_grow`, env `FIDESLIB_KEY_GROW=1`) to get the fallback instead: the
key is reloaded through its `reloader` and **rebuilt from scratch** at the higher level by
`rebuildAtLevel`, i.e. `resetDecompAndDigit()` followed by the same `generateDecompAndDigit` +
`loadDecompDigit` pair `Initialize` uses. Rebuilding rather than appending the missing limbs keeps a
single implementation of the limb layout: the key-switching kernels reach key limbs through the
device-side `C_.pos_in_digit` tables plus `DECOMPlimbptr`/`DIGITlimbptr`/`limbptr`, so a second
allocation path that drifts from the first produces silent out-of-bounds device reads.
* `GetBootstrapKeyLevelPlan(cc, slots, GPUcc)` (openfhe-interface): StC keys ->
  `L - FHECKKSRNS::GetBootstrapDepth(levelBudget, dist) + levelBudget[1] + margin`; every index that is
  also used by CtS/Accumulate stays complete. `AddBootstrapKeys` applies the plan through the new
  `AddRotationKeys(publicKey, GPUcc, indexes, maxLevels)` overload.
* API: `CryptoContextImpl::truncate_keys`, `key_level_margin`, `allow_key_grow`, `GetKeyDeviceBytes()`,
  `GetGrownKeyCount()`; `pyfideslib.Engine(..., truncate_keys=, allow_key_grow=)`.
* `examples/key-truncation`: runs the same bootstrap with and without truncation and checks precision,
  memory and that no key had to be grown (it enables `allow_key_grow` because measuring the plan is
  the point of the example).
* `examples/key-truncation` also builds `key-grow-repro`: the three truncated-key scenarios (used inside
  the plan / above it / above it with growth enabled) in one small binary, for `compute-sanitizer`.

## Things to verify on real hardware (could not be compiled here)
1. `lbcrypto::FHECKKSRNS::GetBootstrapDepth(const std::vector<uint32_t>&, SecretKeyDist)` is the
   signature in the patched OpenFHE (`fideslib-ref-v1.5.1.1`). If not, use the overload with
   `approxModDepth` or compute `levelBudget[0] + levelBudget[1] + GetModDepthInternal(dist)`.
2. Run `examples/key-truncation`; if `keys grown at runtime > 0`, raise `key_level_margin` and report
   which levels were requested (the warning prints the key and both levels). Expected: 0 with margin 1.
   Then run `key-grow-repro all`, also under `compute-sanitizer --tool memcheck`.
3. `GetRotationKey(index, keyID, slots, actual_index)` may resolve a CtS request to an StC key through
   its "modulo-slot compatible" fallback; with the default policy that now surfaces as an `ensureLevel`
   exception naming the index, and the plan should mark that index complete.
4. `fusedHoistRotateBatch` (LimbPartitionBatch.cu) uses `ksk->limbptr` for the decomp digit: the table
   is rebuilt in `loadDecompDigit` with `nullptr` for missing limbs, and the kernel grid only spans
   `limbsize + special`, so truncation is safe as long as `level <= maxLevel` (guaranteed by ensureLevel).
5. Multi-GPU contexts force `maxLevel = -1`.

## How much this actually saves, and where

Measured on a GV100 (see `bugs/FIX-commit-6316173-bugs.md`): with `slots = N/2` the plan truncates
**zero** bootstrap keys. `GetBootstrapKeyLevelPlan` can only truncate a key that StC uses and CtS does
not, and at full slots the two linear transforms use the same 46 rotation indices. THOR bootstraps at
full slots, so *this* lever buys the project nothing.

The saving that does matter is the other consumer of the same machinery: `SetRotationKeyLevels` on
THOR's ~250 fixed rotation keys, each of which is used at one known level. `python/thorfhe`'s
`plan_rotation_keys` derives that table from a dry run of the stages (docs/thor_port.md), so it stays
correct as the port grows.

## Next steps (same design, more savings)
* CtS layer `i` keys can be truncated to `L - i`. This is the only way the bootstrap keys can be
  truncated at full slots, since every StC index is also a CtS index there.
* Seed-expanded `a` component (halves every key) - needs the patched OpenFHE key generation.
* On-demand generation of the CtS/StC diagonal plaintexts (currently ~2-3 GiB resident for N=2^16).
