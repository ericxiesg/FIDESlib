# Bug note: GPU segfault when growing a level-truncated rotation key (ensureLevel)

## Status
**Severe — blocks CUDA path of the level-truncated-key feature.** Stopped per instructions.

## Environment
- GPU: Quadro GV100 (sm_70), CUDA 12.9 toolkit (CUDA 13.3 cannot target sm_70).
- OpenFHE: patched 1.5.1.1 at /home/zhiyuan/workspace/THOR-FIDE/openfhe-install.
- Commit: 00e3905 (bootstrap-dev), FIDESlib built with arch auto-detect → 70-real.

## Symptom
On the CUDA backend, calling `EvalRotate` with a rotation index whose key was
loaded **level-truncated** segfaults (SIGSEGV) the first time the ciphertext's
level exceeds the key's `maxLevel`, i.e. exactly when `KeySwitchingKey::ensureLevel`
must **grow** the key in place. A second distinct rotation index also crashes
even if the first succeeded, because the first grow corrupts device state.

Reproducer (pyfideslib, device cuda:0):
```python
import numpy as np, pyfideslib as pf
# key for index 2 truncated to "remaining level 5"; ciphertext sits at level 12.
e = pf.Engine("cuda:0", rotation_indexes={2: 5},
              log_n=13, depth=12, scaling_bits=50, first_mod_bits=55, dnum=3)
x = np.random.uniform(-1, 1, e.slots)
cx = e.encrypt(x)            # level 12
e.rotate(cx, 2)             # SEGFAULT here (no output, no Python exception)
```
- `e.cc.GetKeyDeviceBytes()` reports 18 MiB (truncated) right before the crash.
- `GetGrownKeyCount()` is never reached.
- Same sequence with `truncate_keys=False` (full keys) works: all rotations pass,
  err ~1e-9, grown==0.
- CPU backend is unaffected: `test_rotate_truncated_key_low_level` passes
  (err ~1e-6, grown==0) — the CPU path never grows keys (OpenFHE keeps full keys).

## Root cause (preliminary)
The grow path is new code from the level-truncated-keys patch:
`KeySwitchingKey::ensureLevel` (src/CKKS/KeySwitchingKey.cu) →
`RNSPoly::growDecompAndDigitToLevel` →
`LimbPartition::growDecompAndDigitToLevel` (src/CKKS/LimbPartition.cu), followed by
`loadDecompDigit` with the reloaded raw key.

`growDecompAndDigitToLevel` extends the DECOMP/DIGIT limb vectors and rewrites the
device pointer table (`DECOMPlimbptr`/`DIGITlimbptr`) for *this* key's polynomials,
but the key-switching kernels in `Ciphertext::rotate` / `LinearTransform.cu` use
the context's shared auxiliary polynomial (`cc.getKeySwitchAux()`) and digit/gather
limbs that may be sized for the *original* (truncated) layout. After the grow, a
mismatch between the newly-extended key limbs and the pre-generated gather/digit
metadata of the context aux poly is the most likely source of the out-of-bounds
device read. The crash is silent (no CUDA error raised before the segfault),
consistent with an invalid device pointer/dimension used by a launch.

The pointer-table entries are zeroed for missing limbs on truncation
(`cudaMemsetAsync(..., 0, ...)` in `generateAllDecompAndDigit`/
`generateAllDigitLimb`); `growDecompAndDigitToLevel` appends new entries and
copies the *full* table back, but the DIGIT `generate()` call it relies on writes
only the new entries — the interaction with the previously-zeroed tail and the
context gather limbs needs auditing.

## What is NOT the bug
- The compile-time namespace/OPS/ConstPlaintext issues (fixed separately, see below).
- Complex-slot encoding (fixed: added `SetCKKSDataType(COMPLEX)`).
- `EvalSub(scalar, ct)` GPU sign error (fixed).
- The level plan / `GetBootstrapKeyLevelPlan` computation itself (it produces a
  correct plan; the crash is in the on-demand grow, not in the planning).

## Impact on tests
- CPU pytest: 16/16 pass (all stages 1-4).
- CUDA pytest: stage 1 (4/4) pass; stage 2 first 4 tests pass
  (after the `EvalSub(scalar,ct)` fix), then `test_rotate` segfaults when it
  exercises a rotation whose key was truncated below the ciphertext level.
  Stages 3-4 not reached on CUDA.

## Suggested next steps (not executed — severe bug, stopped)
1. Add a minimal C++ reproducer in `examples/key-truncation` that calls
   `SetRotationKeyLevels` + a single rotate that forces a grow, run under
   `cuda-memcheck` / `compute-sanitizer` to locate the invalid access.
2. Audit `LimbPartition::growDecompAndDigitToLevel` against
   `generateAllDecompAndDigit` (the initial-generation path) for full symmetry,
   especially the DIGIT limb `generate()` call and the `DECOMPlimbptr`/`DIGITlimbptr`
   table rebuild + the gather-limb regeneration.
3. Check whether `cc.getKeySwitchAux()` (shared across keys) needs to be
   re-generated/regrown to match the grown key's digit layout before the
   `MGPUkeySwitchCore` call in `Ciphertext::relinearize`/`rotate`.
4. Until fixed, the CUDA path can run with `truncate_keys=False` (or env
   `FIDESLIB_KEY_TRUNCATION=0`) as a workaround; the level-plan computation
   should short-circuit to all-complete when grow is known-broken on GPU.
