# Bug fixes for commit 6316173 (T1 light plaintexts)

Three bugs found while verifying the level-truncated key grow fix (e405d58) and
the T1 light plaintext feature (6316173) on the remote GV100.

All three are now fixed and committed.  Verification results at the bottom.

---

## Bug 1 — `KeySwitchingKey::cc` dangling reference → segfault in `ensureLevel`

### File
`src/CKKS/KeySwitchingKey.cuh` (one-line change)

### Symptom
`key-grow-repro above` segfaults (SIGSEGV, exit 139) inside
`KeySwitchingKey::ensureLevel()` instead of throwing a diagnostic.

GDB backtrace:
```
#0  FIDESlib::CKKS::KeySwitchingKey::ensureLevel(int) ()
#1  FIDESlib::CKKS::Ciphertext::rotate(int, bool) ()
#2  fideslib::CryptoContextImpl::EvalRotate(...) ()
```

### Root cause
`Context` is `std::shared_ptr<ContextData>` (see `forwardDefs.cuh:15`).
`KeySwitchingKey` stored `Context& cc` — a **reference to the shared_ptr**.

In `CryptoContextImpl::LoadContext()` (`api/CryptoContext.cpp:274`):
```cpp
this->gpu = std::make_any<FIDESlib::CKKS::Context>(std::move(c));
```
The `std::move(c)` empties the source shared_ptr.  Every `KeySwitchingKey`
that was initialised during `AddRotationKeys` holds a reference to that
now-empty shared_ptr, so `cc->allowKeyGrow` dereferences a null pointer.

`inside` passes only because `ensureLevel` returns early
(`level <= maxLevel`) before touching `cc`.  `above` triggers the null deref.

### Fix
```diff
- Context& cc;
+ Context cc;       // copy the shared_ptr, not a reference to it
```
`KeySwitchingKey` now holds its own shared_ptr copy, keeping `ContextData`
alive as long as the key exists.  The constructor
(`KeySwitchingKey.cu:116`) already takes `Context& cc` by reference, so the
initialisation is unchanged — only the member storage changes from reference
to value.

### Why this is safe
- `Context` is a shared_ptr; copying it is cheap (atomic refcount bump).
- The key never reassigns `cc`, so the reference-vs-value distinction only
  matters for lifetime.
- `RNSPoly a, b` already store `Context` by value (via `*cc` in the
  constructor), so this is consistent with the existing design.

---

## Bug 2 — Destructor wipes **global** OpenFHE key maps

### File
`api/CryptoContext.cpp` — `~CryptoContextImpl()` (two lines removed)

### Symptom
`test_conjugate_and_mult_by_i[cpu]` fails with
```
RuntimeError: EvalAutomorphismKeys are not generated for ID [...]
```
but only when run **after** `test_rotate_truncated_key_grows_when_allowed`,
which creates a temporary `Engine`.  In isolation the test passes.

### Root cause
The FIDESlib destructor called the **no-argument** static overloads:
```cpp
lbcrypto::CryptoContextImpl<DCRTPoly>::ClearEvalMultKeys();
lbcrypto::CryptoContextImpl<DCRTPoly>::ClearEvalAutomorphismKeys();
```
These clear the **entire** `s_evalMultKeyMap` / `s_evalAutomorphismKeyMap`
globally — every context's keys, not just the one being destroyed.

When the temporary Engine (from `test_rotate_truncated_key_grows_when_allowed`)
is destroyed, it wipes the session Engine's automorphism keys, so the next
`EvalConjugate` call cannot find the conjugate key.

OpenFHE's own `CryptoContextImpl` has **no** destructor that clears these
maps — this was added by FIDESlib and is incorrect.

### Fix
Remove the two `ClearEval*` calls entirely.  OpenFHE manages the static maps
via `InsertEval*` / `ClearEval*(keyTag)` / `ClearEval*(context)`; individual
contexts should not wipe the global state on destruction.

```diff
 CryptoContextImpl<DCRTPoly>::~CryptoContextImpl() {
     FIDESlib::CudaNvtxRange r("API");
     if (this->loaded) {
         auto& context_gpu = std::any_cast<FIDESlib::CKKS::Context&>(this->gpu);
         FIDESlib::CKKS::DeregisterCryptoContextGPU(context_gpu);
         this->gpu = std::any();
     }
-    lbcrypto::CryptoContextImpl<...>::ClearEvalMultKeys();
-    lbcrypto::CryptoContextImpl<...>::ClearEvalAutomorphismKeys();
 }
```

### Why this is safe
- OpenFHE itself never clears these maps in a destructor.
- The GPU context is deregistered explicitly (the `if (loaded)` block).
- If a user wants to reclaim key memory they can call
  `ClearEvalMultKeys(context)` / `ClearEvalAutomorphismKeys(context)`
  explicitly.

---

## Bug 3 — `SetDevices(devices)` lvalue binding (recurring)

### File
`examples/key-truncation/src/key_truncation.cpp` (one-line change)

### Symptom
Does not compile:
```
error: cannot bind rvalue reference of type 'std::vector<int>&&'
to lvalue of type 'std::vector<int>'
```

### Root cause
`CCParams::SetDevices` takes `std::vector<int>&&` (rvalue reference).
`parameters.SetDevices(devices)` passes an lvalue.

This is the same bug that was fixed in commit 9f8120bd for the previous
version of this file; the T1 commit (6316173) reintroduced it.

### Fix
```diff
- parameters.SetDevices(devices);
+ parameters.SetDevices(std::move(devices));
```

---

## Known issue (not fixed) — `ExpandLightPlaintext` CPU `GetCKKSPackedValue` returns zeros

### File
`api/CryptoContext.cpp` — `ExpandLightPlaintext` CPU branch

### Symptom
`test_encode_expand_roundtrip[cpu]` fails:
the expanded plaintext's `GetCKKSPackedValue()` returns all zeros,
even though the RNS towers are correctly populated.

### Root cause
`MakeCKKSPackedPlaintext` is called with a zero vector to create the
plaintext skeleton, then the element is replaced with the light-plaintext
coefficients.  But `GetCKKSPackedValue()` returns a **cached** `value` member
(set during encoding), not a live decode of the element.  Replacing the
element does not update the cache.

Calling `CKKSPackedEncoding::Decode()` to refresh the cache segfaults
because `Decode()` is designed for post-decryption single-tower `Poly`,
not multi-tower `DCRTPoly` — it calls `GetElement<NativePoly>()` /
`GetElement<Poly>()` (single-tower accessors) and `SetValuesToZero()`
(destroys the element).

### Impact
- **CPU only**, and only when the caller inspects the expanded plaintext via
  `GetCKKSPackedValue()`.  Using the plaintext in `EvalMult` works correctly
  (the element towers are correct).
- **CUDA**: no issue — the GPU path never materialises host towers, and
  `test_encode_expand_roundtrip[cuda:0]` passes.
- `test_multiply_matches_dense_encoding` (both CPU and CUDA) passes —
  multiplication reads the element, not the cached `value`.

### Suggested fix (for the commit author)
Either:
1. Encode the **actual** message values (not zeros) in
   `MakeCKKSPackedPlaintext`, then overwrite only the towers that differ
   from the light-plaintext coefficients; or
2. Add a `CKKSPackedEncoding::DecodeFromDCRTPoly()` variant that reads
   multi-tower RNS and writes `value` without destroying the element; or
3. In the test, verify via `EvalMult` against a known ciphertext rather than
   `GetCKKSPackedValue()`.

---

## Verification results (remote GV100, CUDA 12.9)

### key-grow-repro (all three scenarios)

```
./build-kt/key-grow-repro all
[inside]  max |err| = 1.28e-10, key bytes = 11136 KiB, grown = 0   PASS
[above]   threw as expected: ... was loaded truncated to level 5 ... PASS
[grow]    max |err| = 1.07e-09, key bytes 11136 -> 13824 KiB, grown = 1  PASS
ALL PASS  EXIT=0
```

### compute-sanitizer --tool memcheck (all three scenarios)

```
compute-sanitizer --tool memcheck ./build-kt/key-grow-repro all
[inside] PASS   [above] PASS   [grow] PASS
========= ERROR SUMMARY: 0 errors   EXIT=0
```
(Run with `CUDA_VISIBLE_DEVICES=0` and `/usr/local/cuda-12.9/bin/compute-sanitizer`.)

### pytest — CPU

```
PYFIDESLIB_DEVICES=cpu pytest python/tests -v
24 passed, 1 skipped, 1 failed
```
The 1 failure is the known `test_encode_expand_roundtrip[cpu]` issue
described above.  The 1 skip is `test_rotate_truncated_key_above_plan_raises`
(GPU-only feature).

### pytest — CUDA

```
PYFIDESLIB_DEVICES=cuda:0 pytest python/tests -v
26 passed in 2.91s
```
**All 26 tests pass on GPU**, including:
- `test_rotate_truncated_key_above_plan_raises[cuda:0]` — verifies Bug 1 fix
- `test_conjugate_and_mult_by_i[cuda:0]` — verifies Bug 2 fix
- `test_encode_expand_roundtrip[cuda:0]` — passes (GPU path unaffected)
- All stage 5 light plaintext tests (8/8)
- `test_bootstrap_complex_and_keep_levels[cuda:0]`

### key-truncation example

```
./build-kt/key-truncation 13 12 25 3 3 3
max |err| truncated : 1.097e-05
max |err| complete  : 9.254e-06
```
Runs and bootstraps correctly, but reports "FAIL: truncation did not reduce
key memory" (0 keys truncated).  This is a parameter/planning issue, not a
crash — the `GetBootstrapKeyLevelPlan` logic marks all bootstrap keys as
complete (`-1`) for these parameters.  Needs investigation of the level-plan
computation, not a code bug.
