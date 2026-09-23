# CtS / StC review, the test evidence for them, and whether the weight cache must be regenerated

2026-09-23. Against `842585e`. Four questions: review the CoeffsToSlots / SlotsToCoeffs code, find the
evidence in the test suite that argues they are correct, decide whether the remote has to rebuild the
plaintext cache, and assess building the diagonals ourselves instead of taking OpenFHE's.

---

## Executive summary

**The weight cache does not need regenerating.** The premise it rests on is not merely asserted, it is
pinned by a test that has been running on both devices since stage 5, at a tolerance of 1e-9. §3.

**The CtS/StC tests are well built but were not running our configuration**, so a green result could
not be cited as evidence about the benchmark. The gap was five parameters wide, and two of them reach
directly into the linear transform. This is now fixed rather than just reported: the bootstrap suite
takes a THOR parameter set, and the CoeffsToSlots test no longer pins a baby-step split the benchmark
never uses. §2.4.

**The code review found no defect in CtS/StC.** The one construct matching our failure signature —
`alignToDiagonals` dropping to the minimum diagonal level — was already excluded empirically on 09-15.
§1.

**On building the diagonals ourselves: not yet, and not for speed.** It is one-time setup, so it is
not on the inference hot path at all. The format risk is real and worth closing, but a validator
closes it in hours where a reimplementation costs days and would remove the only independent oracle we
have. §5.

---

## 1. The code

`src/CKKS/CoeffsToSlots.cu`, 206 lines. `EvalCoeffsToSlots(ctxt, slots, decode)` at :76 serves both
directions off one body — `decode` picks `StC` or `CtS` out of the precomputation — and
`EvalLinearTransform` at :27 is the single-step form used when the level budget is {1,1}.

Four observations, none of them a defect:

**`alignToDiagonals` drops to the minimum diagonal level.** The lambda carries its own hazard note:
uneven diagonals would be "wrong differently per slot because each diagonal feeds different slots",
which is very close to the signature we have been chasing. It also carries a once-per-process
`std::cerr` warning behind `static bool reportedUnevenDiagonals`. That warning has never fired:
`bugs/RESPONSE-bootstrap-noise-is-iid-20260915.md:9` records the diagnostic running and the diagonals
being level-consistent within a step. The hypothesis is excluded, and it stays excluded.

**`constexpr bool BATCHED = false` at :25** makes the batched branch dead code — it is entirely
commented out. Not a risk, but not a thing that can be tested either.

**`assert(step.slots == step.A.size())`** is compiled out under NDEBUG, so the release build has no
check that a precomputation step's diagonal count matches its slot count. See §5.3.

**`if (ctxt.NoiseLevel == 2) ctxt.rescale();` at the entry** normalises the incoming scale degree
silently. This is the same observe-rather-than-declare shape the bootstrap output contract in
`BootstrapPrecomputation.cuh` was added to correct, one level down. Not urgent, but it is the third
instance of the pattern.

Production takes the dense path throughout: `Engine.slots = 1 << (log_n - 1)` = N/2
(`pyfideslib/__init__.py:57`), so `cc.N / 2 == slots` and the sparse-encapsulation branches in
`Bootstrap.cu` are not reached. Their lack of coverage costs us nothing.

---

## 2. The test evidence

### 2.1 The tests are structurally strong

`TEST_P(OpenFHEBootstrapTest, CoeffsToSlots)` at `test/OpenFheInterfaceTests.cu:2995` and
`SlotsToCoeffs` at :3151 do the right thing: they run OpenFHE's own `FHE->EvalCoeffsToSlots` /
`EvalSlotsToCoeffs` on the CPU and FIDESlib's `EvalCoeffsToSlots(GPUct1, slots, false|true)` on the
GPU, from the *same* ciphertext, decrypt both, and compare. Both values of `decode` are covered, so
there is no untested branch in the dispatch.

`ASSERT_ERROR_OK` (`test/ParametrizedTest.cuh:559`) takes the per-slot absolute difference of the real
parts over all `result->GetSlots()` = 32768 and asserts on the **maximum**. A single corrupted slot
raises the maximum, so this is the right statistic for what we are looking for — an RMSE would have
hidden it.

### 2.2 One real laxity in the assertion, but not the limiting one

```cpp
std::cout << "Max error: " << Max << " (Expected: " << pow(2.0, -result->GetLogPrecision() + 1) << ")...";
ASSERT_LE(Max, pow(2.0, -result->GetLogPrecision() + 4));
```

It prints one bar and asserts on another eight times looser. Worth tightening. It is not what is
limiting the evidence here: the deviations we are chasing are 10^2 to 10^3 times the expected error
(slot 0 at EvalMod, 776x; the stage-2 row 725x too small), and those clear either bar by orders of
magnitude.

### 2.3 What was actually limiting: the tests did not run our configuration

Both tests are instantiated from `TTALL64BOOT` (:3853), which resolves to eight parameter sets, all of
them `tparams64_13_*` — a misnomer, since `logNboot = 16`. Against `thorfhe.bench`
(`python/tests/test_bootstrap_noise_level.py:33`):

| | `TTALL64BOOT` | THOR bench | |
|---|---|---|---|
| logN | 16 | 16 | same |
| slots | N/2 = 32768 | N/2 = 32768 | same |
| level budget | {3,3} | (3,3) | same |
| dnum | 1, 2, 3, 4 | 4 | covered |
| depth L | 23 | **37** | differs |
| scale bits | 59 | **50** | differs |
| first modulus | 60 | **55** | differs |
| baby-step split (dim1) | **{16,16}** in the CtS test | **{0,0}**, OpenFHE chooses | differs |
| scaling technique | FIXEDAUTO, FLEXIBLEAUTOEXT | **FIXEDMANUAL** | never run |
| secret key dist | **UNIFORM_TERNARY** (hardcoded, `ParametrizedTest.cuh:514`) | **SPARSE_TERNARY** | never run |

Two of those reach straight into the code under review:

- **dim1.** The baby-step/giant-step split decides how many diagonals each step holds, which rotations
  it needs, and what `bStep`/`gStep` are per step. `{16,16}` is a different decomposition from the one
  that runs. (The SlotsToCoeffs test at :3188 already passed `{0,0}`; only CoeffsToSlots pinned it.)
- **FIXEDMANUAL.** `EvalCoeffsToSlots` opens with a FIXEDMANUAL-relevant rescale and sits between
  FIXEDMANUAL branches in `Bootstrap.cu`. No parameter set in the suite ever reached them.

The secret key distribution matters one step further out rather than inside CtS: FIDESlib selects the
Chebyshev coefficient set and the double-angle depth from it (3 and depth 10 sparse, 6 and depth 13
uniform), so it changes the level budget CtS hands over, not CtS itself.

### 2.4 Fixed, not just reported

- `GeneralTestParams` now carries `secretKeyDist` (default `UNIFORM_TERNARY`, so every existing set is
  unchanged), and `SetUp` reads it instead of hardcoding.
- The context cache key in `SetUp` folded in ring dim, depth, technique, dnum and scale bits but not
  `firstModSize` or the key distribution, so two configurations differing only in those would have
  silently shared a context. Both are now in the key. No existing pair collided; this is insurance.
- `tparams64_16_thor_fixmanual`: logN 16, L 37, dnum 4, first modulus 55, scale 50, FIXEDMANUAL,
  SPARSE_TERNARY. `TTALL64BOOTTHOR` = `TTALL64BOOT` plus that one, and **only the bootstrap suite**
  takes it — depth 37 at logN 16 is the expensive end and the interface suite has no use for it.
- The CoeffsToSlots test's `EvalBootstrapSetup({3,3}, {16,16}, slots)` becomes `{0,0}`, matching both
  the benchmark and the SlotsToCoeffs test next to it.

None of this is compiled here — there is no compiler on this machine. If depth 37 at logN 16 makes the
bootstrap suite too heavy for the box, dropping `tparams64_16_thor_fixmanual` back out of
`TTALL64BOOTTHOR` is a one-line revert and the other four changes stand on their own.

### 2.5 The honest summary of the evidence

The tests establish that the GPU linear transform computes the same map as OpenFHE's, for the
configurations they run. That is real and it rules out a gross algorithmic error in CtS/StC. Until the
change above they did not cover the configuration where the anomaly appears, and in particular never
ran FIXEDMANUAL or the auto-selected baby-step split — so a green `CoeffsToSlots` could not be offered
as evidence that CtS is correct in the benchmark. After the change it can be, once the remote runs it.

---

## 3. The weight cache: no regeneration needed

The question is whether `7a03a0d` ("Encode a light plaintext from two RNS towers instead of the whole
chain") changed the bytes on disk, because the cache would not notice if it had.

**What is stored.** `LightPlaintextImpl::coeffs` — N signed int64 centred representatives of RNS tower
0, taken mod q0 (`api/CryptoContext.cpp:2219-2229`). Nothing else.

**What the commit changed.** Only `encodeLevel`, from 0 to `cheapestLevel = towers - 2`, and only when
`encodeLevelMatters` is false, i.e. not FLEXIBLE*. OpenFHE drops towers from the tail of the chain, so
tower 0 is q0 at every level. The stored bytes therefore change only if the encoded integer
coefficients depend on the level — which under FIXEDMANUAL they do not, because the scaling factor
does not.

**That premise is tested, not assumed.** The commit message called it unverified. It was already
verified, by a test written earlier for a different reason:
`python/tests/test_stage5_light_plaintext.py:29 test_multiply_matches_dense_encoding`. The dense path
encodes at `depth - level(ct)` (`pyfideslib/__init__.py:169`), which is **0** for a fresh ciphertext;
the light path encodes at `cheapestLevel`, which is **11** under the small parameters. The test
requires the two products to agree to **1e-9**, on both `cpu` and `cuda:0`. That is exactly the
level-0-versus-cheapest comparison, at a tolerance far below the noise floor.

**The cache key carries what it needs to.** `plaintext_cache_tag` (`thorfhe/bench.py:141`) folds in
model, log_n, depth, scaling_bits, first_mod_bits, residual_scale and score_refresh_scale — every input
that changes an encoded coefficient under FIXEDMANUAL. `dnum` and the key distribution are correctly
absent: neither touches plaintext encoding.

**Conclusion: keep the cache.** Regenerating it would cost hours and change nothing.

Two caveats, neither a reason to regenerate now:

- The tag does not carry the scaling technique. Under FLEXIBLE* the level hint is honoured and
  coefficients do become level-dependent, so a FIXEDMANUAL-written cache read by a FLEXIBLE* engine
  would be silently wrong. Production is always FIXEDMANUAL, so this is latent, and it is one string
  to add to the tag.
- The tag carries no encoder version. `LAYOUT = "v1"` is documented as tracking the on-disk layout and
  the manifest, not the encoder's semantics. `7a03a0d` happened to be a no-op on the bytes. The next
  such change might not be, and nothing would invalidate the tree.

**One residual, now closed.** `test_multiply_matches_dense_encoding` runs at depth 12, so it compares
level 0 against level 11; the benchmark's gap is level 0 against level 36. The mechanism —
level-independence of the FIXEDMANUAL scaling factor — has no depth dependence, so this is a formality
rather than a doubt. `test_encoding_is_level_independent_at_bench_depth` has been added to
`test_stage5_light_plaintext.py` to check it at the benchmark's own parameters, behind
`PYFIDESLIB_BENCH_PARAMS=1` like the other bench-parameter tests.

---

## 4. What this unblocks

`bugs/RESPONSE-bootstrap-precision-diagnostic-20260915.md` found a **period-2048** structure in the
bootstrap error: the spread across `slot % 2048` is 0.015 against 0.0007 across `slot % 16`, a factor
of 23. 2048 is THOR's group size, 128 tokens by 16 blocks. Its recommended follow-up was to look at the
CtS/StC plaintext diagonals, and it was never done.

That follow-up is now worth running, for a reason that did not hold last week: the CoeffsToSlots and
SlotsToCoeffs tests will, once the remote rebuilds, be running FIXEDMANUAL at depth 37 with the
benchmark's baby-step split. A green result there kills the CtS hypothesis for the period-2048
structure outright. A red one localises it to a specific diagonal step, which is a far better starting
point than the whole bootstrap.

---

## 5. Building the diagonals ourselves

### 5.1 Where they come from today

`AddBootstrapPlaintexts` (`src/CKKS/openfhe-interface/RawCiphertext.cu:1318`) does not compute
anything. It reads two members straight out of OpenFHE's precomputation map and transcribes them:

```cpp
auto& A    = precom->m_U0hatTPreFFT;   // CtS
auto& invA = precom->m_U0PreFFT;       // StC
...
result.CtS.at(i).A.emplace_back(GPUcc_, GetRawPlainText(cc, A.at(A.size() - 1 - i).at(j)));
...
result.StC.at(i).A.emplace_back(GPUcc_, GetRawPlainText(cc, invA.at(i).at(j)));
```

Everything about the format is inherited: the layer count, the diagonal count per layer, the level each
diagonal is encoded at, the ordering within a layer, and the ordering of the layers themselves. Note
that the last of those is **reversed on one side only** — `A.at(A.size() - 1 - i)` for CtS against
`invA.at(i)` for StC. That asymmetry is correct as far as we know, but it is a convention we copied,
not one we derived, and nothing in the tree checks it.

The code already knows this is a seam. The `std::cerr` at :1362 exists to print the diagonals' tower
count next to the ciphertext's, with the comment that the former "is decided entirely by OpenFHE's
`EvalBootstrapSetup`" while the latter "is chosen here, by scaling technique". It prints the two
numbers; it does not compare them.

### 5.2 CUDA-ising it would buy nothing

This is one-time setup per (context, slot count), not per bootstrap. It is not on the inference path at
all: the device's own `--time-ops` run put all 22 bootstraps at 4.16 s of a 1154 s layer, and the
precomputation is outside even that. Whatever it costs, it costs once per process, and moving it to the
GPU would shorten startup, not inference.

So the case for reimplementing is entirely about format control, and should be argued on that basis.

### 5.3 The case against reimplementing now

Two reasons, and the second is the stronger one.

The diagonals are the hardest part of bootstrap to get right. The level-budget-3 form is not a plain
diagonal extraction — it is OpenFHE's collapsed-FFT decomposition of `U0^T`, where the merging of FFT
stages into layers and the rotation indices each layer needs are precisely the fiddly part.
Reimplementing it is days, with a real chance of introducing exactly the class of defect we are
currently hunting.

And OpenFHE's diagonals are our **only independent oracle**. The CtS/StC tests are worth something
specifically because the CPU side is a different implementation. If we generate the diagonals
ourselves and feed them to both sides, the test compares our arithmetic against our arithmetic and
stops being evidence. We would have to keep OpenFHE's construction anyway, as the reference — at which
point we own two implementations instead of one.

### 5.4 What to do instead: check the contract rather than replace the producer

The format risk is real. It is closed by validating the imported precomputation against invariants we
declare, which is hours of work and no new arithmetic — the same move that `BootstrapPrecomputation`'s
`OutputContract` made for the bootstrap's exit state, applied to its entry state.

The invariants worth asserting at import, all cheap:

1. `A.size() == result.CtS.size()` and likewise for StC — the layer count matches the level budget.
   Today a mismatch surfaces as a `std::out_of_range` from `.at(i)` with no explanation.
2. `step.slots == step.A.size()` for every step. This is the assert at `CoeffsToSlots.cu` that NDEBUG
   removes; at import it costs nothing and is always on.
3. Every diagonal within a layer has the same level. `alignToDiagonals` assumes this and warns to
   `std::cerr` at use time; at import it can be an error, and it is the invariant whose violation the
   09-15 investigation was specifically looking for.
4. Layer *k*'s level is exactly layer *k−1*'s level minus one — each step consumes one level. This is
   the assumption that makes the level budget arithmetic work, and nothing checks it.
5. CtS layer 0's level is at least the post-ModRaise ciphertext level, so the drop `alignToDiagonals`
   performs is non-negative. This is the comparison the existing `std::cerr` prints both halves of
   without making.
6. The layer ordering: for CtS, layer *i* must have a *lower* level than layer *i−1*. If OpenFHE ever
   changed its ordering convention, the reversal at :1354 would silently invert the transform, and
   this is the one line that would catch it.

If all six hold at THOR's parameters, the inherited format is confirmed and reimplementation has no
motive. If one fails, we know which one — and then a targeted reimplementation of that specific piece
is a well-posed job rather than a rewrite.

### 5.5 The one place an independent encoding might pay, and why it is a trade

A separate idea, worth naming and then setting aside. The diagonals are stored expanded — every tower
materialised on device. At level budget 3 in each direction that is roughly 6 layers of order 60
diagonals, each holding tens of towers of N = 65536 uint64: single-digit GiB, on a card where stage_07
already peaks at 0.3 GiB of headroom. Storing them in the light (single-tower) form instead would cut
that by an order of magnitude.

The cost is that expansion is an NTT per tower per use, and unlike a weight — used once per layer — a
diagonal is used once per bootstrap, 22 times a run. That is the trade, and it is not obviously
favourable. It is worth measuring only if device memory becomes the binding constraint again; it is
not a correctness argument and should not be mixed with one.

---

## 6. For the remote

1. Rebuild and run `OpenFHEBootstrapTests/OpenFHEBootstrapTest.CoeffsToSlots` and `.SlotsToCoeffs`.
   The new case is the `tparams64_16_thor_fixmanual` instantiation; report its `Max error` line
   alongside the existing eight so the two configurations can be compared directly. If depth 37 at
   logN 16 does not fit, say so and revert `TTALL64BOOTTHOR` to `TTALL64BOOT` at :3853 — the rest of
   the change is independent of it.
2. `PYFIDESLIB_BENCH_PARAMS=1 pytest python/tests/test_stage5_light_plaintext.py -k level_independent`.
   Expected to pass. If it fails, the weight cache *is* stale, `7a03a0d` is wrong, and the answer in §3
   inverts — so it is worth the one run despite the confidence.
3. **Do not regenerate the weight cache.**
4. Report the `[FIDESlib] bootstrap diagonals:` line from `RawCiphertext.cu:1362` at THOR's parameters.
   It is already printed; nobody has quoted it back. It is invariant 5 of §5.4 with the comparison left
   to the reader, and it is free.

Still outstanding from earlier instructions, unchanged: the six repeated complete bootstraps for the
78.8% stage-2 question, the rebuild for `e339ff3` (the CPU `stopAfterStage` throw, which invalidates
the earlier GPU-vs-CPU comparisons), the `842585e` output contract, and
`FIDESLIB_USE_FUSED_KEYSWITCH=1` for accuracy as well as speed.
