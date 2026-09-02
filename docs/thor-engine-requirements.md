# THOR (CCS'25) on FIDESlib/OpenFHE — engine requirements and gap analysis

Sources: THOR paper (ePrint 2024/1881), official implementation `THOR/src/thor/he.py` (desilofhe engine,
1609 lines: every stage of BERT-base is written against ~22 engine primitives), draft `thor-openfhe-main`
(OpenFHE CPU port of one encoder layer, v21: 5654 s/layer, 35 bootstraps, 40 GB level-aware keys,
RMSE 1.34e-3 on real BERT-base/MRPC layer 0, HEStd_NotSet).

## 1. What the official engine provides (desilofhe) and how FIDESlib maps to it

| # | desilofhe primitive (he.py) | Used for | FIDESlib API (fideslib::CryptoContext) | Status |
|---|---|---|---|---|
| 1 | `Engine(use_bootstrap_to_14_levels, mode="async gpu", device_id, compact)` | context | `SetDevices({})` = CPU/OpenFHE, `SetDevices({id})` = CUDA | exists (API already has CPU fallback) |
| 2 | `create_secret_key / relinearization_key / conjugation_key` | keys | `KeyGen`, `EvalMultKeyGen`, **`EvalConjugateKeyGen`** | added |
| 3 | `create_fixed_rotation_key(sk, delta, level)` (~250 keys, levels 8..14) | rotations | `EvalRotateKeyGen` + **`SetRotationKeyLevels({delta: level})`** -> level-truncated device keys | added (uses level-truncated key storage) |
| 4 | `create_bootstrap_key(size=large/medium)`; `bootstrap_deltas` reusable for rotation | bootstrap | `EvalBootstrapSetup/KeyGen`; bootstrap rotation keys are usable by `EvalRotate` (same map) | exists |
| 5 | `add / add_inplace / subtract` (ct, pt, float) | everywhere | `EvalAdd*/EvalSub*` | exists |
| 6 | `multiply(ct, light_pt)` after `prepare_for_multiply = ntt(rescale(ct))` | PC-MM, masks | `EvalMult(ct, Plaintext)`; ntt/intt are no-ops (FIDESlib keeps NTT form) | exists; **light plaintext = GAP (see 2.2)** |
| 7 | `multiply(ct, ct)` / `square(ct)` returning a *non-relinearised* ciphertext, `add` on them, one `relinearize` | CC-MM (Alg. 2 lines 16-17), Stockmeyer, Goldschmidt | none: `mult` always key-switches | **GAP (see 2.1)** |
| 8 | `multiply(ct, float)` (consumes a level), `multiply(ct, int)` (free) | DeltaCiphertext bookkeeping | `EvalMult(ct,double)`, **`EvalMultByInteger`** | added |
| 9 | `multiply_imaginary_integer(ct, 1)` | complexification (free) | **`EvalMultByI`** (monomial X^{N/2}) | added |
| 10 | `conjugate(ct)` | split real/imag, Re() | **`EvalConjugate`** | added |
| 11 | `rotate(ct, delta)` | all | `EvalRotate` (+ level-aware keys) | exists |
| 12 | `rescale(ct)` explicit | scale control | `Rescale` (needs FIXEDMANUAL, see 3) | exists |
| 13 | `level_down(ct, by)` | level alignment | **`EvalLevelReduce`** | added |
| 14 | `ct.level`, `level_available`, `engine.max_level` | scheduling | **`GetRemainingLevels`** | added |
| 15 | `encode(msg, level)`, `encrypt(msg, sk, level)`, `decrypt` | I/O, masks | `MakeCKKSPackedPlaintext(v, 1, level, nullptr, slots)`, `Encrypt`, `Decrypt` | exists |
| 16 | `read_light_plaintext(path)` (110 GB on disk for 12 layers) | weights, biases, masks | **`LightPlaintext` (GAP, see 2.2)** | to do |
| 17 | `bootstrap(ct)` on a *complex* full-slot ciphertext, output at 14 levels | 35 bootstraps/layer | `EvalBootstrap` then `EvalLevelReduce` to the schedule's level | exists; verify complex-slot precision |

Everything in he.py's stage_01..stage_18 is expressible with the rows above; the Python orchestration can
be ported mechanically once rows 7 and 16 exist.

## 2. FIDESlib gaps

### 2.1 Lazy relinearisation (three-component ciphertexts)
THOR's CC-MM multiplies ~64 pairs per output and relinearises 4 times per output ciphertext
(`ttemp` accumulation); Stockmeyer/Goldschmidt use the same idiom. Relinearising every product would
multiply the key-switch count of CC-MM by ~16. Required:
* `EvalMultNoRelin(ct, ct)` / `EvalSquareNoRelin` -> ciphertext with (c0, c1, c2)
* `EvalAdd` on degree-2 ciphertexts (and mixed degree-1/2)
* `EvalRelinearize(ct)` = ModUp(c2) -> dotKSK(evk) -> ModDown, added to (c0, c1)
FIDESlib's `Ciphertext::mult` already computes the tensor product and immediately key-switches; the
change is to split it and to give `Ciphertext` an optional `c2` RNSPoly.

### 2.2 Light plaintexts (weights on demand)
desilofhe's "light plaintext" is the encoded polynomial in a compact non-RNS form (the 110 GB / ~221k
plaintexts figure gives ~0.5 MB each = one 64-bit coefficient vector for N=2^16), expanded to RNS limbs
(reduce mod q_i + NTT) when multiplied. This is exactly the "plaintexts on demand" memory lever:
weights for a whole stage fit on the device in light form (stage_12: 6144 x 0.5 MB = 3 GB) and are
expanded in batches. Required:
* `LightPlaintext { level, scale, std::vector<int64_t> coeffs }` + encoder (FFT encode, round to int64;
  fits when scaling <= 2^50 and |m|*Delta < 2^63)
* GPU expansion kernel: per limb i, `coeffs mod q_i` (signed) then FIDESlib NTT; small expanded-plaintext
  cache keyed by (light id, level)
* `EvalMult(ct, LightPlaintext)`.

### 2.3 Bootstrap output level contract
he.py assumes bootstrap returns exactly 14 levels; the draft's `KeepRemainingLevels` drops surplus
limbs after OpenFHE's bootstrap. Same on FIDESlib: `EvalBootstrap` then `EvalLevelReduce` to the
schedule's target (`thor_level_schedule.h`: 9/10/12 depending on stage).

### 2.4 Scaling technique
he.py manages rescaling explicitly (`rescale`, `prepare_for_multiply`, `level_down`). FIXEDMANUAL is
the one-to-one match; FLEXIBLEAUTO needs the draft's `Compress`-based alignment tricks. FIDESlib's
`rescaleTechnique` supports FIXEDMANUAL; whether the FIDESlib bootstrap path is exercised under
FIXEDMANUAL must be verified on hardware (the shipped examples use FLEXIBLEAUTO).

## 3. Memory budget (V100 32 GB, N=2^16, ~33 limbs, dnum=3 -> alpha=K=11)
* Rotation keys: ~250 THOR keys at remaining level <= 14: `2*ceil(15/11)*(15+11)*0.5 MiB = 52 MiB`
  each -> ~13 GB; bootstrap keys (~57, complete) ~7.5 GB. Total ~20 GB; further reducible with larger
  key sharing (bootstrap deltas reused by THOR rotations), CtS truncation, and seeded `a`.
* Working set: <= 128 ciphertexts at ~10 limbs (softmax copies) ~ 1.3 GB.
* Weights: light form per stage <= 3 GB.
The draft's OpenFHE measurements (40 GB of level-aware keys at 50-bit primes) are consistent with this
model; the GPU budget is dominated by keys, which is why level truncation was the first change.

## 4. Architecture decision
`fideslib::CryptoContext` already switches between OpenFHE (CPU) and CUDA per call, so it *is* the
device-switchable engine. Plan:
1. Finish the two gaps (2.1, 2.2) inside FIDESlib so both backends expose the same 22 primitives.
2. pybind11 module over the fideslib API (`thorengine`), `Engine(device="cpu"|"cuda:0")`.
3. Port `he.py` stage by stage onto `thorengine` (mechanical replacement of desilofhe calls; ntt/intt
   become no-ops; `read_light_plaintext` reads the FIDESlib light format written by a ported
   `encode_weights`).
4. Validate each stage against the draft's C++/OpenFHE v21 outputs (same slot layout and error metrics),
   first on CPU (bit-for-bit the same backend), then on CUDA.

## 5. Verification checklist for the new API calls (not compiled here)
* `context->EvalAutomorphismKeyGen(sk, {2N-1})` + `InsertEvalAutomorphismKey(keys, tag)` for the
  conjugation key (alternative: `FHECKKSRNS::ConjugateKeyGen`).
* `context->GetScheme()->MultByMonomial(ct, N/2)` and `MultByInteger(ct, k)` exist on
  `SchemeBase` in the patched OpenFHE 1.5.1.1.
* `Compress(ct, towers - levels)` semantics under FLEXIBLEAUTO (draft relies on it).
* GPU `Ciphertext::conjugate`, `multMonomial`, `multIntScalar`, `dropToLevel` are the low-level ops
  wrapped; `dropToLevel` under FLEXIBLEAUTO adjusts scale (`adjustScaleAndLevel`), which differs from
  Compress — under FIXEDMANUAL both are plain limb drops.
