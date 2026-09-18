# Light plaintexts

## Why

BERT-base under THOR needs roughly 220 000 encoded weight plaintexts. A normal encoded plaintext is
`(L+1)` RNS towers of `N` 64-bit words — 16 MiB at `N = 2^16, L = 30` — so the weights alone are
~110 GiB of device-shaped data. A 32 GiB V100 cannot hold a single layer's worth.

Encoding, however, factors into a level-independent part and a per-level projection:

```
message  --special IFFT-->  reals  --* Delta, round-->  N integer coefficients  --mod q_i, NTT-->  towers
         \_______________________ level independent _______________________/     \__ per level __/
```

Under FIXEDMANUAL (and FIXEDAUTO) the scaling factor `Delta` does not depend on the level, so the
integer coefficient vector *is* the plaintext: `N` int64 words, 0.5 MiB, expandable at whatever level
the ciphertext happens to sit at. That is a 33x reduction at `L = 32`, and it is what desilofhe calls
`encode_to_light_plaintext` / `write_light_plaintext` / `read_light_plaintext` — the three calls
`THOR/src/thor/he.py` uses for every weight and mask.

## Representation

`fideslib::LightPlaintextImpl` (api/LightPlaintext.hpp):

| field | meaning |
|---|---|
| `coeffs` | the `N` **centred** coefficients of `round(Delta * IFFT(message))`, natural (non bit-reversed) coefficient order — the order an OpenFHE `DCRTPoly` uses in `Format::COEFFICIENT` |
| `scale` | the `Delta` they carry |
| `slots`, `noise_scale_deg` | plaintext metadata, copied through to the expansion |
| `level_hint` | level the plaintext is meant for, or -1; only enforced under FLEXIBLE* scaling |
| `uid` | process-wide identity, keys the expansion cache |

Centred means each coefficient lies in `(-q0/2, q0/2)`, which is why one int64 is enough: `Delta` is
at most `2^50` and messages are `O(1)`, so `|c| < 2^54 < q0/2`. `MakeLightPlaintext` **verifies** this
by re-reducing the extracted coefficients modulo the second tower and comparing — a message too large
for a single tower has no compact form and is rejected instead of silently wrapping.

## Encoding and expansion

`MakeLightPlaintext` does not implement its own CKKS encoder. It calls OpenFHE's
`MakeCKKSPackedPlaintext`, converts the result to `Format::COEFFICIENT` and centre-lifts tower 0.
Using OpenFHE's encoder is the point: any discrepancy between the compact and the dense path would
otherwise be our own IFFT's rounding, not a real difference.

**It asks that encoder for two towers, not the whole chain.** OpenFHE materialises one NTT per tower,
and this reads exactly two of them — tower 0 for the coefficients and tower 1 for the check below —
so the level it encodes at is a pure cost knob under FIXEDMANUAL, where the diagram above says the
coefficients do not depend on the level at all. Encoding at level 0 built all 38 towers at depth 37
to keep two, and that was **96% of a layer's wall clock on the device**: 19 137 encodes at 58 ms, or
1109 s of 1154 s, against 4.16 s for all 22 bootstraps. Under FLEXIBLE* the scaling factor *is*
level-dependent, so there the `level_hint` is honoured instead.

`ExpandLightPlaintext(lp, level)` (`level` = *consumed* levels, OpenFHE's convention):

* **CPU backend** — build each tower as a `NativePoly` of `coeffs[j] mod q_i`, assemble a `DCRTPoly`
  in `Format::COEFFICIENT`, `SetFormat(EVALUATION)`, and swap it into a plaintext made at that level
  (which supplies the right params and scaling factor). OpenFHE does the NTT.
* **CUDA backend** — upload the `N` int64 coefficients once (0.5 MiB, not 16 MiB), run
  `expandCentredCoeffs_` (one signed remainder per coefficient per limb) and then FIDESlib's forward
  NTT: `RNSPoly::loadCentredCoefficients` → `Plaintext::loadLight`. The host never sees a tower.

`EvalMult(ct, lp)` / `EvalAdd(ct, lp)` expand at the ciphertext's level through a FIFO cache keyed by
`(uid, level)`, sized by `light_plaintext_cache_capacity` (default 64, 0 disables). THOR reuses the
same masks across a layer, so the cache turns most uses back into a plain plaintext multiply;
`ClearLightPlaintextCache()` bounds device memory between stages.

## File format

`LightPlaintextImpl::Save` / `Load`, little-endian, 32-byte header:

```
char[8]  "FLPT0001"
double   scale
uint32   slots
uint32   noise_scale_deg
int32    level_hint
uint32   n
int64[n] coeffs
```

0.5 MiB + 32 bytes per file at `N = 2^16`. Deliberately raw — THOR reads hundreds of thousands of them.

## Scaling technique

Only FIXEDMANUAL and FIXEDAUTO make a light plaintext level-agnostic. Under FLEXIBLEAUTO(EXT) the
scaling factor differs per level, so the coefficients are only valid at the level they were encoded
for; `ExpandLightPlaintext` throws when `level_hint` is set and does not match. This project runs
FIXEDMANUAL (see HANDOFF), so the restriction only matters if someone switches the engine over.

## Verified on hardware (GV100, CUDA 12.9)

The coefficient ordering assumption held: `RNSPoly::loadCentredCoefficients` feeds natural-order
coefficients straight to FIDESlib's forward NTT, no bit reversal on either side. `REVERSE == false`
in `openfhe-interface/RawCiphertext.cuh` (the evaluation layouts already agree) plus `INTT`/`NTT`
being exact inverses inside FIDESlib is the reason, and
`test_stage5_light_plaintext.py::test_multiply_matches_dense_encoding` passes on **both** backends —
which is what would have caught a mismatch, since CPU goes through OpenFHE's NTT and CUDA through
FIDESlib's. All 8 stage-5 tests pass on CPU and CUDA.

## Known limitation: the expansion's cached packed value

`CKKSPackedEncoding` keeps the packed message it was encoded from in a cached `value` member.
`ExpandLightPlaintext`'s CPU branch builds the skeleton plaintext by encoding zeros and then replaces
its *element*, and there is no way to refresh that cache: `CKKSPackedEncoding::Decode()` assumes a
single-tower post-decryption `Poly` and destroys a `DCRTPoly` element.

So `GetCKKSPackedValue()` on an expanded light plaintext reports the skeleton's zeros, while
`EvalMult` / `EvalAdd` — which read the element — are correct. On CUDA the question does not arise:
the expansion has no host copy at all. **Inspect a light plaintext by multiplying and decrypting, not
by decoding its expansion.** No THOR path decodes a weight plaintext, so this stays a wart rather than
a gap; fixing it properly would mean a `Decode` variant that reads multi-tower RNS without consuming
the element.

## Cost

One 64-bit signed remainder per coefficient per limb, then the existing NTT. If the remainder ever
shows up next to the NTT in a profile, replace `%` with a Barrett reduction against
`C_.prime_better_barret_mu`.

## Next steps

* THOR's `encode_weights.py` writes one file per weight; port it onto `Engine.write_light_plaintext`
  so the on-disk layout matches what `he.py` reads (T2).
* An expanded plaintext currently holds `(level+1)` towers on the device for as long as it is cached.
  For stages that stream many weights, prefer `light_plaintext_cache_capacity = 0` and let each
  multiply expand into scratch.
