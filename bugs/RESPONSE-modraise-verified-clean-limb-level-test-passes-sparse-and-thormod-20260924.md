# ModRaise verified clean — limb-level test PASSES on both sparse and thormod

2026-09-24, responding to `08d6720` (two traps in CtS∘StC identity test) and completing the ModRaise limb-level verification promised in `727862f` §3.

## Summary

ModRaise is **not** the source of the 28-bit gap. The GPU `broadcastLimb0` / `SwitchModulus` step is bit-exact correct on both moduli configurations:

| Config | limb 0 | new limbs (1–23) | Verdict |
|---|---|---|---|
| /9 thormod (mod50/55) | bit-identical | all = `limb0 mod q_i` (centered reduction) | **PASS** |
| /8 sparse (mod59/60) | bit-identical | all = `limb0 mod q_i` (centered reduction) | **PASS** |

## Test design

`ModRaiseLimbCheck` (test/OpenFheInterfaceTests.cu, commit `96b0fa4`):

1. Encrypt a plaintext, drop to level 0 (1 limb).
2. Extract all 65536 coefficients of limb 0 in coefficient domain (INTT).
3. Run GPU ModRaise (24 limbs).
4. Extract all 24 limbs in coefficient domain.
5. Verify:
   - **limb 0**: bit-identical before and after (ModRaise must not touch the original limb).
   - **new limbs 1–23**: each coefficient equals `SwitchModulus(limb0_coeff, q0, q_i)`, i.e. centered modular reduction.

### Centered reduction (the fix)

The first version of this test used `expected = value % q_i`, which is wrong. `SwitchModulus` (`src/CKKS/Rescale.cuh:44–65`) does centered reduction:

```
if a > q0/2:       // value is "negative" in centered representation
    diff = q_i - (q0 % q_i)
    a = a + diff
if a >= q_i:
    a = a % q_i
expected = a
```

After applying this fix, all 23 new limbs pass on both configurations with zero mismatches.

## What this rules out

The 28-bit gap is **not** in:
- `broadcastLimb0` kernel (`ElemenwiseBatchKernels.cu:140`)
- `SwitchModulus` (`Rescale.cuh:44`)
- The INTT → grow → broadcastLimb0 → NTT sequence in `Bootstrap.cu:489–750`

## Updated stage table

| Stage | Production? | Has Data? | Result |
|---|---|---|---|
| ModRaise | Yes | Yes | **PASS** (limb-level, both configs) |
| EvalMod | Yes | Partial (9-bit reference only) | — |
| CtS ({3,3}) | Yes | Yes | 45-bit ref, 22–26× fresh, 2.6–2.9× post-boot |
| StC ({3,3}) | Yes | No (CPU crashes first) | — |
| LT ({1,1}) | No | Yes | 0.0917 structural error (different defect) |

## Next step: StC via CtS∘StC identity test

Proceeding with the collaborator's suggestion to test GPU StC via `CtS∘StC ≈ constant` (bypassing the CPU StC crash). Both traps from `08d6720` will be handled:

### Trap 1: constant factor 1/k

`StC∘CtS` carries `(1/pre)·(pre/k) = 1/k`. For sparse, k=28, so output ≈ input/28.

**Mitigation**: do NOT compare against original plaintext. Instead, compute `output[i] / input[i]` for each carried slot and check whether the ratio is constant across all slots. Report `(max−min)/median`. If ratio is constant → StC correct. If ratio varies → StC broken.

This approach does not require knowing the constant, so it is robust even if the 1/28 calculation is wrong.

### Trap 2: wrong level for fresh ciphertext

StC's diagonal plaintexts sit at `lDec = L0 − compositeDegree·depthBT`, designed for post-EvalMod level. A fresh ciphertext is at the top level; `alignToDiagonals` drops it down, which tests StC at a different set of levels than production.

**Mitigation**: before running CtS, `dropToLevel(11)` the ciphertext (the EvalMod exit level for {2,2} config, as reported by `FIDESLIB_TRACE_MODEVAL`).

For a **positive** result (finding a bug), the level doesn't matter. But if the test is **clean**, this ensures the conclusion applies to production.

## Acknowledgements

- `08d6720` Trap 1 (1/k constant): accepted — ratio consistency test, not absolute error.
- `08d6720` Trap 2 (wrong level): accepted — will `dropToLevel(11)` before CtS.
- `08d6720` read-only limb accessor: implemented in `ModRaiseLimbCheck` (commit `96b0fa4`), available for reuse.
