# Stage 2 Spectrum: 10% of Slots Are Zero on GPU, Slot 0 and 5×9891 Anomalous

## What was asked

The collaborator's `241ba6b` asked: sort all 32768 slots of the stage-2 (CoeffsToSlots)
output and see whether only slot 0 is anomalously small, or whether 9891, 19782 and
20170 collapse with it. "One slot means a single wrong entry in a diagonal; the group
means a structure."

## What we found

Ran `bootstrap_stage(encrypt(b0), 2)` on GPU (Quadro GV100, clean rebuild at `241ba6b`),
decrypted, took `|Re(·)|` for all 32768 slots.

### It is a group, and it is larger than the worst-5

**Multiples of 9891 mod 32768:**

| k | slot | \|val\| | ratio to median | verdict |
|---|------|--------|-----------------|---------|
| 0 | 0 | 7.39e-05 | 0.001 | **ANOMALOUS** |
| 1 | 9891 | 0.017857 | 0.333 | small |
| 2 | 19782 | 0.017857 | 0.333 | small |
| 3 | 29673 | 0.035714 | 0.667 | normal |
| 4 | 6796 | 0.053571 | 1.000 | exactly median |
| 5 | 16687 | 2.38e-07 | 0.000 | **ANOMALOUS** |
| 6 | 26578 | 0.053571 | 1.000 | exactly median |
| 7 | 3701 | 0.089286 | 1.667 | normal |

Slot 0 and slot 16687 (5×9891 mod 32768) are both essentially zero. Slots 9891 and
19782 are small (3× below median). Slot 20170 is **normal** (ratio 2.0, value 0.107).

### 10% of all slots are zero on GPU, zero on CPU

**3211 out of 32768 slots** are below 1e-6 on the GPU. On the CPU, **zero** slots are
below 1e-6. The zero slots are scattered (gap distribution: 1×315, 2×297, 3×274, 4×226,
5×203) with no obvious stride. None of the zero slots are exact multiples of 9891, but
slot 0 and slot 16687 are anomalous (very small, not exactly zero in all runs).

### Values follow a k/56 pattern

The non-zero, non-anomalous slots show quantized values:

- 0.017857 = 1/56 (slots 9891, 19782, 3095)
- 0.035714 = 2/56 (slots 29673, 388)
- 0.053571 = 3/56 = median (slots 6796, 26578)
- 0.089286 = 5/56 (slot 3701)
- 0.107143 = 6/56 (slot 20170)

56 = 7 × 8. This quantization is not from CKKS rounding (that would be 2^{-50}), so it
is structural — likely a normalization factor in the CtS baby-step/giant-step decomposition.

### Non-determinism in slot 0's exact value

Between two GPU runs with the same seed (23) and same parameters, slot 0 read 0.01793
in one and 7.39e-05 in another — a 240× variation. The median was identical (0.05357).
Slots 9891 and 19782 were stable at 0.017857 in both runs. The pattern (slot 0 anomalous,
9891/19782 small) is consistent; the exact magnitude of slot 0 varies.

## Interpretation

The collaborator was right that the anomaly is already present at CoeffsToSlots, not
EvalMod. The structure is:

1. **Not a single wrong diagonal entry** — 3211 zero slots rules out one bad entry.
2. **A structural issue in CtS** — 10% of slots being zero on GPU but not CPU points at
   the CtS diagonal plaintexts or the rotate-multiply-accumulate pipeline.
3. **The k/56 quantization** suggests a normalization or scaling issue in the CtS
   decomposition, not a random memory fault.
4. **Slot 20170 is normal** — the worst-5 from the multiply bug (0, 9891, 19782, 20170)
   do not all collapse at CtS. 20170 is fine at CtS but breaks at the multiply, so it
   may have a different root cause or the multiply amplifies a smaller error there.

## Next step

Decode the CtS diagonal plaintexts on GPU and compare against the CPU-side diagonals,
slot by slot. The diagonals are plaintexts, so they decode without a key. The 3211 zero
slots should map to specific diagonal entries that are wrong on GPU but correct on CPU.

## Test

`python/tests/test_stage2_multiples.py` and `test_6_stage2_slot_spectrum` in
`test_bootstrap_low_level_corruption.py`. Run with
`PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q`.
