# Weight Cache Is Irrelevant to Bootstrap Tests; Encoding Change Confirmed

## The question

The collaborator asked whether the test code uses a weight cache, whether the
cache needs regeneration if CtS/StC encoding changed, and whether previous
reports are trustworthy.

## Answers

### 1. Test code does not use the weight cache

All bootstrap debugging tests (`test_bootstrap_low_level_corruption.py`,
`test_stage2_stability.py`, `test_stage2_multiples.py`, `test_stage_stability.py`)
use `bench_params()` from `test_bootstrap_noise_level.py`, which constructs a
fresh `pf.Engine(device, **bench_params())` per test. The engine constructor calls
`EvalBootstrapSetup` and `EvalBootstrapKeyGen` every time, building the CtS/StC
diagonals from scratch in memory. No `--plaintext-cache`, no disk store.

### 2. The disk cache is for THOR model weights, not bootstrap diagonals

`/home/zhiyuan/ptcache/` holds 19,181 `.flpt` files (9.5 GiB) under
`v1/textattack_bert-base-uncased-MRPC-n16-d37-sb50-fmb55-rs256-srs16/`.
This is `thorfhe.plaintext_store.PlaintextStore` — a cache for THOR BERT layer
weights (attention_dense, output_dense, etc.), keyed by provenance (layer/field).
It is used by `thorfhe.bench` via `--plaintext-cache DIR`, not by the bootstrap
tests.

The bootstrap CtS/StC diagonals are `std::vector<Plaintext>` in
`BootstrapPrecomputation.cuh`, built by OpenFHE's `EvalBootstrapSetup` at engine
construction. They are never serialized to disk and never go through the light
plaintext path.

### 3. Commit `7a03a0d` (Sep 18) changed light-plaintext encoding

`7a03a0d` "Encode a light plaintext from two RNS towers instead of the whole chain"
changed `MakeLightPlaintext` to encode at `towers - 2` instead of level 0, cutting
38 NTTs down to 2. This affects THOR weight encoding (`encode_to_light_plaintext`),
but **not** bootstrap diagonals, which use `MakeCKKSPackedPlaintext` directly.

The disk cache at `/home/zhiyuan/ptcache/` was created Sep 21, **after** `7a03a0d`,
so it reflects the new encoding. Even if it were stale, it would only affect THOR
benchmark runs, not the bootstrap debugging tests.

### 4. The in-memory `light_plaintext_cache` (capacity 64) is also weight-only

`Engine.__init__` sets `light_plaintext_cache_capacity = 64`, which is an LRU cache
for `GetExpandedLightPlaintext` — the THOR weight expansion path. The bootstrap code
(`Bootstrap.cu`, `CoeffsToSlots.cu`, `LinearTransform.cu`, `ApproxModEval.cu`)
never calls `MakeLightPlaintext`, `ExpandLightPlaintext`, or
`GetExpandedLightPlaintext`. Confirmed by grep: zero references to light plaintext
in any bootstrap source file.

### 5. Previous reports are trustworthy

All debugging-session reports (Sep 22-23) ran with fresh engine construction and no
disk cache. The only caveat is the one the collaborator already identified in
`e339ff3`: the CPU "clean reference" in the stage-2 spectrum report (`8236fd7`)
was invalid because the CPU path ignored `stopAfterStage`. That was corrected by
the stability test (`db84cca`), which compares GPU-to-GPU across repeats with no
CPU baseline.

## Summary

| Question | Answer |
|----------|--------|
| Do tests use weight cache? | No — fresh engine per test, no `--plaintext-cache` |
| Does cache need regen if CtS/StC encoding changed? | N/A — bootstrap diagonals are never cached |
| Is `7a03a0d` encoding change relevant? | Only to THOR weights, not bootstrap |
| Are previous reports trustworthy? | Yes, except the CPU-vs-GPU comparison already retracted |

The non-determinism finding (`db84cca`) stands: GPU CtS is non-deterministic
across 6 runs on the same input, CPU is perfectly stable, and this is independent
of any cache.
