# FIXEDAUTO experiment (per NOTE-t2-thor-stages-01-05.md request)

## Setup

Temporarily set `scaling_technique=_core.FIXEDAUTO` in conftest.py's `SMALL` and
in test_stage6's `thor_engine` fixture, then ran the full pytest suite on CPU.

## Result: 4 failures, 31 passed, 1 skipped

```
FAILED test_stage2_linear.py::test_mult_plaintext_and_rescale[cpu]
FAILED test_stage2_linear.py::test_mult_float_scalar_consumes_level[cpu]
FAILED test_stage5_light_plaintext.py::test_expands_at_the_ciphertext_level[cpu]
FAILED test_stage6_thor_qkv.py::test_fhe_stages_match_clear[cpu]
```

## Root cause

FIXEDAUTO automatically rescales after every multiplication, so the level
tracking differs from FIXEDMANUAL's explicit-rescale model:

- `test_mult_plaintext_and_rescale`: after `multiply` + `rescale`, FIXEDAUTO
  has already auto-rescaled, so the explicit `rescale` is a no-op (or drops
  an extra level), and the expected level is wrong.

- `test_mult_float_scalar_consumes_level`: same issue — `multiply` by a float
  scalar triggers auto-rescale, changing the level immediately.

- `test_expands_at_the_ciphertext_level`:
  ```
  assert 12 == ((12 - 0) - 1)  # level didn't drop after multiply + rescale
  ```
  FIXEDAUTO auto-rescaled the multiply, so the explicit `rescale` call is
  redundant and the level stays at 12 instead of dropping to 11.

- `test_fhe_stages_match_clear`:
  ```
  assert [5, 5] == [4, 4]  # FHE level 5, clear level 4
  ```
  The clear engine (ClearEngine) models FIXEDMANUAL explicitly. Under
  FIXEDAUTO, one of the auto-rescales in the stage pipeline happens
  implicitly, so the FHE ciphertext ends up one level higher than the clear
  model predicts.

## Conclusion

**FIXEDAUTO is not compatible with the current test suite and thorfhe code.**
The level/scale discipline is designed for FIXEDMANUAL's explicit rescale
model, where every `rescale()` call is deliberate and `level_down()` is a
manual operation. Switching to FIXEDAUTO would require rewriting the level
tracking in both the tests and the `thorfhe.stages` / `thorfhe.clear` modules
to account for implicit rescaling.

**Recommendation: keep FIXEDMANUAL.** The submitter's workaround (rewriting
`prepare_for_multiply` and `rotate_internal` for FIXEDMANUAL) is already in
place and all 36 tests pass. Switching to FIXEDAUTO would save those two
rewrites but require changes everywhere else.
