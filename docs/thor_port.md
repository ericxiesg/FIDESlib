# Porting THOR onto fideslib (stages 01-05)

`python/thorfhe/` is THOR's BERT layer rewritten against `pyfideslib.Engine`, so the same code runs on
OpenFHE (CPU) and FIDESlib (CUDA). This note covers what stages 01-05 do, the three places the port
had to diverge from `THOR/src/thor/he.py`, and how any of it is checked.

## Layout

| module | what it holds |
|---|---|
| `geometry.py` | `Geometry`: the slot layout as data. `THOR_BERT` is BERT-base; `SMALL` is the same structure in 4096 slots. |
| `encoding.py` | numpy encoders: activations, weights, biases, masks, and `decode_linear_output`. Ports `model_encoder.py` + `data_encoder.py`. |
| `stages.py` | `Stages`: stages 01-05 and their helpers, written against an engine-shaped object. |
| `clear.py` | `ClearEngine`: the same primitives over numpy, with strict level/scale checking. |
| `he.py` | light-plaintext encoding of the weights, and `plan_rotation_keys`. |
| `weights_io.py` | THOR's on-disk weight tree, in the light-plaintext format. |

## The packing

One ciphertext holds `pack` groups of `dim` tokens; each token owns `n_slot` consecutive slots of which
`n_blocks` carry data:

```
slot = group * group_size + token * n_slot + block
group_size = dim * n_slot      slot_count = pack * group_size
```

A hidden vector is carried in *diagonal* form. Input feature `f` of token `t` sits on lower diagonal
`(f - t) mod n_in_complex` of block `(f // n_in) % horizontal`, in the real part when
`f % n_in < n_in_complex` and the imaginary part otherwise - two features to a complex slot, which is
what halves the ciphertext count. The output of a linear layer comes back as

```
out[ct][group, token, block] = y[token, n_out * block + ((ct * pack + group) + token) % n_out]
```

`encode_activations` / `decode_linear_output` are those two formulas; `Geometry` is the constants.

THOR hard-codes all of this as `SLOT_COUNT = 2**15`, `GROUP_SIZE = 2**11`, `PACK = 16`, `DIM = 128` plus
bare `12`, `16` and `64` literals. Separating them is what lets the whole pipeline run in 4096 slots in
a test - and it caught a real ambiguity, because `pack` and `n_slot` are both 16 in THOR and are *not*
the same quantity. `SMALL` sets `pack=4, n_slot=8` so a port that conflated them fails.

## What the stages compute

* **01 complexify** - splits each input into its real and imaginary halves (which layernorm needs
  later) and hands the QKV product the complex-packed original, dropped six levels.
* **02 rotated copies** - `pack` rotations by `group_size`, so copy `r` has group `r` aligned to group
  0. These `n_input_ciphertexts * pack` copies feed all three of Q, K and V.
* **03/04/05 query/key/value** - `pcmm`: a plaintext-ciphertext inner product over the copies for each
  of `diag_count` block diagonals (`parallel_diagonal_pc_mult`), then the diagonals are folded together
  by rotating within each token's block window (`rotate_internal`), then bias, then `y + conj(y)` to
  make the result real.

End to end that is a linear layer, and that is exactly how it is tested:

```
decode_linear_output(stage_03(...)) == x @ w.T + 2 * b     (to 1e-12, in numpy)
```

The `2 *` is THOR's: `encode_weight` halves the weights to pay for the `y + conj(y)` doubling, and
`encode_b` does not halve the bias. The port reproduces it rather than fixing it, because the
downstream stages are calibrated against THOR's numbers.

## Three divergences from `he.py`

**1. Rotation direction.** desilofhe's `rotate(ct, delta)` moves slot `i` to slot `i + delta`; OpenFHE
and FIDESlib move it to `i - delta`. Every delta in `stages.py` reads as it does in `he.py`, and
`Stages.rotate` negates once. This is not cosmetic: with the wrong sign the QKV output is not even
injective in the input, which is how it was found.

**2. Plaintext operand order.** `he.py` calls `multiply(mask, x)` and `add(bias, wx)` with the
plaintext first. `Engine.multiply` dispatches on its second argument, so `Stages.add`/`multiply` swap.

**3. Scaling technique.** This is the substantive one. desilofhe manages CKKS scales automatically, so
`he.py` can call `rescale` where a rescale is merely *pending*, and can subtract a freshly masked
ciphertext from an unmasked one. fideslib runs FIXEDMANUAL, where a ciphertext is canonical at scale
`Delta`, a product sits at `Delta^2` until rescaled, and adding the two is silently wrong. So:

* `prepare_for_multiply` is the identity here. In `he.py` it is `ntt(rescale(x))`: the NTT is a no-op
  because fideslib keeps ciphertexts in the evaluation domain, and the rescale is desilofhe settling a
  deferred product. The rescale that FIXEDMANUAL needs already happens after the accumulation, in
  `parallel_diagonal_pc_mult`.
* `rotate_internal` multiplies by the mask *and* by its complement instead of masking once and
  subtracting, so both halves sit at `Delta^2` and one rescale brings the result back.
  `x - mask * x` and `(1 - mask) * x` are the same value; only the bookkeeping differs. It costs one
  extra plaintext multiply per call - five per output ciphertext per stage.
* The `1/2` in stage 01 is a scalar multiply and so costs a level; `he.py` leaves that rescale to
  desilofhe.

`ClearEngine` enforces the contract (`ScaleMismatch` on mismatched level or scale), so a slip fails a
millisecond test instead of surfacing as noise after a GPU run.

**FIXEDMANUAL is the decision, not a placeholder.** Moving to FIXEDAUTO would restore desilofhe's
automatic rescaling and let the stages read exactly like `he.py`, but the explicit discipline is what
makes the level schedule - the thing the whole memory budget rests on - visible and checkable. So the
rules below apply to every stage still to be ported:

* rescale after a plaintext multiply before adding the result to anything canonical;
* to mask off part of a ciphertext, multiply by `mask` and by `1 - mask`, never `x - mask * x`;
* drop `he.py`'s rescales of already-canonical ciphertexts (they are desilofhe settling a deferred
  product), but keep every `level_down` (those are THOR scheduling levels on purpose).

## Level schedule and rotation keys

Stages 01-05 cost 8 levels from a fresh ciphertext: THOR's deliberate `level_down(x, 6)` in stage 01,
plus one for `parallel_diagonal_pc_mult`'s rescale and one for `rotate_internal`.

`plan_rotation_keys(geometry, depth, layer_index)` returns `{rotation index: highest level used}`,
ready for `SetRotationKeyLevels`. It runs the stages on the clear engine with zero data and records the
level of every rotation, so the key schedule cannot drift from the code that uses the keys - THOR keeps
the same table (`rotation_contexts` in `he.py`) by hand. At BERT geometry and depth 20 it yields
`{2048: 14, 7..11: 13, -5..-1: 13}` for stages 01-05.

## Weights on disk

`weights_io.write_linear` / `read_linear` write THOR's tree (`stage_03/layer_0/w_0_2_37`) with one
light plaintext per file. One BERT QKV projection is 1540 files and 770 MiB; the three projections
across 12 layers are 27 GiB, on the way to THOR's ~110 GiB for the whole model. Loading the checkpoint
is deliberately left to the caller - pass numpy arrays and keep safetensors and transformers off the
GPU box.

## What is checked, and where

| check | cost | where |
|---|---|---|
| the packing reproduces `x @ w.T + 2b` exactly, at BERT geometry **and** at `SMALL` | ~3 s, numpy | `test_qkv_computes_xw_plus_bias` |
| padding slots stay empty | ms | `test_unused_slots_stay_empty` |
| the level schedule is what the key plan assumes | ms | `test_level_schedule`, `test_rotation_plan_covers_every_rotation` |
| the FIXEDMANUAL contract is enforced | ms | `test_scale_discipline_is_enforced` |
| the FHE port agrees with the numpy mirror | seconds | `test_fhe_stages_match_clear` |
| the on-disk weight tree round-trips | seconds | `test_weight_files_round_trip` |

The draft OpenFHE repo (`qkv_pcmm_openfhe.cpp`) that the handoff names as the reference is not
available in this checkout, so the reference here is the linear algebra itself, which is stronger: it
is independent of both implementations.

## Open questions

* **Layer >= 1 input.** `stage_01_complexify_x` has a second branch, for the `2 * n_input_ciphertexts`
  real/imaginary ciphertexts a previous layer produces, which folds in a rotation by `n_blocks // 2`.
  It is ported but only reachable once stage 16 exists, so it is untested.
* **`temp` (the real/imaginary split) is unused by stages 01-05** and its extra rescale is charged
  where it happens rather than where THOR charges it. Re-check when stage 11 lands.
