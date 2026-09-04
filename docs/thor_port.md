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

## Stage 06, and a bug in he.py

Stage 06 forms the attention scores as a ciphertext-ciphertext product. It runs in three parts, each
an exact operation with a stated formula (`thorfhe/attention.py`):

| part | what it does |
|---|---|
| `transpose_upper_to_lower` | a permutation of the packed key: `out[ct][group, tau, b] = k[t, f]` with `ct*pack + group = (t - d) mod n_out` and `tau = d + n_out * ((d > t mod n_out) XOR (t // n_out))` |
| `make_copies` | broadcasts every diagonal across all groups: `copies[l][group, t, b] = q[t, n_out*b + (l+t) mod n_out] / 2` |
| `stage_06_attention_score` | `out[ct][group, tau, b] = (Q_b K_b^T)[tau, (ct*pack + group + tau) mod dim]` |
| `stage_08_attention_context` | `out[ct][group, tau, b] = ctx[tau, n_out*b + (ct*pack+group+tau) mod n_out] + i * ctx[tau, n_out*b + ((ct+2)*pack+group+tau) mod n_out]`, `ctx = A_b V_b` |

All three were derived by probing the numpy model with one-hot inputs, the same way the QKV layout was,
and all three are checked against those formulas with zero tolerance.

**`he.py` computes the wrong attention score.** The inner product accumulates into four columns per
output ciphertext, and a contribution goes to columns 2-3 or 0-1 depending on whether the source
ciphertext index wrapped past zero. The `in_index % pack != 0` branch routes on exactly that
(`i - in_index // pack < 0`). The `in_index % pack == 0` branch instead routes on `i == 0`. The two
agree when `in_index // pack == 1` and disagree when it is 2 or 3 - so two of the sixty-four
inner-product terms land in the wrong accumulator, where they are then conjugated and multiplied by i
before being summed.

The effect is not marginal: with `he.py`'s routing the score is off by ~32% of its own magnitude, in
output ciphertexts 1, 2, 5 and 6. With the other rule it is `Q K^T` to machine precision (3e-16). The
port uses the correct rule, `AttentionScore.accumulator_column` is overridable, and
`test_he_py_accumulator_table_is_wrong` pins the difference so a well-meaning revert fails.

Stage 08 settles what the general rule is. There `out_dim` is 2 rather than 4, so offsets reach -4 and
a source index can wrap *twice* and land back in columns 0-1 - which means the rule is the parity of
the wrap count, `2 * ((offset // out_dim) % 2)`, not simply `offset < 0`. `he.py`'s stage 08 tables
follow that in every one of their entries, in both branches; `test_accumulator_routing_reproduces_he_py_stage_08`
checks all sixteen. So the same rule reproduces three of the four tables in `he.py` and only the score's
`j == 0` table stands apart.

This is worth double-checking against THOR's reported MRPC accuracy before treating it as settled -
an error this size in the attention scores should be visible end to end, so either the artifact
differs from the paper's numbers here or something compensates downstream.

## Stage 07: the softmax

There is no exponential and no division in CKKS, so THOR builds a low-temperature softmax out of a
degree-15 polynomial and sharpens it with Goldschmidt iterations (`thorfhe/numeric.py`,
`thorfhe/softmax.py`). Measured against the real thing on the numpy engine: row sums within 0.23% of
one, individual weights within 2.6e-3, and the whole attention chain - stages 06, 07 and 08 - within 2%
of `softmax(Q K^T) V`.

Three things about it are easy to get wrong and are now written down where they bite:

* **The exponent bookkeeping.** `he_exp1` gives `exp(u/2)` for the `u` handed to `he_softmax`;
  `he_exp2` gives `exp(u/4)`, because its range is twice as wide; each `update_inv_D` doubles the
  exponent; and `stage_07_softmax`'s bootstrap fold hands `he_softmax` **twice** the score. Which
  polynomial is used is decided by `max_x >= 30`, so a re-calibration silently changes the temperature
  unless `l` moves with it.
* **The input window is narrow in both directions.** Above `max_x` the degree-15 fit stops being an
  exponential and diverges fast - 10% past the range overflows the denominator. Below `inv_epsilon`
  the Goldschmidt iteration is inverting something outside the range it was set up for. The usable
  denominator window is about three decades, and the `softmax_scale = 1/512` folded into the key
  projection is what exists to hit it. `softmax.calibrate` computes the same choice from a sample of
  scores, which is what a re-calibration on real activations would do.
* **The level schedule.** The clear engine enforces FIXEDMANUAL, and it caught three real errors while
  this was being written: a missing rescale after the scalar division, two ciphertext products whose
  operands had drifted apart in level, and a rescale applied to an integer multiply that does not need
  one. `Stages.align` is where operands that took different routes are brought together.

## Open questions

* **The he.py score bug.** Confirmed against the linear algebra, not against THOR's own outputs (the
  draft repo is not in this checkout). Re-check when a real MRPC sample runs end to end.
* **Layer >= 1 input.** `stage_01_complexify_x` has a second branch, for the `2 * n_input_ciphertexts`
  real/imaginary ciphertexts a previous layer produces, which folds in a rotation by `n_blocks // 2`.
  It is ported but only reachable once stage 16 exists, so it is untested.
* **`temp` (the real/imaginary split) is unused by stages 01-05** and its extra rescale is charged
  where it happens rather than where THOR charges it. Re-check when stage 11 lands.
