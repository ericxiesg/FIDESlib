# Porting THOR onto fideslib (stages 01-18, one layer end to end)

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

## Stages 10-16: the dense layers, LayerNorm, and the feed-forward block

Everything after the attention chain is linear algebra plus two non-linearities, and all of it is
checked slot-by-slot against numpy on the clear engine:

| stage | file | what it computes | measured |
|---|---|---|---|
| 10 attention dense | `dense.py` | `ctx @ W.T`, folded from 2 output ciphertexts to the 6-block form | 7.2e-16 |
| 11 / 16 LayerNorm | `layernorm.py` | `2 * (gamma * (x - mean) / sqrt(var + eps) + beta)` | 3.3e-6 |
| 12 intermediate dense | `feedforward.py` | `x @ W1.T / 64`, 768 -> 3072 | exact |
| 13 GELU | `feedforward.py` | `gelu` elementwise, two polynomials in series | 4.7e-5 abs |
| 14 output dense | `feedforward.py` | `h @ W2.T`, 3072 -> 768 | exact; 12->13->14 within 2e-4 |
| 15 prepare LayerNorm | `layernorm.py` | the residual, bootstrapped | exact (see below) |

### The feed-forward window is a property of the product, not of the packing

Neither feed-forward weight fits a block diagonal: the expansion is (3072, 768) and the contraction
(768, 3072). THOR splits each into four square pieces - `vsplit` for the first, `hsplit` for the
second - and packs two per `rep`, so a rep carries twelve block rows where the attention stages carry
six. Those twelve live in a token's sixteen slots as **two windows of six** (`FF_SLOT_INDICES`), and
the partial products are recombined with `block_diag_2`: a window of six on a stride of **eight**, not
twelve on sixteen.

The first attempt made that a second `Geometry` (256 half-tokens of 8 slots). The slot arithmetic is
identical, so it looks clean, and it is wrong: the *encoder* indexes by token, and re-indexing the
packing walks `arange(dim)` against a 128-row block and overflows `group_size`. The window belongs to
the multiply. `rotate_internal` and `pcmm` therefore take `window`, `masks` and `complements`
explicitly, and `block_diagonal_masks` takes `stride` and `width`; the geometry is unchanged.

The resulting layout is pinned exactly, for both the expansion and its GELU:

```
out[rep][ct][group, t, s] = z[t, 128 * block + (ct*pack + group + t) mod 128]
block = 12 * rep + FF_SLOT_INDICES.index(s)
```

and the contraction lands back in the ordinary 6-block form the LayerNorm reads.

### The 64 is load-bearing

`GELU_SCALE = 64` is not a nicety. THOR composes a degree-31 with a degree-27 polynomial because a
single minimax fit of `tanh` over the pre-activation range would need a hopeless degree; the composite
is only valid for an argument in `[-1, 1]`, so the ciphertext carries the pre-activation **divided by
64** and `stage_12`'s weight is encoded with `scale = 1/64` to put it there. Outside that window the
degree-31 inner polynomial stops being bounded, which a test asserts so nobody quietly drops the
scale. Thirteen levels, 4.7e-5 absolute.

### Two folds that look the same and are not

Stages 13 and 15 both merge two real ciphertexts into one complex one, bootstrap once instead of
twice, and split back out with `x + conj(x)` / `i * (conj(x) - x)`. That split **doubles**. Stage 13
multiplies by `1/2` first, so the values that come out are the ones that went in - the halving is
there for the bootstrap's input bound, not for the arithmetic. Stage 15 does not, so it hands stage 16
*twice* the residual, deliberately: `variance_window` accepts four times the variance for exactly the
LayerNorm variants (2 and 3) that stage 16 routes to, and those are exactly the ones that halve their
input. The two conventions cancel, and a test states each half of that so neither can be "fixed" alone.

### The dead mask in stage 14

`he.py` builds a `slot % 16 < 6` mask in stage 14 and never applies it. It is not needed, but the
reason is worth writing down: the `rotate(temp, -8)` that sums the two windows pulls the *next*
token's low window into slots 8..13, so the output is meaningful only on slots 0..5. LayerNorm's
`value_mask` zeroes everything else as its first operation, so the pollution never reaches an
arithmetic result. Slots 6, 7, 14 and 15 do stay clean, because the packed weight is zero there.

## Stages 17 and 18: the classification head

The pooler and the classifier act on the CLS token alone, which is why they look unlike everything
before them (`thorfhe/pooler.py`). The pooler masks token 0 out of the last LayerNorm output and
broadcasts it back over all 128 tokens, so every group already holds the same token and there is no
rotated-copy dimension: `encode_w_pooler` folds the pack offset into the *input* index instead, giving
a (6, 4) weight where an ordinary dense layer needs (8, 6, 64). It closes with an `interval_sum` over
the groups, because here the groups carry parts of one inner product rather than independent output
diagonals. The layout that comes out is exact, and is exactly the one `encode_w_cls` expects:

```
slot n_slot * t + block  ==  (W @ cls)[dim * block + t]
```

Two details are easy to lose:

* **The pooler bias is halved** (`encode_b_pooler`), so the closing `y + conj(y)` gives `+b`. This is
  the opposite of the QKV convention, where the bias is not halved and the layer computes `x @ w.T + 2b`.
* **The classifier folds six slots, not eight.** `temp + rot(temp, -1)`, then `+ rot(., -2)` off that
  pair, then `+ rot(pair, -4)` - off the *pair* again, not the quad. A plain doubling chain would sum
  eight and pull the padding slots in. Then `dim` tokens by `n_slot` through `n_slot * dim`.

The pooler's `tanh` is two degree-15 polynomials with a bootstrap on either side - far cheaper than
GELU's degree-31/27 composite, because the `/40` in stage 17 puts the argument well inside the fit.
It is accurate to 1.1e-2 for a pre-activation in +/-10, and past that it *plateaus* at 0.18 rather
than diverging, which is the opposite of how the GELU inner polynomial fails. End to end the head
reproduces `w_cls @ tanh(W @ cls + b) + b_cls` to 1.4e-3, the tanh fit being the whole of the error.

## The doubled-ciphertext convention

**Every ciphertext in THOR carries twice the value it represents.** This is not stated anywhere in
`he.py`; it was found by running one real BERT layer against the plaintext model and reading the
best-fit scale of every stage (`python -m thorfhe.bench fhe --per-stage`). Three things maintain it:

* the weight encoders halve (`gather_upper_diagonal_batch` returns `(real - i*imag) / 2`), so a
  plaintext-ciphertext product of a doubled input comes back singled;
* the bias is added **before** the `y + conj(y)` that doubles, so relative to a doubled input the
  bias is single - which is why the projections compute `x @ w.T + b` on the doubled footing and not,
  as it first appears, `x @ w.T + 2b`;
* LayerNorm's final doubling, which is never cancelled, puts the next layer back on the same footing.

The practical consequence is that **layer 0 has to be entered on that footing too**: the embeddings
are encrypted at `2 * hidden`. Feeding them at 1x makes every stage individually plausible and the
layer as a whole wrong by about 50% RMS.

GELU is the single exception, and it has to be. The composite is only valid for an argument in
`[-1, 1]`, so it needs the *actual* pre-activation over 64, not a multiple of it.
`GeluMixin.gelu(x, carrier)` therefore divides the tanh's argument by the carrier while keeping the
linear factor at 64, which lands the result back on the doubled footing. That costs one level (14
rather than 13) and is a deliberate deviation from `he.py`.

## The softmax, calibrated against a real model

Three things had to be right before stage 07 worked on real activations. Each failed quietly - the
chain ran and produced plausible-looking numbers.

**The key scaling is 1/64, not THOR's 1/512.** A softmax is not scale-invariant, and `he_softmax(x)`
approximates `softmax(x)` - which is what `test_stage9` pins down, and the units THOR's window is in
(its narrow `[-27.2, 21.7]` is the range of a BERT-base attention score). Counting the factors:
q and k are each doubled, stage 06's own masks contribute a half, and `stage_07_softmax`'s bootstrap
fold doubles again, so `he_softmax` sees `4 * (q.k) * scale`. THOR writes 1/512, which its own
softmax must compensate elsewhere; 1/64 is what this port's stages measure.

**The attention mask is a list, one plaintext per score ciphertext.** The scores are held as
diagonals: slot `(group, tau, block)` of ciphertext `ct` carries key position
`(ct*pack + group + tau) mod dim`. A mask keyed on the slot alone can only say "this *query* is
padding"; the denominator needs "this *key* position is padding". With the wrong mask the denominator
sums the exponential over all 128 key positions and the softmax is off by a third.

**The window has to be calibrated on the denominator, not on the scores.** `softmax.calibrate`
rejects a window whose row sums exceed 1, because the Goldschmidt iteration needs a denominator in
`[epsilon, 1]`. The `thor-openfhe` draft reaches the same conclusion from the other direction: a
per-element range is not enough, the row sum has to be calibrated with it.

With all three right, stage 07 reproduces the true softmax to `2.3e-6` relative, at a best-fit scale
of `1.0023` - which is not error but THOR's own `int(1/(2*delta)) + 1`, a fixed slight
over-normalisation that the `thor-openfhe` draft measures independently as `1.0022567`.

## One layer, end to end, on a real checkpoint

`textattack/bert-base-uncased-MRPC`, a real MRPC pair, layer 0 encrypted and the rest in plaintext:

| stage | best-fit scale | relRMSE |
|---|---:|---:|
| query, value | 2.0000 | 2.9e-7 |
| scores (06) | 0.5000 | 2.9e-7 |
| softmax (07) | 1.0023 | 2.3e-6 |
| attention dense (10) | 2.0044 | 3.0e-4 |
| LayerNorm 1 (11) | 1.9998 | 6.5e-4 |
| intermediate (12) | 0.0313 | 4.8e-4 |
| GELU (13) | 2.0001 | 2.3e-3 |
| output dense (14) | 2.0000 | 2.2e-3 |
| LayerNorm 2 (16) | 1.9998 | 1.1e-3 |

The layer output is faithful to `1.05e-3` relative RMSE, max absolute `3.7e-3` - the same order as the
`thor-openfhe` draft's v21 baseline (RMSE ~1.3e-3, max ~9e-3). Every number above is exact arithmetic
under the FIXEDMANUAL contract, so it measures the *schedule and algebra*; CKKS noise is what a GPU run
adds on top.

## Rotation keys do not fit, and truncation does not save them

One layer uses 210 distinct rotation indices. At N=2^16, depth 50 and dnum=4 a rotation key is
248 MiB, so that is **51 GiB** - on a 32 GB card, alongside 22 GiB of bootstrap plaintexts and keys.

Level truncation, which exists for exactly this, only takes it to 40 GiB. The level histogram says
why: 140 of the 210 keys are used at level 40-49, because stages 01-05 run *before* the first
bootstrap on a still-full ciphertext, and a key whose plan level reaches `L` is not truncated at all.
So `0 truncated` in the GPU log is correct behaviour, not a broken feature.

What does fit is decomposing every rotation into powers of two. Rotations compose additively in slot
space and cost no levels, so 15 keys (`log2(slot_count)`) reach every index:

| | keys | untruncated | truncated | rotations per layer |
|---|---:|---:|---:|---:|
| one key per index | 210 | 50.9 GiB | 40.3 GiB | 1802 |
| `binary_rotations` | 15 | 3.6 GiB | 3.3 GiB | 8138 |

4.5x the rotations for 14x less key memory, and the decrypted values are bit-identical - a rotation is
an exact permutation, so splitting it changes nothing but the key-switching noise. It is off by
default (`Stages.binary_rotations`, `bench --binary-rotations`) because it is a trade worth making
only when the keys do not otherwise fit.

The middle of that curve is unexplored and probably where the answer is: most of the 1802 rotations
are a handful of indices (stage 02's `group_size`, `rotate_internal`'s small deltas), so keeping
dedicated keys for those and decomposing the long tail should recover most of the time for a few GiB.
`Stages.rotation_steps` is the one place that would change.

## An inserted refresh, and the depth it buys

THOR's own schedule needs a **depth of 52** to run one layer, and at N=2^16 that does not fit a 32 GiB
card. The reason is a single long chain. Measured from the softmax's own bootstrap down to the rescale
in front of GELU's:

| segment | levels | cumulative |
|---|---:|---:|
| tail of the softmax | 14 | 14 |
| attention context | 2 | 16 |
| attention dense | 3 | **19** |
| LayerNorm | 14 | 33 |
| feed-forward expansion | 3 | 36 |
| the rescale in front of GELU's bootstrap | 1 | 37 |

Thirty-seven levels with nothing refreshing the data path in between - LayerNorm bootstraps its
*statistic*, not the value. One extra refresh splits it, and the split that minimises the longer half
is right after stage 10: 19 before, 18 after.

`LayerNormStages.refresh` is that refresh, and `EncoderLayer(refresh_after_dense=True)` inserts it. It
folds pairs into complex ciphertexts first, so eight ciphertexts cost four bootstraps, and it is
**exactly value-neutral**: the halving in front of the bootstrap and the doubling in `x + conj(x)`
cancel, the same way they do in stage 13. Measured on a real MRPC sample, the logits come out
bit-identical to a run without it.

| | minimum depth | one layer at N=2^16, dnum 4, level budget (3,3) |
|---|---:|---:|
| THOR's schedule | 52 | 35.3 GiB - does not fit |
| `refresh_after_dense` | **34** | **27.4 GiB, 4.6 GiB spare** |

Four extra bootstraps out of eighteen, for eighteen levels of depth. It is off by default because it
is a deviation from `he.py`, and it is the only thing so far that puts a layer inside the card.

## Open questions

* **The he.py score bug.** Confirmed against the linear algebra, not against THOR's own outputs (the
  draft repo is not in this checkout). Re-check when a real MRPC sample runs end to end.
* **Layer >= 1 input.** `stage_01_complexify_x` has a second branch, for the `2 * n_input_ciphertexts`
  real/imaginary ciphertexts a previous layer produces, which folds in a rotation by `n_blocks // 2`.
  It is ported but only reachable once stage 16 exists, so it is untested.
* **`temp` (the real/imaginary split) is unused by stages 01-05** and its extra rescale is charged
  where it happens rather than where THOR charges it. Re-check when stage 11 lands.
* **The level budget is not yet the real one.** The feed-forward tests run with a bootstrap level of
  60 so that the *schedule* is what is tested rather than the budget. Stages 12-14 cost 2 + 13 + 2
  levels; whether that fits THOR's actual bootstrap output is a T5 question.
* **The pooler tanh window.** The fit is valid to a pre-activation of about 10. Real BERT pooler
  pre-activations have not been measured here - the tests use random weights - so this needs checking
  against a real checkpoint before T5's end-to-end numbers mean anything.
