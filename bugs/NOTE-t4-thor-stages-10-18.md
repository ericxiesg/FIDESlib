# NOTE (T4): stages 10-18 are ported, and what that means for the GPU side

T4 is done in `python/thorfhe/`: attention dense, LayerNorm, the feed-forward block, GELU, the pooler
and the classifier. Everything is verified slot-by-slot against numpy on the strict clear engine
(`ClearEngine`, which enforces FIXEDMANUAL levels and scales), so what follows is the *schedule* and
the *layout*, not measured hardware numbers. Nothing here has run on a GPU yet.

| stage | what | accuracy vs numpy |
|---|---|---|
| 10 attention dense | `ctx @ W.T`, folded to the 6-block form | 7.2e-16 |
| 11 / 16 LayerNorm | `2 * (gamma * (x - mean) / sqrt(var + eps) + beta)` | 3.3e-6 |
| 12 intermediate dense | `x @ W1.T / 64`, 768 -> 3072 | exact |
| 13 GELU | elementwise, degree 31 then 27 | 4.7e-5 abs |
| 14 output dense | `h @ W2.T`, 3072 -> 768 | exact; 12->13->14 within 2e-4 |
| 15 prepare LayerNorm | the residual, bootstrapped | exact |
| 17 / 18 pooler + classifier | `w_cls @ tanh(W @ cls + b) + b_cls` | 1.4e-3 |

## 1. There is a second mask family, and it is modulo 8

THOR's `rotate_internal` has two block-diagonal modes. `block_diag_1` wraps twelve blocks in a token's
sixteen slots; `block_diag_2` wraps **six blocks on a stride of eight**, i.e. its masks are `slot % 8 < i`
for `i` in 1..7, not `slot % 16`. It is used by stages 12, 14 **and** the pooler.

If the GPU-side mask plaintexts are generated from a single `n_slot`-based formula, half of them will
be silently wrong - the products still typecheck, they just sum the wrong blocks. `pre_encode_masks`
in THOR writes the two families to separate directories (`rotate_internal/block_diag_1/{1..15}` and
`rotate_internal/block_diag_2/{1..7}`); the port's `block_diagonal_masks(g, stride, width)` produces
either.

The first port attempt modelled this as a second geometry (256 half-tokens of 8 slots). Do not do
that: the *encoder* indexes by token, and re-indexing the packing overflows `group_size`. The window
belongs to the multiply, so `rotate_internal` and `pcmm` take it as an argument.

## 2. Rotation indices and levels for the new stages

Measured off the clear engine, which records every rotation a stage actually performs. These feed
`SetRotationKeyLevels` and therefore the key budget. Signs are in fideslib's direction (slot `i` moves
to `i - delta`), i.e. already negated from how they read in `he.py`.

```
stage 12       12 indices:  +-1 +-2 +-3 +-4 +-5, -8, 2048
stage 14       12 indices:  +-1 +-2 +-3 +-4 +-5, +8, 2048
stages 17+18   28 indices:  +-1 +-2 +-3 +-4 +-5,
                            +-16 +-32 +-64 +-128 +-256 +-512 +-1024,
                            2048 4096 8192 16384
```

The `+-1..+-5` are `block_diag_2`'s `rotate_internal`; `+-8` is the window fold; `2048` is `group_size`;
`4096/8192/16384` are the pooler's `interval_sum` over the groups; `16..1024` is the CLS broadcast and
the classifier's token fold. Stages 01-05 needed `{2048, 7..11, -5..-1}`, so the union is still small -
the `+-1..+-5` overlap and only `+-8` and the power-of-two chain are new.

## 3. The level budget is the open question for T5

Stages 12-14 cost **2 + 13 + 2 = 17 levels** from whatever the bootstrap leaves. GELU is 13 of those:
THOR composes a degree-31 with a degree-27 because a single minimax fit of `tanh` over the
pre-activation range is hopeless. The tests deliberately run with `bootstrap_level = 60` so that the
schedule is what is being tested rather than the budget - **whether 17 levels actually fit after a real
bootstrap is unverified and is the first thing T5 has to answer.**

The pooler is cheaper: two degree-15 polynomials with a bootstrap on either side.

## 4. Two scaling conventions that will look like bugs if you meet them cold

* **`GELU_SCALE = 64` is load-bearing.** The ciphertext carries the pre-activation *divided by 64*
  (stage 12's weight is encoded with `scale = 1/64`) so that the composite's argument lands in
  `[-1, 1]`, where the fit is valid. Outside that the degree-31 inner polynomial stops being bounded.
  The pooler's equivalent is `/40`; that one plateaus instead of diverging.
* **Stages 13 and 15 look identical and are not.** Both merge two real ciphertexts into one complex
  one, bootstrap once instead of twice, and split back with `x + conj(x)` / `i * (conj(x) - x)`. That
  split doubles. Stage 13 multiplies by `1/2` first and so returns what it was given; stage 15 does
  not, and hands LayerNorm **twice** the residual on purpose - `variance_window` accepts four times the
  variance for exactly the variants stage 16 routes to.

Related: the pooler bias *is* halved (`encode_b_pooler`), so the closing fold gives `+b`. The QKV bias
is *not*, so those stages compute `x @ w.T + 2b`. Both are THOR's, both are reproduced.

## 5. What would be useful from the V100

Nothing in T4 is blocked on hardware, so this is not a request to debug anything. When there is spare
time, the two measurements that would most change T5's plan are:

1. **What level does a bootstrap actually leave** under the current parameter set, and does 17 fit.
2. **How long the light-plaintext expansion takes for a (2, 8, 6, 64) weight** - stages 12 and 14 each
   need 6144 of them per layer, which is the largest single weight in the model, and the CPU encoder
   takes about a minute per weight in numpy alone.

The layout formulas the GPU path has to reproduce are in `docs/thor_port.md`; the expansion's is

```
out[rep][ct][group, t, s] = z[t, 128 * block + (ct*pack + group + t) mod 128]
block = 12 * rep + FF_SLOT_INDICES.index(s),  FF_SLOT_INDICES = [0..5, 8..13]
```

and the pooler's output, which is also what `encode_w_cls` expects, is `slot n_slot*t + block ==
(W @ cls)[dim*block + t]`.
