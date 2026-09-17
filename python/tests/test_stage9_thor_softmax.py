"""Stage 9: THOR's softmax, and attention end to end (stages 06 -> 07 -> 08).

Unlike the stages before it, this one is *approximate by construction*: there is no exponential and no
division in CKKS, so THOR builds a low-temperature softmax out of a degree-15 polynomial and sharpens
it with Goldschmidt iterations. The tests therefore assert accuracy rather than equality, and they
assert it against the true softmax - the thing the network needs - not against THOR's own output.

The whole chain runs on the numpy engine at THOR's production geometry, which is exact arithmetic under
a strict FIXEDMANUAL level and scale contract. So a failure here is a *schedule* or *algebra* error,
never CKKS noise; the noise is what the GPU run adds on top.
"""
import numpy as np
import pytest

from thorfhe import THOR_BERT, ClearEngine, block_diagonal_masks
from thorfhe.attention import attention_rotate_masks, ccmm_masks, make_copies_masks, transpose_masks
from thorfhe.layer import DEFAULT_SOFTMAX_SCALE, SOFTMAX_SCALES
from thorfhe.numeric import DivisionMixin
from thorfhe.softmax import Softmax, calibrate

G = THOR_BERT
DEPTH = 60


@pytest.fixture(scope="module")
def mask_families():
    return (block_diagonal_masks(G), transpose_masks(G), make_copies_masks(G),
            attention_rotate_masks(G), ccmm_masks(G))


def used_slots():
    return ((np.arange(G.slot_count) % G.n_slot) < G.n_blocks).astype(float)


def build(mask_families, depth=DEPTH):
    (low, high), transpose, copies, attention, ccmm = mask_families
    engine = ClearEngine(G, depth=depth, bootstrap_level=depth)
    stages = Softmax(engine, G, masks=low, complement_masks=high, transpose=transpose, copies=copies,
                     attention=attention, ccmm=ccmm, ones=engine.encrypt(used_slots()))
    return engine, stages


def encode_score_diagonals(engine, scores):
    """The layout stage 06 leaves the scores in: `out[ct][g, tau, b] = S[b][tau, (ct*pack+g+tau) % dim]`."""
    out = []
    for ct in range(2 * G.n_output_ciphertexts):
        msg = np.zeros(G.slot_count, dtype=complex)
        for group in range(G.pack):
            diagonal = ct * G.pack + group
            for tau in range(G.dim):
                for block in range(G.n_blocks):
                    msg[G.slot(group, tau, block)] = scores[block, tau, (diagonal + tau) % G.dim]
        out.append(engine.encrypt(msg))
    return out


def decode_broadcast_diagonals(engine, diagonals):
    """Read a (heads, dim, dim) matrix out of the broadcast diagonals stage 07 produces."""
    out = np.zeros((G.n_blocks, G.dim, G.dim))
    for diagonal in range(G.dim):
        slots = np.real(engine.decrypt(diagonals[diagonal]))
        for tau in range(G.dim):
            for block in range(G.n_blocks):
                out[block, tau, (diagonal + tau) % G.dim] = slots[G.slot(0, tau, block)]
    return out


def true_softmax(scores):
    shifted = np.exp(scores - scores.max(axis=2, keepdims=True))
    return shifted / shifted.sum(axis=2, keepdims=True)


def test_softmax_matches_the_real_thing(mask_families):
    """THOR's construction, measured against softmax itself."""
    rng = np.random.default_rng(61)
    scores = rng.uniform(-8, 8, (G.n_blocks, G.dim, G.dim))

    engine, stages = build(mask_families)
    out = stages.he_softmax(encode_score_diagonals(engine, scores),
                            [used_slots()] * 2 * G.n_output_ciphertexts, **Softmax.NARROW)
    assert out.shape == (G.dim,)

    got = decode_broadcast_diagonals(engine, out)
    want = true_softmax(scores)

    # output_alpha = 0.01 is the precision the last Goldschmidt is asked for; the rows come out well
    # inside it, and the individual weights an order of magnitude better again
    sums = got.sum(axis=2)
    assert np.abs(sums - 1.0).max() < Softmax.NARROW["output_alpha"]
    assert np.abs(got - want).max() < 5e-3
    assert np.abs(got - want).mean() < 1e-4


def test_recentring_the_fit_buys_iterations_without_moving_the_softmax(mask_families):
    """Lowering the centre lifts the denominator, and the softmax it produces has to stay put.

    `he_exp` evaluates the fit at `(score - centre) / 32`, so every denominator term carries a common
    factor of `exp(-centre / 2)`. Lowering the centre therefore multiplies the whole denominator and
    leaves the ratio between its ends alone - which is worth doing, because the Goldschmidt iteration
    costs one level per iteration and the count comes from `inv_epsilon`, the *smallest* denominator.
    It is a constant folded into a plaintext, so it costs nothing.

    What it is not free of is the polynomial: the fit is a minimax approximation to an exponential
    over a finite domain, and a lower centre evaluates it further out. That is the thing this test
    exists to pin - not the level saving, which is arithmetic, but that the softmax coming out the
    far end is still a softmax. `Softmax.LAYERS` applies this per layer, where it is worth 37 levels
    across the twelve.
    """
    rng = np.random.default_rng(63)
    scores = rng.uniform(-8, 8, (G.n_blocks, G.dim, G.dim))
    masks = [used_slots()] * 2 * G.n_output_ciphertexts

    midpoint = calibrate(scores)
    lifted = calibrate(scores, target=0.5)
    assert lifted["shift"] < midpoint["shift"], "target=0.5 should have walked the centre down"
    assert calibrate(scores, shift=midpoint["shift"]) == midpoint, (
        "passing the centre it would have chosen must change nothing")

    cost = {name: DivisionMixin.goldschmidt_iterations(p["inv_epsilon"], Softmax.internal_alpha / 10)
            for name, p in (("midpoint", midpoint), ("lifted", lifted))}
    assert cost["lifted"] < cost["midpoint"], (
        f"recentring bought nothing: {cost['midpoint']} iterations -> {cost['lifted']}")

    want = true_softmax(scores)
    errors = {}
    for name, parameters in (("midpoint", midpoint), ("lifted", lifted)):
        engine, stages = build(mask_families)
        got = decode_broadcast_diagonals(
            engine, stages.he_softmax(encode_score_diagonals(engine, scores), masks, **parameters))
        assert np.abs(got.sum(axis=2) - 1.0).max() < parameters["output_alpha"], (
            f"{name}: the rows are not distributions")
        errors[name] = float(np.abs(got - want).max())

    # Measured on this input: 6.04e-4 at the midpoint over nine iterations, 6.44e-4 lifted over four.
    # Flat, which is the claim - the fit is being evaluated 13.5 units further out for five fewer
    # levels and gives up nothing for it. On BERT's own score distributions it comes out ahead
    # instead, most of all on layer 2's wider polynomial (3.2e-3 -> 5.4e-4); a uniform [-8, 8] has no
    # such structure to recover, so flat is what it should show here.
    assert errors["lifted"] < 5e-3
    assert errors["lifted"] <= 2 * errors["midpoint"], (
        f"recentring cost accuracy: {errors['midpoint']:.3g} -> {errors['lifted']:.3g}")


def test_a_padded_input_leaves_the_padding_rows_at_zero(mask_families):
    """Padding query rows must come out empty, not merely unread.

    `padding_mask` masks both sides, so a padding query's denominator is the sum of an exponential
    over no keys at all: exactly zero. Nothing downstream reads those rows - but `he_inv` is handed
    them along with the real ones, and `1/0` does not stay small. It grows through every Goldschmidt
    iteration and then goes through a bootstrap, which recovers a message only well inside `q0/Delta`.
    The device reported an inverse denominator whose *median* was 1.9e159, which is what that looks
    like: with 30 real tokens, 98 of the 128 rows are padding.

    So `ones` - the indicator the iteration starts from - is restricted to the real rows, and those
    rows stay at zero all the way through. That is also what keeps `_check_inversion_range` honest:
    a zero denominator is below any `epsilon`, so without this the check fires on every padded input.
    """
    from thorfhe.layer import EncoderLayer

    tokens = 30
    rng = np.random.default_rng(64)
    scores = rng.uniform(-8, 8, (G.n_blocks, G.dim, G.dim))

    engine, stages = build(mask_families)
    masks = EncoderLayer(engine).padding_mask(tokens)
    parameters = calibrate(scores[:, :tokens, :tokens], target=0.5)

    out = stages.he_softmax(encode_score_diagonals(engine, scores), masks, **parameters)
    got = decode_broadcast_diagonals(engine, out)

    padding_rows = got[:, tokens:, :]
    assert np.abs(padding_rows).max() == 0.0, (
        f"padding rows carry up to {np.abs(padding_rows).max():.3g}; they have to be empty, because "
        f"whatever is in them is bootstrapped")

    real = got[:, :tokens, :]
    assert np.abs(real[:, :, tokens:]).max() == 0.0, "padding keys leaked into a real row"
    assert np.abs(real[:, :, :tokens].sum(axis=2) - 1.0).max() < parameters["output_alpha"]
    want = true_softmax(scores[:, :tokens, :tokens])
    assert np.abs(real[:, :, :tokens] - want).max() < 5e-3


def test_the_wide_path_wants_the_same_score_the_narrow_one_does(mask_families):
    """`he_exp2` with `l = 4` has to reproduce softmax of the score it is handed, exactly as `he_exp1`
    with `l = 2` does. That is what makes `SOFTMAX_SCALES` one constant rather than one per layer.

    The two paths differ in every intermediate quantity - a looser fit over twice the range, half the
    slope, an extra factor of 128, and one more squaring - and they are supposed to cancel to the same
    exponent. `thorfhe.softmax`'s module docstring derives that they do; nothing checked it, and layer
    2 carried THOR's `softmax_scale / 2` on the strength of the derivation not being trusted. In these
    units that halves layer 2's score, which is a doubling of its temperature: measured against BERT's
    own softmax on the real checkpoint, 5.2e-4 handed the score and 0.61 handed half of it.

    A softmax that is wrong by 0.61 is wrong the way the four-times-too-large `softmax_scale` was, and
    for the same reason - a constant nobody ran.
    """
    rng = np.random.default_rng(65)
    scores = rng.uniform(-32, 32, (G.n_blocks, G.dim, G.dim))   # layer 2's own range, so `wide` holds
    parameters = calibrate(scores, l=4, target=0.5)
    assert parameters["max_x"] >= 30, "this input should have selected the wide polynomial"
    assert parameters["l"] == 4, "the wide path takes one more squaring than the narrow one"

    engine, stages = build(mask_families)
    out = stages.he_softmax(encode_score_diagonals(engine, scores),
                            [used_slots()] * 2 * G.n_output_ciphertexts, **parameters)
    got = decode_broadcast_diagonals(engine, out)

    assert np.abs(got.sum(axis=2) - 1.0).max() < parameters["output_alpha"]
    error = float(np.abs(got - true_softmax(scores)).max())
    # 6.8e-3 measured here. A uniform [-32, 32] is harsher than the real distribution - BERT's own
    # layer 2 gives 2.9e-3 at THOR's centre and 5.2e-4 at the one `LAYERS` uses - and the wide fit is
    # the loose one. The bound is set to separate *that* from a temperature error, which is two
    # orders away: the same path handed half the score is off by 0.61.
    assert error < 1e-2, (
        f"the wide path is off by {error:.3g} on the score it was handed; if it wanted half or twice "
        f"that score this is where it would say so, at around 0.6")


def test_the_layer_table_is_the_one_calibrate_produces():
    """`Softmax.LAYERS` is measured output, so what it has to satisfy is its own contract.

    Twelve rows because the constants are per layer; layer 2 alone on the wide polynomial, because
    its scores are twice as wide; and every centre below THOR's, because that is the direction that
    lifts the denominator. The numbers themselves come from `calibrate(scores, target=0.5)` on the
    real checkpoint and cannot be re-derived without it - what is checked here is that they are
    internally consistent and that nobody has half-edited the table.
    """
    assert sorted(Softmax.LAYERS) == list(range(12))
    for index, row in Softmax.LAYERS.items():
        base = Softmax.WIDE if index == 2 else Softmax.NARROW
        assert (row["max_x"] >= 30) == (index == 2), f"layer {index}: wrong polynomial"
        assert row["l"] == base["l"] and row["n"] == base["n"], f"layer {index}: temperature moved"
        assert row["shift"] < (row["min_x"] + row["max_x"]) / 2, (
            f"layer {index}: centre {row['shift']} is not below the midpoint, so it lifts nothing")
    # The level cost is the reason the table exists, so it is the thing to pin. Ten is layer 8,
    # whose smallest and largest denominators differ by 1.2e-4 whatever the centre; the rest sit at
    # five to nine. Leaving the centres at THOR's midpoint costs 123 for the same twelve layers -
    # and that is with `inv_epsilon` set honestly, which is where this started: layer 8's measured
    # minimum floors to 2^-15, *below* NARROW's 2^-14, so the untuned table was not merely expensive
    # but wrong, and `he_inv` saturates rather than failing when it is.
    cost = [DivisionMixin.goldschmidt_iterations(row["inv_epsilon"], Softmax.internal_alpha / 10)
            for row in Softmax.LAYERS.values()]
    assert max(cost) <= 10, f"a layer wants {max(cost)} Goldschmidt iterations, i.e. that many levels"
    assert sum(cost) == 86, f"the table costs {sum(cost)} levels across the twelve layers, not 86"


def test_softmax_is_a_distribution_on_a_peaked_input(mask_families):
    """A near-one-hot row must come out near-one-hot: the sharpening has to actually sharpen.

    The background scores span the same range as the test above, and that is not cosmetic. NARROW's
    `inv_epsilon` states the window the Goldschmidt iteration is set up for, and a background an
    order of magnitude tighter puts the denominator below it - this test used to do exactly that,
    with a median of 2.5e-4 against a bound of 4.9e-4, and passed anyway because exact arithmetic
    and the later refinements recover from a saturated first inverse. The device has neither
    luxury. `he_inv` now refuses the out-of-range denominator, which is what turned this up.
    """
    rng = np.random.default_rng(62)
    scores = rng.uniform(-8, 8, (G.n_blocks, G.dim, G.dim))
    scores[0, 0, 7] = 15.0  # one clear winner

    engine, stages = build(mask_families)
    out = stages.he_softmax(encode_score_diagonals(engine, scores),
                            [used_slots()] * 2 * G.n_output_ciphertexts, **Softmax.NARROW)
    got = decode_broadcast_diagonals(engine, out)

    assert got[0, 0].argmax() == 7
    assert got[0, 0, 7] > 0.9
    assert np.abs(got - true_softmax(scores)).max() < 1e-2


def test_attention_end_to_end(mask_families):
    """Stages 06 -> 07 -> 08: from Q, K, V to softmax(Q K^T) V, all in the packed domain.

    The scale of q and k is not arbitrary. THOR's softmax parameters are a calibration: the polynomial
    covers `[min_x, max_x]` and, more tightly, `inv_epsilon` asserts that the denominator lands in
    `[inv_epsilon, 1]` - which for the narrow parameters is a range of only three decades. That is what
    the `softmax_scale = 1/512` folded into the key projection is for. Scores an order of magnitude
    smaller push the denominator below `inv_epsilon` and the Goldschmidt iteration returns nonsense, so
    the ranges are asserted here rather than left to chance.
    """
    rng = np.random.default_rng(63)
    q = rng.normal(size=(G.dim, G.features)) * 0.75
    k = rng.normal(size=(G.dim, G.features)) * 0.75
    v = rng.normal(size=(G.dim, G.features)) * 0.1

    engine, stages = build(mask_families)

    def encode_qkv(y):
        out = []
        for ct in range(G.n_output_ciphertexts):
            msg = np.zeros(G.slot_count, dtype=complex)
            for group in range(G.pack):
                diagonal = ct * G.pack + group
                for token in range(G.dim):
                    for block in range(G.n_blocks):
                        msg[G.slot(group, token, block)] = y[token, G.n_out * block
                                                            + (diagonal + token) % G.n_out]
            out.append(engine.encrypt(msg))
        return np.array(out, dtype=object)

    per_head = np.stack([q[:, G.n_out * h:G.n_out * (h + 1)] @ k[:, G.n_out * h:G.n_out * (h + 1)].T
                         for h in range(G.n_blocks)])
    # THOR's constants were calibrated on real activations; these scores are synthetic, so calibrate
    # for them the same way rather than hoping they land in the same window. The factor of two is the
    # doubling stage_07_softmax's bootstrap fold applies - see its docstring.
    parameters = calibrate(2 * per_head)

    scores = stages.stage_06_attention_score(encode_qkv(q), encode_qkv(k))
    weights = stages.stage_07_softmax(scores, [used_slots()] * 8, layer_index=0, parameters=parameters)
    context = stages.stage_08_attention_context(encode_qkv(v), weights)

    attention = true_softmax(per_head)
    want = np.zeros((G.dim, G.features))
    for block in range(G.n_blocks):
        columns = slice(G.n_out * block, G.n_out * (block + 1))
        want[:, columns] = attention[block] @ v[:, columns]

    got = np.zeros((G.dim, G.features))
    for ct in range(2):
        slots = np.asarray(engine.decrypt(context[ct]), dtype=complex)
        for group in range(G.pack):
            for tau in range(G.dim):
                for block in range(G.n_blocks):
                    value = slots[G.slot(group, tau, block)]
                    real = (ct * G.pack + group + tau) % G.n_out
                    imag = ((ct + 2) * G.pack + group + tau) % G.n_out
                    got[tau, G.n_out * block + real] = value.real
                    got[tau, G.n_out * block + imag] = value.imag

    error = np.abs(got - want).max() / np.abs(want).max()
    assert error < 0.02, f"attention output is off by {error:.4f} relative"


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_score_refresh_scale_changes_only_what_stage_07_bootstraps(mask_families, scale):
    """Divide the scores before the refresh, multiply them back after: nothing downstream moves.

    Stage 07 is one of two sites that hand their bootstrap an unhalved value, and on the real
    checkpoint it hands over 1.99 - unremarkable against q0/Delta = 32 and 99.5% of the bound
    EasyFHE's parameters would give. The division goes into the key projection, where it reaches the
    scores and nothing else, and the multiplication back is by an integer, which costs neither a
    level nor a scale degree. So `he_softmax` sees the scores it always saw.

    The route not taken is recorded in `Softmax.score_refresh_scale`: really halving the scores and
    taking the temperature back with another squaring works arithmetically - measured at 2.1e-3
    against the true softmax, against 6.5e-4 unscaled - but squaring the numerators collapses the
    denominator from 1.2e-4 to 5.8e-11, and Goldschmidt goes from 9 iterations to 19. Ten levels
    against a budget with one to spare.
    """
    rng = np.random.default_rng(67)
    scores = rng.uniform(-8, 8, (G.n_blocks, G.dim, G.dim))

    results, peaks = {}, {}
    for factor in (1.0, scale):
        seen = []

        class Probe(ClearEngine):
            def bootstrap(self, ct, keep_levels=None):
                seen.append(float(np.max(np.abs(ct.slots))))
                return super().bootstrap(ct, keep_levels)

        (low, high), transpose, copies, attention, ccmm = mask_families
        engine = Probe(G, depth=DEPTH, bootstrap_level=DEPTH)
        stages = Softmax(engine, G, masks=low, complement_masks=high, transpose=transpose,
                         copies=copies, attention=attention, ccmm=ccmm,
                         ones=engine.encrypt(used_slots()))
        stages.score_refresh_scale = factor
        encoded = encode_score_diagonals(engine, scores / factor)
        out = stages.stage_07_softmax(list(encoded), [used_slots()] * 8, layer_index=0,
                                      parameters=Softmax.NARROW)
        results[factor] = decode_broadcast_diagonals(engine, out)
        peaks[factor] = max(seen)

    difference = float(np.max(np.abs(results[1.0] - results[scale])))
    assert difference < 1e-9, (
        f"scaling the refresh moved the softmax by {difference:.3g}; it must not move at all")
    assert peaks[1.0] / peaks[scale] == pytest.approx(scale, rel=1e-6), (
        f"and it has to scale what is bootstrapped: {peaks[1.0]:.4g} -> {peaks[scale]:.4g}")


def test_stage_06_hands_the_softmax_berts_own_score(mask_families):
    """`he_softmax` has to see BERT's attention score, and `SOFTMAX_SCALES` is the constant for it.

    A softmax is not scale-invariant, so this is not a calibration that can absorb a uniform factor -
    and nothing else in the suite can catch one either: the per-stage fidelity check rescales by the
    best fit, `calibrate` fits its window to whatever scores it is given, and every stage test builds
    its own synthetic input. So the factor is *run* here rather than asserted, end to end from the
    amplitude a layer is actually entered at down to the number the exponential receives.

    Three factors have to cancel, and the third is the one that was missed:

    * the projections give `s * (x @ W.T) + 2b` (`test_qkv_computes_xw_plus_bias`), i.e. `s` times
      BERT's own `q` and `k` - so `s` enters the score **twice**;
    * stage 06 carries the product exactly (`test_attention_score_is_exactly_q_k_transpose`);
    * `stage_07_softmax`'s bootstrap fold doubles it once more.

    Counting only the last two gives `scale = 1/16`, which is what this was for a day. It puts four
    times BERT's score into the exponential, and since that is an exponential the denominator - which
    `he_inv` needs inside `[epsilon, 1]` - goes from 9e-4 to 534 on MRPC's first validation row.
    """
    from thorfhe import encode_bias, encode_weight
    from thorfhe.attention import AttentionScore
    from thorfhe.encoding import encode_activations
    from thorfhe.layer import ACTIVATION_SCALE

    rng = np.random.default_rng(31)
    query_weight = rng.normal(size=(G.features, G.features)) * 0.04
    query_bias = rng.normal(size=G.features) * 0.1
    key_weight = rng.normal(size=(G.features, G.features)) * 0.04
    key_bias = rng.normal(size=G.features) * 0.1
    x = rng.normal(size=(G.dim, G.features)) * 0.5

    (low, high), transpose, copies, attention, ccmm = mask_families
    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    stages = AttentionScore(engine, G, masks=low, complement_masks=high, transpose=transpose,
                            copies=copies, attention=attention, ccmm=ccmm)

    # the two lines `encode_layer` uses for the query and the key, and nothing else: encoding a whole
    # layer here would build the 3072x768 feed-forward plaintexts, which is gigabytes for no gain
    packed = np.array([engine.encrypt(message)
                       for message in encode_activations(G, ACTIVATION_SCALE * x)], dtype=object)
    _, complexified = stages.stage_01_complexify_x(packed, layer_index=0)
    rotated = stages.stage_02_make_rotated_copies(complexified)
    scores = stages.stage_06_attention_score(
        stages.stage_03_query(rotated, encode_weight(G, query_weight), encode_bias(G, query_bias)),
        stages.stage_04_key(rotated, encode_weight(G, key_weight, scale=DEFAULT_SOFTMAX_SCALE),
                            encode_bias(G, key_bias, scale=DEFAULT_SOFTMAX_SCALE)))

    # BERT's own score from the same weights: q k^T / sqrt(head_dim), per head
    query = x @ query_weight.T + query_bias
    key = x @ key_weight.T + key_bias
    bert = np.stack([query[:, h * G.n_out:(h + 1) * G.n_out]
                     @ key[:, h * G.n_out:(h + 1) * G.n_out].T for h in range(G.n_blocks)])
    bert = bert / np.sqrt(G.n_out)

    # Checked where the score is largest: that is the slot the polynomial's range has to accommodate,
    # and the one an overstated score overflows first.
    head, token, other = np.unravel_index(np.abs(bert).argmax(), bert.shape)
    diagonal = (int(other) - int(token)) % G.dim
    held = np.real(np.asarray(engine.decrypt(scores[diagonal // G.pack])))[
        G.slot(diagonal % G.pack, int(token), int(head))]

    # what stage 07 hands he_softmax is twice what stage 06 holds
    ratio = 2 * held / float(bert[head, token, other])
    assert ratio == pytest.approx(1.0, rel=1e-9), (
        f"he_softmax would see {ratio:.4f} times BERT's score at head {head}, "
        f"({token}, {other}); softmax_scale {DEFAULT_SOFTMAX_SCALE} is off by that factor")
    assert SOFTMAX_SCALES == {}, (
        f"a layer has its own softmax scale ({SOFTMAX_SCALES}); both polynomial paths land on the "
        f"same exponent, so every layer wants BERT's score and a per-layer factor is a temperature "
        f"error - see test_the_wide_path_wants_the_same_score_the_narrow_one_does")
