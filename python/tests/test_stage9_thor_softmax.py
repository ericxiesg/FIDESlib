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
