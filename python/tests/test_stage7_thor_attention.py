"""Stage 7: THOR's attention data movement - the transpose and the diagonal broadcast.

Both are exact permutations of their input (up to one deliberate halving), so they are checked slot by
slot against a formula rather than by an end-to-end error. That formula is the contract the
ciphertext-ciphertext product downstream relies on, and it is stated in `thorfhe/attention.py`.

These run in numpy at THOR's production geometry - 32768 slots - because `he.py` writes this stage as
literal tables for four ciphertexts of sixteen groups and there is no smaller instance to reduce to.
Numpy at that size is seconds; an FHE run would be `log_n = 16` and is not attempted here.
"""
import numpy as np
import pytest

from thorfhe import THOR_BERT, ClearEngine, block_diagonal_masks
from thorfhe.attention import (AttentionStages, attention_rotate_masks, make_copies_masks,
                               require_thor_shape, transpose_masks)

G = THOR_BERT
DEPTH = 20


@pytest.fixture(scope="module")
def masks():
    """Building the mask families is the slow part; they do not depend on the data."""
    return block_diagonal_masks(G), transpose_masks(G), make_copies_masks(G), attention_rotate_masks(G)


def build(masks, depth=DEPTH):
    (low, high), transpose, copies, attention = masks
    engine = ClearEngine(G, depth=depth)
    return engine, AttentionStages(engine, G, masks=low, complement_masks=high,
                                   transpose=transpose, copies=copies, attention=attention)


def encode_qkv_output(engine, y):
    """The layout stages 03-05 leave their result in; the attention stage consumes it."""
    out = []
    for ct in range(G.n_output_ciphertexts):
        msg = np.zeros(G.slot_count, dtype=complex)
        for group in range(G.pack):
            diagonal = ct * G.pack + group
            for token in range(G.dim):
                for block in range(G.n_blocks):
                    msg[G.slot(group, token, block)] = y[token, G.n_out * block + (diagonal + token) % G.n_out]
        out.append(engine.encrypt(msg))
    return np.array(out, dtype=object)


def test_shape_guard_is_explicit():
    """A geometry these tables were not written for must say so, not compute something wrong."""
    from thorfhe import SMALL
    with pytest.raises(NotImplementedError, match="transcribed from he.py"):
        require_thor_shape(SMALL)


def test_transpose_is_the_stated_permutation(masks):
    """`out[ct][g, tau, b] = k[t, f]` for the (ct, g, tau) that attention.py documents - exactly."""
    rng = np.random.default_rng(11)
    k = rng.normal(size=(G.dim, G.features))

    engine, stages = build(masks)
    lower = stages.transpose_upper_to_lower(encode_qkv_output(engine, k))

    expect = np.zeros((4, G.slot_count))
    for token in range(G.dim):
        for feature in range(G.features):
            block, d = divmod(feature, G.n_out)
            diagonal = (token - d) % G.n_out
            tau = d + G.n_out * (((d > token % G.n_out) + (token // G.n_out)) % 2)
            ct, group = divmod(diagonal, G.pack)
            expect[ct, G.slot(group, tau, block)] = k[token, feature]

    got = np.stack([np.real(engine.decrypt(c)) for c in lower])
    assert np.abs(got - expect).max() == 0.0
    assert [engine.level(c) for c in lower] == [DEPTH - 1] * 4


def test_transpose_moves_every_entry_exactly_once(masks):
    """A permutation, not a projection: all 128 x 768 entries survive and nothing else appears."""
    rng = np.random.default_rng(12)
    k = rng.normal(size=(G.dim, G.features))
    engine, stages = build(masks)
    lower = stages.transpose_upper_to_lower(encode_qkv_output(engine, k))
    nonzero = sum(np.count_nonzero(np.abs(engine.decrypt(c)) > 1e-12) for c in lower)
    assert nonzero == G.dim * G.features


def test_make_copies_broadcasts_each_diagonal(masks):
    """`copies[l]` is diagonal `l` of the input, halved, repeated in every group."""
    rng = np.random.default_rng(13)
    q = rng.normal(size=(G.dim, G.features))

    engine, stages = build(masks)
    copies = stages.make_copies(encode_qkv_output(engine, q))
    assert copies.shape == (G.n_out,)

    expect = np.zeros((G.n_out, G.slot_count))
    for diagonal in range(G.n_out):
        for group in range(G.pack):
            for token in range(G.dim):
                for block in range(G.n_blocks):
                    expect[diagonal, G.slot(group, token, block)] = (
                        0.5 * q[token, G.n_out * block + (diagonal + token) % G.n_out])

    got = np.stack([np.real(engine.decrypt(c)) for c in copies])
    assert np.abs(got - expect).max() == 0.0
    # the halves recombine into something real; a stray imaginary part would corrupt the CC product
    assert max(np.abs(np.imag(engine.decrypt(c))).max() for c in copies) == 0.0
    assert engine.level(copies[0]) == DEPTH - 2


def test_interval_sum_folds_the_slot_vector(masks):
    """Every window of `interval` slots ends up holding the sum over all of them."""
    engine, stages = build(masks)
    interval = 2 * G.group_size
    values = np.arange(G.slot_count, dtype=float)
    folded = engine.decrypt(stages.interval_sum(engine.encrypt(values), interval))

    windows = values.reshape(-1, interval)
    assert np.abs(np.real(folded).reshape(-1, interval) - windows.sum(axis=0)).max() < 1e-6


# ---------------------------------------------------------------- the attention score itself
@pytest.fixture(scope="module")
def score_masks():
    from thorfhe.attention import ccmm_masks
    return ccmm_masks(G)


def build_score(masks, score_masks, depth=30):
    from thorfhe.attention import AttentionScore
    (low, high), transpose, copies, attention = masks
    engine = ClearEngine(G, depth=depth)
    return engine, AttentionScore(engine, G, masks=low, complement_masks=high, transpose=transpose,
                                  copies=copies, attention=attention, ccmm=score_masks)


def expected_scores(engine, q, k):
    """`out[ct][group, tau, b] = (Q_b K_b^T)[tau, (ct*pack + group + tau) mod dim]`."""
    per_head = np.stack([q[:, G.n_out * h:G.n_out * (h + 1)] @ k[:, G.n_out * h:G.n_out * (h + 1)].T
                         for h in range(G.n_blocks)])
    expect = np.zeros((2 * G.n_output_ciphertexts, G.slot_count))
    for ct in range(2 * G.n_output_ciphertexts):
        for group in range(G.pack):
            diagonal = ct * G.pack + group
            for tau in range(G.dim):
                for block in range(G.n_blocks):
                    expect[ct, G.slot(group, tau, block)] = per_head[block, tau, (diagonal + tau) % G.dim]
    return expect


def test_attention_score_is_exactly_q_k_transpose(masks, score_masks):
    """Stage 06 computes Q K^T per head - to machine precision, not just approximately."""
    rng = np.random.default_rng(21)
    q = rng.normal(size=(G.dim, G.features)) * 0.1
    k = rng.normal(size=(G.dim, G.features)) * 0.1

    engine, stages = build_score(masks, score_masks)
    out = stages.stage_06_attention_score(encode_qkv_output(engine, q), encode_qkv_output(engine, k))

    assert out.shape == (2 * G.n_output_ciphertexts,)
    got = np.stack([np.asarray(engine.decrypt(o), dtype=complex).real for o in out])
    assert np.abs(got - expected_scores(engine, q, k)).max() < 1e-12
    # the score is real; a surviving imaginary part would mean the conjugate fold went wrong
    assert max(np.abs(np.asarray(engine.decrypt(o), dtype=complex).imag).max() for o in out) == 0.0


def test_attention_score_costs_four_levels(masks, score_masks):
    """transpose, the group rotation, make_copies and the product's own rescales."""
    rng = np.random.default_rng(22)
    q = rng.normal(size=(G.dim, G.features)) * 0.1
    k = rng.normal(size=(G.dim, G.features)) * 0.1
    engine, stages = build_score(masks, score_masks, depth=30)
    out = stages.stage_06_attention_score(encode_qkv_output(engine, q), encode_qkv_output(engine, k))
    assert [engine.level(o) for o in out] == [30 - 4] * 8


def test_he_py_accumulator_table_is_wrong(masks, score_masks):
    """Pin the divergence from he.py, so a well-meaning "fix" back to the literal table fails here.

    he.py routes the `in_index % pack == 0` contribution by `out_index == 0`; the rule that makes the
    result equal Q K^T is the one its own other branch uses, `offset < 0`. They agree for
    `in_index // pack == 1` and differ for 2 and 3 - in_index 32 and 48 of the 64 inner-product terms.
    """
    from thorfhe.attention import AttentionScore

    class HePyRouting(AttentionScore):
        def accumulator_column(self, out_index, offset, j):
            if j == 0:
                return 2 if out_index == 0 else 0
            return super().accumulator_column(out_index, offset, j)

    rng = np.random.default_rng(23)
    q = rng.normal(size=(G.dim, G.features)) * 0.1
    k = rng.normal(size=(G.dim, G.features)) * 0.1

    (low, high), transpose, copies, attention = masks
    engine = ClearEngine(G, depth=30)
    stages = HePyRouting(engine, G, masks=low, complement_masks=high, transpose=transpose,
                         copies=copies, attention=attention, ccmm=score_masks)
    out = stages.stage_06_attention_score(encode_qkv_output(engine, q), encode_qkv_output(engine, k))
    got = np.stack([np.asarray(engine.decrypt(o), dtype=complex).real for o in out])

    expect = expected_scores(engine, q, k)
    error = np.abs(got - expect).max()
    assert error > 0.1 * np.abs(expect).max(), "he.py's routing should be visibly wrong, not marginal"

    # and it is wrong only where the two rules disagree: the middle two output ciphertexts of each half
    per_ciphertext = [np.abs(got[i] - expect[i]).max() for i in range(8)]
    assert [i for i, e in enumerate(per_ciphertext) if e > 1e-12] == [1, 2, 5, 6]
