"""Stage 12: the feed-forward expansion.

The intermediate weight is (4 * features, features) - four times too tall for one block diagonal - so
THOR splits it into four pieces and stacks them two at a time. Each of the two `rep`s then carries
twelve block rows in a token's sixteen slots, laid out as two windows of six, and the partial products
are recombined with a window of six on a stride of eight rather than twelve on sixteen.

That window is what `block_diag_2` is, and it is a property of the product rather than of the packing -
which is why `pcmm` takes it as an argument. The test pins the resulting layout down exactly:

    out[rep][ct][group, t, s] = z[t, 128 * block + (ct*pack + group + t) % 128]
    block = 12 * rep + FF_SLOT_INDICES.index(s)

with `z` the pre-activation divided by 64, ready for GELU.
"""
import numpy as np
import pytest

from thorfhe import (FEEDFORWARD_STRIDE, FEEDFORWARD_WINDOW, THOR_FEEDFORWARD, ClearEngine,
                     block_diagonal_masks, encode_weight_ff)
from thorfhe.encoding import FF_SLOT_INDICES
from thorfhe.feedforward import FeedForwardStages, feedforward_masks, input_mask
from thorfhe.numeric import GELU_SCALE

F = THOR_FEEDFORWARD
DEPTH = 50
INTERMEDIATE = 4 * F.features


def feature_of(ct, group, token, block):
    return F.n_out * block + (ct * F.pack + group + token) % F.n_out


def encode_input(engine, x):
    """The 6-block-of-128 layout LayerNorm leaves behind."""
    out = []
    for ct in range(F.n_output_ciphertexts):
        msg = np.zeros(F.slot_count)
        for group in range(F.pack):
            for token in range(F.dim):
                for block in range(FEEDFORWARD_WINDOW):
                    msg[F.slot(group, token, block)] = x[token, feature_of(ct, group, token, block)]
        out.append(engine.encrypt(msg))
    return np.array(out, dtype=object)


def zero_bias():
    bias = np.empty((2, F.n_output_ciphertexts), dtype=object)
    for rep in range(2):
        for column in range(F.n_output_ciphertexts):
            bias[rep, column] = np.zeros(F.slot_count)
    return bias


@pytest.fixture(scope="module")
def encoded():
    """Encoding the expansion is the slow part - 6144 slot vectors - so do it once."""
    rng = np.random.default_rng(111)
    x = rng.normal(size=(F.dim, F.features)) * 0.1
    w = rng.normal(size=(INTERMEDIATE, F.features)) * 0.02
    weight = encode_weight_ff(w, dim=F.dim, pack=F.pack, n_slot=F.n_slot, group_size=F.group_size,
                              slot_count=F.slot_count, n_in=F.n_in, n_out=F.n_out, split=4,
                              scale=1.0 / GELU_SCALE)
    return x, w, weight


def test_encoded_shape_matches_thor(encoded):
    """he.py declares `weight = np.full((2, 8, 6, 64))`; the geometry has to reproduce that."""
    _, _, weight = encoded
    assert weight.shape == (2, F.n_output_ciphertexts, F.diag_count, F.n_in_complex)
    assert weight.shape == (2, 8, 6, 64)


def test_feedforward_window_is_six_on_eight():
    """`block_diag_2`: six blocks wrapping in half a token, not twelve in a whole one."""
    low, high = feedforward_masks(F)
    within = np.arange(F.slot_count) % FEEDFORWARD_STRIDE
    assert np.array_equal(low[3], ((within < 3) & (within < FEEDFORWARD_WINDOW)).astype(float))
    # every used slot is in exactly one of the two halves
    assert np.array_equal(low[3] + high[3], (within < FEEDFORWARD_WINDOW).astype(float))
    assert (input_mask(F) > 0).sum() == F.slot_count / F.n_slot * FEEDFORWARD_WINDOW


def test_intermediate_dense_computes_the_expansion(encoded):
    """`x @ W.T / 64` in the layout the module docstring states - exactly."""
    x, w, weight = encoded
    engine = ClearEngine(F, depth=DEPTH)
    low, high = block_diagonal_masks(F)
    stages = FeedForwardStages(engine, F, masks=low, complement_masks=high)

    out = stages.stage_12_intermediate_dense(encode_input(engine, x), weight, zero_bias())
    assert out.shape == (2, 8)

    want = x @ w.T / GELU_SCALE
    error = 0.0
    for rep in range(2):
        for ct in range(F.n_output_ciphertexts):
            slots = np.asarray(engine.decrypt(out[rep, ct]), dtype=complex)
            assert np.abs(slots.imag).max() == 0.0
            for group in range(F.pack):
                diagonal = ct * F.pack + group
                for token in (0, 7, 50, 127):
                    for position, slot in enumerate(FF_SLOT_INDICES):
                        block = FEEDFORWARD_WINDOW * 2 * rep + position
                        error = max(error, abs(slots.real[F.slot(group, token, slot)]
                                               - want[token, F.n_out * block
                                                      + (diagonal + token) % F.n_out]))
    assert error < 1e-12


def test_every_intermediate_feature_is_covered(encoded):
    """Two reps of twelve blocks of 128 is exactly the 3072 the expansion produces."""
    assert 2 * 2 * FEEDFORWARD_WINDOW * F.n_out == INTERMEDIATE
    assert len(FF_SLOT_INDICES) == 2 * FEEDFORWARD_WINDOW
