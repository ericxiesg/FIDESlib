"""Stages 13, 14 and 15: GELU, the feed-forward contraction, and the residual bootstrap.

Stage 13 folds the two ``rep`` ciphertexts into one complex one before bootstrapping, so the GELU
costs eight bootstraps rather than sixteen. The halving in front of the bootstrap and the doubling in
``temp + conj(temp)`` cancel exactly, which is what lets the same trick appear in stage 15 *without* a
halving and deliberately return twice the residual.

Stage 14 contracts 3072 back to 768. The weight is the same ``encode_w_ff`` as stage 12 but split
horizontally, and each ``rep`` contracts two of the four pieces - one per window - which
``rotate(temp, -8)`` then sums. That rotation pulls the next token's low window into slots 8..13, so
only slots 0..5 are meaningful; LayerNorm's ``value_mask`` zeroes the rest before reading anything.
"""
import numpy as np
import pytest

from thorfhe import (FEEDFORWARD_STRIDE, FEEDFORWARD_WINDOW, THOR_FEEDFORWARD, ClearEngine,
                     block_diagonal_masks, encode_weight_ff)
from thorfhe.encoding import FF_SLOT_INDICES
from thorfhe.feedforward import FeedForwardStages
from thorfhe.layernorm import LayerNormStages, value_mask
from thorfhe.numeric import GELU_SCALE

F = THOR_FEEDFORWARD
INTERMEDIATE = 4 * F.features
#: High enough that the schedule, not the level budget, is what the chain is tested against.
DEPTH, BOOTSTRAP_LEVEL = 80, 60


def numpy_gelu(v):
    return 0.5 * v * (1 + np.tanh(np.sqrt(2 / np.pi) * (v + 0.044715 * v ** 3)))


def feature_of(ct, group, token, block):
    return F.n_out * block + (ct * F.pack + group + token) % F.n_out


def encode_six_blocks(engine, x):
    """The 6-block-of-128 layout LayerNorm produces and stage 12 consumes."""
    out = []
    for ct in range(F.n_output_ciphertexts):
        msg = np.zeros(F.slot_count)
        for group in range(F.pack):
            for token in range(F.dim):
                for block in range(FEEDFORWARD_WINDOW):
                    msg[F.slot(group, token, block)] = x[token, feature_of(ct, group, token, block)]
        out.append(engine.encrypt(msg))
    return np.array(out, dtype=object)


def encode_twelve_blocks(engine, h):
    """The two-window layout stages 12 and 13 produce: ``block = 12 * rep + index_of(slot)``."""
    out = np.empty((2, F.n_output_ciphertexts), dtype=object)
    for rep in range(2):
        for ct in range(F.n_output_ciphertexts):
            msg = np.zeros(F.slot_count)
            for group in range(F.pack):
                for token in range(F.dim):
                    for position, slot in enumerate(FF_SLOT_INDICES):
                        block = 2 * FEEDFORWARD_WINDOW * rep + position
                        msg[F.slot(group, token, slot)] = h[token, feature_of(ct, group, token, block)]
            out[rep, ct] = engine.encrypt(msg)
    return out


def zero_plaintexts(*shape):
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        out[index] = np.zeros(F.slot_count)
    return out


def stages(engine):
    low, high = block_diagonal_masks(F)
    return FeedForwardStages(engine, F, masks=low, complement_masks=high)


def encode_contraction(w):
    return encode_weight_ff(w, dim=F.dim, pack=F.pack, n_slot=F.n_slot, group_size=F.group_size,
                            slot_count=F.slot_count, n_in=F.n_in, n_out=F.n_out, split=4, axis=1)


@pytest.fixture(scope="module")
def model():
    """Encoding two 768x3072 weights costs about a minute each, so do it once for the module."""
    rng = np.random.default_rng(3)
    x = rng.normal(size=(F.dim, F.features)) * 0.1
    w1 = rng.normal(size=(INTERMEDIATE, F.features)) * 0.02
    w2 = rng.normal(size=(F.features, INTERMEDIATE)) * 0.02
    expansion = encode_weight_ff(w1, dim=F.dim, pack=F.pack, n_slot=F.n_slot,
                                 group_size=F.group_size, slot_count=F.slot_count, n_in=F.n_in,
                                 n_out=F.n_out, split=4, axis=0, scale=1.0 / GELU_SCALE)
    return dict(x=x, w1=w1, w2=w2, e1=expansion, e2=encode_contraction(w2))


@pytest.fixture(scope="module")
def chain(model):
    """12 -> 13 -> 14 once; the individual tests then read whichever intermediate they need."""
    engine = ClearEngine(F, depth=DEPTH, bootstrap_level=BOOTSTRAP_LEVEL)
    st = stages(engine)
    s12 = st.stage_12_intermediate_dense(encode_six_blocks(engine, model["x"]), model["e1"],
                                         zero_plaintexts(2, F.n_output_ciphertexts))
    s13 = st.stage_13_gelu(s12)
    s14 = st.stage_14_output_dense(s13, model["e2"], zero_plaintexts(F.n_output_ciphertexts))
    return engine, s12, s13, s14


def test_encoded_contraction_matches_thor(model):
    """he.py declares `weight = np.full((2, 8, 6, 64))` for stage 14 as well as stage 12."""
    assert model["e2"].shape == (2, 8, 6, 64)


def test_gelu_costs_thirteen_levels_and_leaves_the_shape_alone(chain):
    engine, s12, s13, _ = chain
    assert s13.shape == s12.shape == (2, F.n_output_ciphertexts)
    assert s13[0, 0].level == BOOTSTRAP_LEVEL - 13
    assert s13[0, 0].scale_exp == 1


def test_stage_13_is_gelu_in_place(chain, model):
    """Elementwise, in the layout stage 12 left behind - so the fold has to have cancelled exactly."""
    engine, _, s13, _ = chain
    want = numpy_gelu(model["x"] @ model["w1"].T)
    error = 0.0
    for rep in range(2):
        for ct in range(F.n_output_ciphertexts):
            slots = np.asarray(engine.decrypt(s13[rep, ct]), dtype=complex)
            assert np.abs(slots.imag).max() < 1e-12
            for group in range(F.pack):
                for token in (0, 40, 127):
                    for position, slot in enumerate(FF_SLOT_INDICES):
                        block = 2 * FEEDFORWARD_WINDOW * rep + position
                        error = max(error, abs(slots.real[F.slot(group, token, slot)]
                                               - want[token, feature_of(ct, group, token, block)]))
    # the polynomial composite's own accuracy, not the packing's
    assert error < 1e-4
    assert error / np.abs(want).max() < 1e-3


def test_stage_14_contracts_to_six_blocks(chain, model):
    """`gelu(x @ W1.T) @ W2.T`, in the 6-block layout LayerNorm reads."""
    engine, _, _, s14 = chain
    assert s14.shape == (F.n_output_ciphertexts,)
    want = numpy_gelu(model["x"] @ model["w1"].T) @ model["w2"].T
    error = 0.0
    for ct in range(F.n_output_ciphertexts):
        slots = np.asarray(engine.decrypt(s14[ct]), dtype=complex)
        for group in range(F.pack):
            for token in (0, 40, 127):
                for block in range(FEEDFORWARD_WINDOW):
                    error = max(error, abs(slots.real[F.slot(group, token, block)]
                                           - want[token, feature_of(ct, group, token, block)]))
    assert error / np.abs(want).max() < 1e-3


@pytest.fixture(scope="module")
def contraction_only():
    """Isolated from GELU the product is linear algebra, so it can be checked exactly."""
    rng = np.random.default_rng(7)
    h = rng.normal(size=(F.dim, INTERMEDIATE)) * 0.05
    w = rng.normal(size=(F.features, INTERMEDIATE)) * 0.02
    engine = ClearEngine(F, depth=DEPTH)
    out = stages(engine).stage_14_output_dense(encode_twelve_blocks(engine, h), encode_contraction(w),
                                               zero_plaintexts(F.n_output_ciphertexts))
    return engine, h, w, out


def test_contraction_alone_is_exact(contraction_only):
    engine, h, w, out = contraction_only
    want = h @ w.T
    error = 0.0
    for ct in range(F.n_output_ciphertexts):
        slots = np.asarray(engine.decrypt(out[ct]), dtype=complex)
        for group in range(F.pack):
            for token in (0, 3, 61, 127):
                for block in range(FEEDFORWARD_WINDOW):
                    error = max(error, abs(slots.real[F.slot(group, token, block)]
                                           - want[token, feature_of(ct, group, token, block)]))
    assert error < 1e-12


def test_stage_14_pollutes_only_slots_layernorm_masks_off(contraction_only):
    """The `rotate(temp, -8)` fold leaves the next token's window at 8..13. Nothing else may move."""
    engine, _, _, out = contraction_only
    within = np.arange(F.slot_count) % F.n_slot
    slots = np.asarray(engine.decrypt(out[0]), dtype=complex).real

    clean = ((within >= FEEDFORWARD_WINDOW) & (within < FEEDFORWARD_STRIDE)) | (within >= 14)
    assert np.abs(slots[clean]).max() < 1e-12
    polluted = (within >= FEEDFORWARD_STRIDE) & (within < FEEDFORWARD_STRIDE + FEEDFORWARD_WINDOW)
    assert np.abs(slots[polluted]).max() > 1e-6
    # ... and that is exactly what LayerNorm drops before it reads anything
    assert np.array_equal(value_mask(F, 1.0) > 0, within < FEEDFORWARD_WINDOW)


def test_stage_15_returns_twice_the_residual():
    """No halving in front of the bootstrap, unlike stage 13 - the doubling is the point."""
    engine = ClearEngine(F, depth=DEPTH, bootstrap_level=BOOTSTRAP_LEVEL)
    st = LayerNormStages(engine, F, masks={}, complement_masks={})
    rng = np.random.default_rng(5)
    a = [rng.normal(size=F.slot_count) for _ in range(8)]
    b = [rng.normal(size=F.slot_count) for _ in range(8)]
    out = st.stage_15_prepare_layernorm([engine.encrypt(v) for v in a],
                                        [engine.encrypt(v) for v in b])
    for index in range(8):
        got = np.asarray(engine.decrypt(out[index]), dtype=complex)
        assert np.abs(got.imag).max() < 1e-12
        assert np.abs(got.real - 2 * (a[index] + b[index])).max() < 1e-12
    assert out[0].level == BOOTSTRAP_LEVEL - 3


def test_the_doubling_is_what_the_wide_variance_window_is_for():
    """`variance_window` accepts 4x for the variants stage 16 uses - i.e. exactly a doubled input."""
    narrow = LayerNormStages.variance_window(0.15, 10.0)      # variant 1, after stage 11: no doubling
    wide = LayerNormStages.variance_window(0.2, 150.0)        # variant 2, after stage 15: doubled
    assert narrow[0] / 0.15 == pytest.approx(1.05)
    assert wide[0] / 0.2 == pytest.approx(4 * 1.05)
