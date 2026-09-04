"""Stages 17 and 18: the pooler and the classifier - the last two stages of a THOR forward pass.

Both act on the CLS token alone, which is why they look unlike everything before them. The pooler
masks token 0 out of the final LayerNorm output and broadcasts it back over all 128 tokens, so every
group already holds the same token and there is no rotated-copy dimension: ``encode_w_pooler`` folds
the pack offset into the input index instead, giving a (6, 4) weight where an ordinary dense layer
needs (8, 6, 64). The classifier is then one inner product per class, done entirely with rotations.

The layout the pooler produces is checked exactly, and it is exactly the one ``encode_w_cls`` expects:

    slot n_slot * t + block  ==  (W @ cls)[dim * block + t]
"""
import numpy as np
import pytest

from thorfhe import (THOR_FEEDFORWARD, ClearEngine, block_diagonal_masks, encode_bias_classifier,
                     encode_bias_pooler, encode_weight_classifier, encode_weight_pooler, pooler_mask)
from thorfhe.numeric import POOLER_INNER, POOLER_OUTER
from thorfhe.pooler import PoolerStages

F = THOR_FEEDFORWARD
DEPTH, BOOTSTRAP_LEVEL = 60, 40
#: The pooler's own scaling: the tanh composite is fitted in ``pre_activation / 40``.
POOLER_SCALE = 40


def feature_of(ct, group, token, block):
    return F.n_out * block + (ct * F.pack + group + token) % F.n_out


def evaluate(coefficients, x):
    return np.polyval(coefficients[::-1], x)


@pytest.fixture(scope="module")
def head():
    """One pooler + classifier run: the LayerNorm output in, two logits out."""
    rng = np.random.default_rng(21)
    y = rng.normal(size=(F.dim, F.features)) * 0.1
    w = rng.normal(size=(F.features, F.features)) * 0.02
    b = rng.normal(size=(F.features,)) * 0.05
    w_cls = rng.normal(size=(2, F.features)) * 0.03
    b_cls = rng.normal(size=(2,)) * 0.1

    engine = ClearEngine(F, depth=DEPTH, bootstrap_level=BOOTSTRAP_LEVEL)
    low, high = block_diagonal_masks(F)
    stages = PoolerStages(engine, F, masks=low, complement_masks=high)

    x = []
    for ct in range(2 * F.n_output_ciphertexts // 2):
        msg = np.zeros(F.slot_count)
        for group in range(F.pack):
            for token in range(F.dim):
                for block in range(F.out_blocks):
                    msg[F.slot(group, token, block)] = y[token, feature_of(ct, group, token, block)]
        x.append(engine.encrypt(msg))

    merged = [stages.add(x[i], stages.multiply_1j(x[i + 4])) for i in range(4)]
    dense = stages.pooler_dense(merged, encode_weight_pooler(F, w), np.zeros(F.slot_count))
    pooled = stages.stage_17_pooler(x, encode_weight_pooler(F, w), encode_bias_pooler(F, b))
    logits = stages.stage_18_classifier(pooled, encode_weight_classifier(F, w_cls),
                                        encode_bias_classifier(F, b_cls))
    return dict(engine=engine, y=y, w=w, b=b, w_cls=w_cls, b_cls=b_cls,
                dense=dense, pooled=pooled, logits=logits)


def test_encoded_pooler_weight_matches_thor():
    """he.py declares `weight = np.full((6, 4))` - no pack dimension, because the CLS token is one row."""
    rng = np.random.default_rng(0)
    weight = encode_weight_pooler(F, rng.normal(size=(F.features, F.features)))
    assert weight.shape == (6, 4)
    assert encode_weight_classifier(F, rng.normal(size=(2, F.features))).shape == (2,)


def test_pooler_mask_is_the_cls_token_of_every_group():
    mask = pooler_mask(F)
    assert (mask > 0).sum() == F.pack * F.out_blocks
    assert np.array_equal(np.flatnonzero(mask),
                          np.concatenate([F.group_size * g + np.arange(F.out_blocks)
                                          for g in range(F.pack)]))


def test_pooler_dense_is_the_cls_inner_product_exactly(head):
    """`(W @ cls)[dim * block + t]` at slot `n_slot * t + block` - no tolerance, it is linear algebra."""
    engine, want = head["engine"], head["w"] @ head["y"][0]
    slots = np.asarray(engine.decrypt(head["dense"][0]), dtype=complex)
    assert np.abs(slots.imag).max() < 1e-12
    error = 0.0
    for block in range(F.out_blocks):
        for token in range(F.dim):
            error = max(error, abs(slots.real[F.n_slot * token + block] - want[F.dim * block + token]))
    assert error < 1e-12
    # the padding slots stay clean, which is what lets the classifier fold six and not eight
    within = np.arange(F.slot_count) % F.n_slot
    assert np.abs(slots.real[within >= F.out_blocks]).max() < 1e-12


def test_pooler_bias_is_halved_unlike_the_qkv_bias(head):
    """`encode_b_pooler` divides by two, so the closing fold gives `+b` and not `+2b`."""
    engine = head["engine"]
    encoded = encode_bias_pooler(F, head["b"])
    assert encoded[F.n_slot * 3 + 2] == pytest.approx(head["b"][F.dim * 2 + 3] / 2)
    slots = np.asarray(engine.decrypt(head["pooled"][0]), dtype=complex).real
    want = np.tanh(head["w"] @ head["y"][0] + head["b"])
    assert abs(slots[F.n_slot * 3 + 2] - want[F.dim * 2 + 3]) < 5e-3


def test_stage_17_is_tanh_of_the_pooler_dense(head):
    engine = head["engine"]
    want = np.tanh(head["w"] @ head["y"][0] + head["b"])
    slots = np.asarray(engine.decrypt(head["pooled"][0]), dtype=complex)
    error = 0.0
    for block in range(F.out_blocks):
        for token in range(F.dim):
            error = max(error, abs(slots.real[F.n_slot * token + block] - want[F.dim * block + token]))
    assert error < 5e-3                       # the degree-15 composite's own accuracy
    assert head["pooled"][0].level == BOOTSTRAP_LEVEL   # it closes with a bootstrap


def test_stage_18_logits(head):
    engine = head["engine"]
    pooled = np.tanh(head["w"] @ head["y"][0] + head["b"])
    want = head["w_cls"] @ pooled + head["b_cls"]
    for class_index in range(2):
        got = np.asarray(engine.decrypt(head["logits"][class_index]), dtype=complex)
        assert abs(got.real[0] - want[class_index]) < 5e-3


def test_the_classifier_folds_six_slots_not_eight():
    """`1, 2, 4` off the *pair*, not a doubling chain - eight would pull in the padding slots."""
    engine = ClearEngine(F, depth=DEPTH)
    low, high = block_diagonal_masks(F)
    stages = PoolerStages(engine, F, masks=low, complement_masks=high)

    marker = np.zeros(F.slot_count)
    marker[:F.n_slot] = 1.0                            # one whole token, padding included
    weight = np.zeros((1,), dtype=object)
    weight[0] = np.zeros(F.slot_count)
    weight[0][:F.n_slot] = [1.0] * F.n_slot
    bias = np.zeros((1,), dtype=object)
    bias[0] = np.zeros(F.slot_count)

    out = stages.stage_18_classifier([engine.encrypt(marker)], weight, bias)
    assert np.asarray(engine.decrypt(out[0]), dtype=complex).real[0] == pytest.approx(F.out_blocks)


def test_the_pooler_tanh_is_valid_to_about_ten_and_then_plateaus():
    """The `/40` exists to keep the argument inside the fit; past it the error saturates, not diverges."""
    def composite(pre_activation):
        return evaluate(POOLER_OUTER, evaluate(POOLER_INNER, pre_activation / POOLER_SCALE))

    for bound in (4, 8, 10):
        z = np.linspace(-bound, bound, 2001)
        assert np.abs(composite(z) - np.tanh(z)).max() < 1.3e-2
    far = np.linspace(-30, 30, 2001)
    error = np.abs(composite(far) - np.tanh(far)).max()
    assert 0.1 < error < 0.25                  # degraded, but still bounded - unlike the GELU inner fit
    assert np.abs(composite(far)).max() < 1.01
