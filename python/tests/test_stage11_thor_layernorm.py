"""Stage 11/16: LayerNorm.

The mean and variance of a token's 768 features have to come out of a representation that spreads them
over 8 ciphertexts, 16 groups and 6 slots. THOR reduces them into slot 0, inverts the square root
there, and broadcasts back - so the expensive part runs on one slot in sixteen.

Two things the tests pin down beyond the arithmetic: every variant returns **twice** the LayerNorm (the
final doubling is never cancelled), and the token variance has to sit inside the variant's window or
the inverse square root has nothing to converge to. The `/2` in variants 2 and 3 moves that window up
by four rather than cancelling the doubling, which is the easy thing to get backwards.
"""
import os
import traceback

import numpy as np
import pytest

from thorfhe import THOR_ATTENTION_DENSE, ClearEngine, block_diagonal_masks
from thorfhe.encoding import encode_bias
from thorfhe.layernorm import LayerNormStages, statistic_mask, value_mask

D = THOR_ATTENTION_DENSE
DEPTH = 60
VAR_E = 1e-5


@pytest.fixture(scope="module")
def masks():
    return block_diagonal_masks(D)


def build(masks, depth=DEPTH):
    engine = ClearEngine(D, depth=depth, bootstrap_level=depth)
    low, high = masks
    return engine, LayerNormStages(engine, D, masks=low, complement_masks=high)


def feature_of(ct, group, token, block):
    """Which feature a slot carries in the post-stage-10 layout."""
    return D.n_out * block + (ct * D.pack + group + token) % D.n_out


def encode(values):
    """(dim, features) -> the eight slot vectors stage 10 leaves behind."""
    out = []
    for ct in range(D.n_output_ciphertexts):
        msg = np.zeros(D.slot_count)
        for group in range(D.pack):
            for token in range(D.dim):
                for block in range(D.out_blocks):
                    msg[D.slot(group, token, block)] = values[token, feature_of(ct, group, token, block)]
        out.append(msg)
    return out


def decode(engine, cts):
    out = np.zeros((D.dim, D.features))
    for ct in range(D.n_output_ciphertexts):
        slots = np.real(engine.decrypt(cts[ct]))
        for group in range(D.pack):
            for token in range(D.dim):
                for block in range(D.out_blocks):
                    out[token, feature_of(ct, group, token, block)] = slots[D.slot(group, token, block)]
    return out


def reference(x, gamma, beta):
    mean = x.mean(axis=1, keepdims=True)
    variance = x.var(axis=1, keepdims=True)
    return gamma * (x - mean) / np.sqrt(variance + VAR_E / D.features ** 2) + beta


def sample(seed, spread=1.0):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(D.dim, D.features)) * spread,
            rng.normal(size=(D.features,)) * 0.3 + 1.0,
            rng.normal(size=(D.features,)) * 0.1)


def run(masks, which, x, gamma, beta):
    engine, stages = build(masks)
    cts = [engine.encrypt(m) for m in encode(x)]
    gammas = encode(np.tile(gamma, (D.dim, 1)))
    betas = encode(np.tile(beta, (D.dim, 1)))
    ones = engine.encrypt(statistic_mask(D))
    out = getattr(stages, which)(cts, gammas, betas, ones)
    return engine, out


@pytest.mark.parametrize("which,bounds,spread", [
    ("he_layernorm1", (0.15, 10.0), 1.0),
    ("he_layernorm2", (0.2, 150.0), 1.0),
    ("he_layernorm3", (0.75, 2500.0), 2.0),
])
def test_layernorm_returns_twice_the_layernorm(masks, which, bounds, spread):
    """Every variant doubles; only the accepted variance window differs."""
    x, gamma, beta = sample(91, spread)
    low, high = LayerNormStages.variance_window(*bounds)
    variances = x.var(axis=1)
    assert low < variances.min() and variances.max() < high, (
        f"{which} accepts variances in [{low:.3g}, {high:.3g}], got "
        f"[{variances.min():.3g}, {variances.max():.3g}]")

    engine, out = run(masks, which, x, gamma, beta)
    got = decode(engine, out)
    want = 2 * reference(x, gamma, beta)
    assert np.abs(got - want).max() / np.abs(want).max() < 1e-3


def test_statistic_lives_in_one_slot(masks):
    """The mask the inverse square root runs on is slot 0 of each token - one slot in sixteen."""
    statistic = statistic_mask(D)
    assert statistic.sum() == D.slot_count / D.n_slot
    assert statistic[D.slot(0, 0, 0)] == 1.0
    assert statistic[D.slot(0, 0, 1)] == 0.0
    # the values occupy the six blocks stage 10 folded them into
    assert (value_mask(D, 1.0) > 0).sum() == D.slot_count / D.n_slot * D.out_blocks


def test_level_cost_is_fixed(masks):
    """Data-independent, so the layer's depth budget is known before it runs."""
    x, gamma, beta = sample(93)
    engine, out = run(masks, "he_layernorm1", x, gamma, beta)
    levels = {engine.level(o) for o in out}
    assert len(levels) == 1
    assert levels.pop() == DEPTH - 14


def test_which_variant_halves_survives_scaling_its_bounds():
    """Variant identity must not be inferred from how large the bounds happen to be.

    `variance_window` and `he_layernorm` both used to decide whether a variant halves its input by
    testing `min_var > 0.16`, a threshold that sits between variant 1's 0.15 and variant 2's 0.2. It
    gives the right answer for the three variants as written and the wrong one the moment the bounds
    are scaled - which is exactly what compensating for a scaled stage-15 residual requires, since
    dividing the input by `s` divides the variance by `s^2`.
    """
    stages = LayerNormStages
    # variant 2's bounds, scaled down the way a 1/128 input scaling would scale them
    scaled_min, scaled_max = 0.2 / 128 ** 2, 150.0 / 128 ** 2

    inferred = stages.variance_window(scaled_min, scaled_max)
    stated = stages.variance_window(scaled_min, scaled_max, halves=stages.HALVES[2])

    assert inferred[1] / inferred[0] == pytest.approx(stated[1] / stated[0]), "the ratio is a variant property"
    assert stated[0] == pytest.approx(4 * 1.05 * scaled_min), "variant 2 halves, so it accepts 4x"
    assert inferred[0] == pytest.approx(1.05 * scaled_min), "and the threshold would have said 1x"
    assert stated[0] != inferred[0], "which is the silent flip this test exists to catch"


def test_the_stated_and_inferred_halving_agree_on_the_unscaled_variants():
    """The inference is right for the bounds as written - that is why it survived this long."""
    stages = LayerNormStages
    for variant, (min_var, max_var) in ((1, (0.15, 10.0)), (2, (0.2, 150.0)), (3, (0.75, 2500.0))):
        assert (stages.variance_window(min_var, max_var)
                == stages.variance_window(min_var, max_var, halves=stages.HALVES[variant]))


@pytest.mark.parametrize("scale", [2.0, 128.0])
def test_stage_16_is_invariant_to_a_scaled_residual(masks, scale):
    """Scaling stage 15's output and the bounds together must leave stage 16's answer alone.

    This is the whole basis of the magnitude fix: stage 15 hands its bootstrap a residual reaching
    132 on the real checkpoint, against a usable window of about 10, so the residual is divided by
    `s` in the plaintexts upstream. What makes that free is that stage 16 normalises - dividing its
    input by `s` divides the variance by `s^2`, and once the bounds follow, the output is the same
    number. Exact arithmetic here, so `the same` is to floating point.

    The level cost is unchanged for a reason visible in the code rather than measured: `he_invsqrt`
    takes `epsilon = min_var / max_var` (layernorm.py:133), a ratio, and `var_e / max_denominator`
    is a ratio too - so dividing both bounds by `s^2` changes neither.
    """
    rng = np.random.default_rng(29)
    values = rng.normal(0, 1.2, (D.dim, D.features))
    gamma = rng.normal(1.0, 0.1, D.features)
    beta = rng.normal(0.0, 0.1, D.features)

    got = {}
    levels = {}
    for factor in (1.0, scale):
        engine, stages = build(masks)
        stages.residual_scale = factor
        x = [engine.encrypt(m) for m in encode(values / factor)]
        out = stages.stage_16_output_layernorm(
            x, encode_bias(D, gamma), encode_bias(D, beta),
            engine.encrypt(np.ones(D.slot_count)), layer_index=0)
        got[factor] = [np.real(np.asarray(engine.decrypt(ct))) for ct in out]
        levels[factor] = engine.level(out[0])

    difference = max(float(np.max(np.abs(a - b))) for a, b in zip(got[1.0], got[scale]))
    magnitude = max(float(np.max(np.abs(a))) for a in got[1.0])
    assert difference / magnitude < 1e-9, (
        f"scaling the residual by {scale:g} moved stage 16's output by {difference:.3g} "
        f"against a magnitude of {magnitude:.3g}")
    assert levels[1.0] == levels[scale], "and it must not change what the stage costs"


def test_residual_scale_does_not_change_what_a_layer_computes():
    """The same check as the stage-16 one above, end to end.

    The algebra is exact and the stage-16 test covers the only step where it is not obvious - stages
    11, 12 and 14 are linear, and gamma * (1/s), W * s and W2 * (1/s) cancel by construction. What
    this adds is coverage of the wiring: that no stage reads `norm_1` besides 12 and 15, and that
    `encode_layer` and `EncoderLayer` were given the same scale. Off by default because two layer
    forwards on the clear engine do not fit the machine this was written on.
    """
    from thorfhe.encoding import encode_activations
    from thorfhe.geometry import THOR_BERT
    from thorfhe.layer import EncoderLayer, encode_layer

    g = THOR_BERT
    rng = np.random.default_rng(17)
    f = g.features
    weights = {
        "query.weight": rng.normal(0, .04, (f, f)), "query.bias": rng.normal(0, .01, f),
        "key.weight": rng.normal(0, .04, (f, f)), "key.bias": rng.normal(0, .01, f),
        "value.weight": rng.normal(0, .04, (f, f)), "value.bias": rng.normal(0, .01, f),
        "attention.output.dense.weight": rng.normal(0, .04, (f, f)),
        "attention.output.dense.bias": rng.normal(0, .01, f),
        "attention.output.LayerNorm.weight": rng.normal(1, .1, f),
        "attention.output.LayerNorm.bias": rng.normal(0, .1, f),
        "intermediate.dense.weight": rng.normal(0, .04, (4 * f, f)),
        "intermediate.dense.bias": rng.normal(0, .01, 4 * f),
        "output.dense.weight": rng.normal(0, .04, (f, 4 * f)),
        "output.dense.bias": rng.normal(0, .01, f),
        "output.LayerNorm.weight": rng.normal(1, .1, f),
        "output.LayerNorm.bias": rng.normal(0, .1, f)}
    activations = rng.normal(0, 1.0, (g.dim, g.features))

    outputs, peaks = {}, {}
    for scale in (1.0, 128.0):
        seen = []

        class Probe(ClearEngine):
            def bootstrap(self, ct, keep_levels=None):
                # by site, not just the largest: `residual_scale` only touches stage 15, and once it
                # has been divided by 128 that site is no longer the largest in the layer - so a
                # check on the global peak stops measuring the thing it was written for
                stage = next((f.name for f in traceback.extract_stack() if f.name.startswith("stage_")),
                             "?")
                seen.append((stage, float(np.max(np.abs(ct.slots)))))
                return super().bootstrap(ct, keep_levels)

        engine = Probe(g, depth=37, bootstrap_level=20)
        engine.bootstrap_message_margin = 1e9     # measure the magnitude rather than refuse it
        layer = EncoderLayer(engine, residual_scale=scale, binary_rotations=True,
                             refresh_after_dense=True)
        for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
            owner.check_ranges = False
        packed = np.array([engine.encrypt(m) for m in encode_activations(g, activations)],
                          dtype=object)
        # `lazy=True` holds one weight field at a time instead of the whole encoded layer, which is
        # 9.7 GB at this geometry - the reason this test was gated off in the first place
        result = layer.forward(packed, encode_layer(weights, 0, residual_scale=scale, lazy=True),
                               layer.padding_mask(g.dim), 0)
        outputs[scale] = np.stack([np.asarray(engine.decrypt(ct)) for ct in result])
        peaks[scale] = max(magnitude for stage, magnitude in seen
                           if stage == "stage_15_prepare_layernorm")
        del layer, engine, packed, result

    difference = float(np.max(np.abs(outputs[1.0] - outputs[128.0])))
    magnitude = float(np.max(np.abs(outputs[1.0])))
    assert difference / magnitude < 1e-9, (
        f"scaling the residual moved the layer's output by {difference:.3g} against {magnitude:.3g}")
    assert peaks[1.0] / peaks[128.0] == pytest.approx(128.0, rel=0.05), (
        f"and it has to actually scale what stage 15 bootstraps: {peaks[1.0]:.4g} -> "
        f"{peaks[128.0]:.4g}. Scaling the residual is only worth doing if it moves that site, and "
        f"only stage 15's own input should move - the rest of the layer's bootstraps are untouched")


@pytest.mark.parametrize("scale", [2.0, 4.0])
def test_refresh_stays_the_identity_under_a_scaled_bootstrap(masks, scale):
    """`refresh` is documented as semantically the identity; scaling its bootstrap must keep it so.

    The halving in front of the bootstrap is not enough on the real checkpoint - the attention dense
    reaches 3.8 to 5.2, so what is bootstrapped is 1.9 to 2.6. Against q0/Delta = 32 that is 6 to 8%
    and unremarkable, which is why it was the third over-bound site to be found and the last. Against
    a bound of 2 it is over.

    Dividing further and multiplying back by an integer costs no level and no scale degree, so the
    identity is exact - and that is what this asserts, rather than a tolerance.
    """
    engine, stages = build(masks)
    rng = np.random.default_rng(31)
    values = [rng.normal(0, 1.5, D.slot_count) for _ in range(8)]
    encoded = [engine.encrypt(v) for v in values]

    seen = []

    class Probe(ClearEngine):
        def bootstrap(self, ct, keep_levels=None):
            seen.append(float(np.max(np.abs(ct.slots))))
            return super().bootstrap(ct, keep_levels)

    results, peaks = {}, {}
    for factor in (1.0, scale):
        probe = Probe(D, depth=DEPTH, bootstrap_level=DEPTH)
        scaled_stages = LayerNormStages(probe, D, masks=masks[0], complement_masks=masks[1])
        scaled_stages.refresh_scale = factor
        seen.clear()
        out = scaled_stages.refresh([probe.encrypt(v) for v in values])
        results[factor] = np.stack([np.real(np.asarray(probe.decrypt(ct))) for ct in out])
        peaks[factor] = max(seen)

    assert np.allclose(results[1.0], results[scale], rtol=1e-12, atol=1e-14), (
        f"refresh stopped being the identity: moved by "
        f"{np.max(np.abs(results[1.0] - results[scale])):.3g}")
    assert peaks[1.0] / peaks[scale] == pytest.approx(scale, rel=1e-9), (
        f"and it has to scale what is bootstrapped: {peaks[1.0]:.4g} -> {peaks[scale]:.4g}")
