"""Stage 11/16: LayerNorm.

The mean and variance of a token's 768 features have to come out of a representation that spreads them
over 8 ciphertexts, 16 groups and 6 slots. THOR reduces them into slot 0, inverts the square root
there, and broadcasts back - so the expensive part runs on one slot in sixteen.

Two things the tests pin down beyond the arithmetic: every variant returns **twice** the LayerNorm (the
final doubling is never cancelled), and the token variance has to sit inside the variant's window or
the inverse square root has nothing to converge to. The `/2` in variants 2 and 3 moves that window up
by four rather than cancelling the doubling, which is the easy thing to get backwards.
"""
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
