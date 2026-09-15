"""Stage 8: the numeric kernels the softmax is built from - polynomial evaluation and exp.

Two separate things are checked, and keeping them apart is the point:

* the **port** is exact - `evaluate_polynomial` is `np.polyval` to machine precision, and `he_exp`
  reproduces THOR's own function (its coefficients, its shift, its squarings) to machine precision;
* the **approximation** is THOR's, not ours - how well those coefficients actually approximate an
  exponential is a property of the paper's fit, so it is measured and recorded rather than asserted
  tightly.

Everything runs on the numpy engine, which enforces the FIXEDMANUAL level and scale discipline, so the
level cost of the deepest arithmetic in the network is checked here too.
"""
import numpy as np
import pytest

from thorfhe import SMALL, ClearEngine, Stages, block_diagonal_masks
from thorfhe.numeric import (EXP1_COEFFICIENTS, EXP2_COEFFICIENTS, GELU_SCALE, DivisionMixin,
                             GeluMixin, InverseSqrtMixin, NumericMixin)

DEPTH = 20


class NumericStages(NumericMixin, Stages):
    pass


@pytest.fixture
def stages():
    engine = ClearEngine(SMALL, depth=DEPTH)
    low, high = block_diagonal_masks(SMALL)
    return engine, NumericStages(engine, SMALL, masks=low, complement_masks=high)


@pytest.mark.parametrize("count", [4, 8, 16])
def test_polynomial_evaluation_is_exact(stages, count):
    engine, st = stages
    rng = np.random.default_rng(41)
    coefficients = rng.normal(size=count) * 0.1
    x = rng.uniform(-1, 1, SMALL.slot_count)

    out = st.evaluate_polynomial(engine.encrypt(x), coefficients)
    assert np.abs(np.real(engine.decrypt(out)) - np.polyval(coefficients[::-1], x)).max() < 1e-12
    # baby-step giant-step over a degree-(count-1) polynomial: log2(count) + 1 levels
    assert engine.level(out) == DEPTH - (int(np.log2(count)) + 1)


def test_polynomial_rejects_awkward_degrees(stages):
    """Baby steps come in fours; anything else would need a different split."""
    _, st = stages
    with pytest.raises(ValueError, match="multiple of four"):
        st.evaluate_polynomial(None, np.zeros(6))


def test_power_basis_levels(stages):
    """`x^(2^k)` sits k levels below x; anything combining them has to align first."""
    engine, st = stages
    x = engine.encrypt(np.linspace(-1, 1, SMALL.slot_count))
    basis = st.power_basis(x, [2, 3, 4, 8])
    values = np.real(engine.decrypt(x))
    for power in (2, 3, 4, 8):
        assert np.abs(np.real(engine.decrypt(basis[power])) - values ** power).max() < 1e-9
    assert engine.level(basis[2]) == DEPTH - 1
    assert engine.level(basis[4]) == DEPTH - 2
    assert engine.level(basis[8]) == DEPTH - 3


@pytest.mark.parametrize("wide,coefficients,bounds,scale", [
    (False, EXP1_COEFFICIENTS, (-27.2493, 21.72692), 32),
    (True, EXP2_COEFFICIENTS, (-70.0, 70.0), 64),
])
def test_he_exp_reproduces_thor_exactly(stages, wide, coefficients, bounds, scale):
    """The port must compute THOR's function, whatever that function is worth."""
    engine, st = stages
    low, high = bounds
    rng = np.random.default_rng(42)
    u = rng.uniform(low, high, SMALL.slot_count)

    out = st.he_exp(engine.encrypt(u / scale), low, high, n=2, wide=wide)
    reference = np.polyval(coefficients[::-1], u / scale - (low + high) / 2 / scale) ** 2
    if wide:
        reference = reference * 128

    assert np.abs(np.real(engine.decrypt(out)) - reference).max() < 1e-9
    assert engine.level(out) == DEPTH - 6  # 5 for the degree-15 fit, 1 for the squaring


def test_he_exp1_is_an_exponential_to_the_precision_thor_chose(stages):
    """With n = 2 and one softmax doubling, he_exp1 gives exp(x - mid) - to about 0.03%.

    That is where the `n` and `l` in `he_softmax1` come from: the fit approximates exp(8t), the n = 2
    squaring doubles the exponent to 16t, and t = (x - mid)/32, so one more doubling inside
    `update_inv_D` lands exactly on exp(x - mid).
    """
    engine, st = stages
    low, high = -27.2493, 21.72692
    mid = (low + high) / 2
    rng = np.random.default_rng(43)
    u = rng.uniform(low, high, SMALL.slot_count)

    out = np.real(engine.decrypt(st.he_exp1(engine.encrypt(u / 32), low, high, n=2)))
    doubled = out ** 2  # the squaring update_inv_D applies

    # Only where the numerator still matters: the fit degrades in the far tail, but a softmax term
    # twelve orders of magnitude below the largest contributes nothing to the sum.
    usable = doubled > 1e-12 * doubled.max()
    fit = np.polyfit(u[usable] - mid, np.log(doubled[usable]), 1)
    assert abs(fit[0] - 1.0) < 1e-4, "one more doubling should land on exp(x - mid)"
    residual = np.abs(np.log(doubled[usable]) - np.polyval(fit, u[usable] - mid)).max()
    assert residual < 1e-3, f"THOR's fit is good to {residual:.5f} in log space"


def test_he_exp2_is_the_wider_weaker_fit(stages):
    """Layer 2 uses a wider range; the same construction is a much looser exponential there."""
    engine, st = stages
    low, high = -70.0, 70.0
    mid = 0.0
    rng = np.random.default_rng(44)
    u = rng.uniform(low, high, SMALL.slot_count)

    out = np.real(engine.decrypt(st.he_exp2(engine.encrypt(u / 64), low, high, n=2)))
    quadrupled = out ** 4  # he_softmax2 uses l = 4, i.e. two doublings

    usable = quadrupled > 1e-12 * quadrupled.max()
    fit = np.polyfit(u[usable] - mid, np.log(quadrupled[usable]), 1)
    assert abs(fit[0] - 1.0) < 0.02
    residual = np.abs(np.log(quadrupled[usable]) - np.polyval(fit, u[usable] - mid)).max()
    # recorded, not asserted tight: this is the accuracy THOR's own coefficients have over the wider
    # range layer 2 needs, and it is an order of magnitude worse than he_exp1's
    assert residual < 0.5


# ---------------------------------------------------------------- Goldschmidt division
class DivisionStages(NumericMixin, DivisionMixin, Stages):
    pass


@pytest.fixture
def division():
    depth = 60
    engine = ClearEngine(SMALL, depth=depth, bootstrap_level=depth)
    low, high = block_diagonal_masks(SMALL)
    return engine, DivisionStages(engine, SMALL, masks=low, complement_masks=high), depth


def used_slots():
    return (np.arange(SMALL.slot_count) % SMALL.n_slot) < SMALL.n_blocks


@pytest.mark.parametrize("epsilon,alpha,expected", [
    (2 ** -11, 0.001, 8),
    (2 ** -18, 0.001, 12),
])
def test_iteration_count_is_data_independent(epsilon, alpha, expected):
    """The loop bound depends only on the declared range, so the level cost is known up front."""
    assert DivisionMixin.goldschmidt_iterations(epsilon, alpha) == expected


def test_he_inv_inverts_over_its_declared_range(division):
    """1/D to within alpha for every D in [epsilon, 1], and nothing outside the used slots."""
    engine, st, depth = division
    epsilon, alpha = 2 ** -11, 0.001
    rng = np.random.default_rng(51)
    denominator = np.exp(rng.uniform(np.log(epsilon), 0.0, SMALL.slot_count))  # log-uniform
    used = used_slots()

    ones = engine.encrypt(used.astype(float))
    out, delta, precision = st.he_inv(engine.encrypt(denominator * used), ones,
                                      epsilon=epsilon, alpha=alpha)

    value = np.real(engine.decrypt(out)) / delta
    relative = np.abs(value[used] - 1.0 / denominator[used]) * denominator[used]
    assert relative.max() < alpha
    assert precision > 1 - alpha
    # the `ones` operand is what confines the result; padding must stay exactly empty
    assert np.abs(value[~used]).max() == 0.0
    assert engine.level(out) == depth - DivisionMixin.goldschmidt_iterations(epsilon, alpha)


def test_he_inv_refuses_a_denominator_below_its_declared_range(division):
    """The range in the docstring is a precondition, not advice, and nothing used to check it."""
    engine, st, _ = division
    epsilon = 2 ** -11
    used = used_slots()
    denominator = np.where(used, epsilon / 4, 0.0)

    with pytest.raises(ValueError, match="iteration is set up for"):
        st.he_inv(engine.encrypt(denominator), engine.encrypt(used.astype(float)),
                  epsilon=epsilon, alpha=0.001)


def test_he_inv_saturates_rather_than_failing_below_its_range(division):
    """What the check above exists to prevent: a finite, plausible, wrong answer.

    Goldschmidt's step count and its per-step `k` come from `epsilon` alone, so a denominator under
    it is not merely less accurate - the iteration runs out of schedule and the answer stops growing.
    Measured here, with the check disabled: at `epsilon/2` the result is 3.6% low, at `epsilon/4` 22%
    low, and by `epsilon/1000` it is pinned near 11500 instead of the 2.0e6 it should be. Nothing
    raises, nothing overflows, and a softmax built on it classifies at chance.
    """
    engine, st, _ = division
    epsilon, alpha = 2 ** -11, 0.001
    used = used_slots()
    errors = {}
    for ratio in (1.0, 0.5, 0.25, 0.001):
        value = epsilon * ratio
        st._check_inversion_range = lambda *a, **k: None   # the point is what happens without it
        out, delta, _ = st.he_inv(engine.encrypt(np.where(used, value, 0.0)),
                                  engine.encrypt(used.astype(float)), epsilon=epsilon, alpha=alpha)
        got = float(np.median((np.real(engine.decrypt(out)) / delta)[used]))
        errors[ratio] = abs(got - 1.0 / value) * value

    assert errors[1.0] < alpha, "inside the range it is accurate"
    assert 0.02 < errors[0.5] < 0.06, "half the bound already costs a few percent"
    assert 0.15 < errors[0.25] < 0.30, "a quarter of it costs a fifth of the answer"
    assert errors[0.001] > 0.9, "and far below, the answer saturates instead of growing"


def test_he_inv_accepts_the_whole_declared_range(division):
    """Both ends of [epsilon, 1] have to pass, or the check would be narrowing the contract."""
    engine, st, _ = division
    epsilon = 2 ** -11
    used = used_slots()
    for value in (epsilon, 1.0):
        out, delta, _ = st.he_inv(engine.encrypt(np.where(used, value, 0.0)),
                                  engine.encrypt(used.astype(float)), epsilon=epsilon, alpha=0.001)
        got = float(np.median((np.real(engine.decrypt(out)) / delta)[used]))
        assert abs(got - 1.0 / value) * value < 0.001


@pytest.mark.parametrize("precision_bits,diverges", [(22, False), (16, True), (12, True)])
def test_he_inv_needs_the_bootstrap_to_be_more_accurate_than_its_denominator(precision_bits,
                                                                            diverges):
    """The iteration refreshes the denominator first, so the refresh has to preserve it.

    A CKKS bootstrap's error is set by `q0/Delta`, not by the value it is handed, so refreshing a
    small number is a fixed absolute perturbation of it. `he_softmax` hands `he_inv` a denominator
    around 2e-4 while `q0/Delta` is 32: at 22 bits of the bound that is a 4% perturbation and the
    iteration absorbs it, at 16 bits it is 250% and the result runs away. This is not a modelling
    artefact - once the error reaches the value, there is no value left to invert.

    Parametrised rather than asserted at one point because what matters is the threshold, and where
    it sits depends on parameters we may yet change.
    """
    engine = ClearEngine(SMALL, depth=60, bootstrap_level=60, noise_model=True,
                         bootstrap_precision_bits=precision_bits, seed=3)
    low, high = block_diagonal_masks(SMALL)
    st = DivisionStages(engine, SMALL, masks=low, complement_masks=high)
    st.check_ranges = False    # the denominator is deliberately below range; that is a separate test
    used = used_slots()

    out, delta, _ = st.he_inv(engine.encrypt(np.where(used, 2.0e-4, 0.0)),
                              engine.encrypt(used.astype(float)), epsilon=2 ** -11, alpha=0.001)
    peak = float(np.max(np.abs((np.real(engine.decrypt(out)) / delta)[used])))

    if diverges:
        assert peak > 1e6, "the inverse should have run away, and it is the refresh that did it"
    else:
        assert peak < 1e4, "and at a faithful refresh it stays the size 1/D should be"


def test_he_inv_keeps_the_ciphertext_off_the_noise_floor(division):
    """The point of the delta bookkeeping: delta shrinks quadratically, the ciphertext must not.

    Without the free integer rescaling and conjugate doubling, the plaintext would sink by ~2^-60 over
    eight iterations and the result would be noise.
    """
    engine, st, _ = division
    epsilon, alpha = 2 ** -11, 0.001
    used = used_slots()
    denominator = np.where(used, 0.5, 0.0)

    out, delta, _ = st.he_inv(engine.encrypt(denominator), engine.encrypt(used.astype(float)),
                              epsilon=epsilon, alpha=alpha)
    magnitude = np.abs(np.real(engine.decrypt(out))).max()

    assert delta < 1e-2, "delta should have shrunk a long way"
    assert magnitude > 1e-3, "but the ciphertext itself must stay well above the noise floor"


# ---------------------------------------------------------------- inverse square root
class InverseSqrtStages(NumericMixin, InverseSqrtMixin, Stages):
    pass


@pytest.fixture
def inverse_sqrt():
    depth = 60
    engine = ClearEngine(SMALL, depth=depth, bootstrap_level=depth)
    low, high = block_diagonal_masks(SMALL)
    return engine, InverseSqrtStages(engine, SMALL, masks=low, complement_masks=high), depth


@pytest.mark.parametrize("epsilon,expected", [(0.015, 5), (0.0013, 6), (3e-4, 7)])
def test_invsqrt_converges_cubically(epsilon, expected):
    """Two more iterations cover fifty times the range - that is what buys LayerNorm its depth."""
    assert InverseSqrtMixin.invsqrt_iterations(epsilon, 0.001) == expected


def test_he_invsqrt_over_its_declared_range(inverse_sqrt):
    engine, st, depth = inverse_sqrt
    epsilon, alpha = 0.015, 0.001
    rng = np.random.default_rng(81)
    mask = used_slots().astype(float)
    denominator = np.exp(rng.uniform(np.log(epsilon), 0.0, SMALL.slot_count)) * mask

    out = st.he_invsqrt(engine.encrypt(denominator), engine.encrypt(mask), mask,
                        epsilon=epsilon, alpha=alpha)

    got = np.real(engine.decrypt(out))
    used = mask > 0
    relative = np.abs(got[used] - 1 / np.sqrt(denominator[used])) * np.sqrt(denominator[used])
    assert relative.max() < alpha
    # confined to the mask, which is what lets LayerNorm keep the statistic in a few slots per token
    assert np.abs(got[~used]).max() == 0.0
    assert engine.level(out) == depth - 2 * InverseSqrtMixin.invsqrt_iterations(epsilon, alpha)


# ---------------------------------------------------------------- GELU
class GeluStages(NumericMixin, GeluMixin, Stages):
    pass


@pytest.fixture
def gelu_stages():
    depth = 40
    engine = ClearEngine(SMALL, depth=depth)
    low, high = block_diagonal_masks(SMALL)
    return engine, GeluStages(engine, SMALL, masks=low, complement_masks=high), depth


def numpy_gelu(u):
    return 0.5 * u * (1 + np.tanh(np.sqrt(2 / np.pi) * (u + 0.044715 * u ** 3)))


def test_polynomial_accepts_any_multiple_of_four(stages):
    """THOR's outer GELU polynomial has 28 coefficients; zero-padding to 32 is free and identical."""
    engine, st = stages
    rng = np.random.default_rng(102)
    coefficients = rng.normal(size=28) * 0.01
    x = rng.uniform(-0.5, 0.5, SMALL.slot_count)
    out = st.evaluate_polynomial(engine.encrypt(x), coefficients)
    assert np.abs(np.real(engine.decrypt(out)) - np.polyval(coefficients[::-1], x)).max() < 1e-9


def test_gelu_matches_the_real_thing(gelu_stages):
    """`gelu(64x)` for a ciphertext carrying the pre-activation divided by 64."""
    engine, st, depth = gelu_stages
    rng = np.random.default_rng(101)
    pre_activation = rng.uniform(-GELU_SCALE, GELU_SCALE, SMALL.slot_count)

    out = st.gelu(engine.encrypt(pre_activation / GELU_SCALE))
    got = np.real(engine.decrypt(out))
    want = numpy_gelu(pre_activation)

    assert np.abs(got - want).max() < 1e-2
    # the composite costs twelve levels and the final multiply one more
    assert engine.level(out) == depth - 13


def test_gelu_is_accurate_where_it_matters(gelu_stages):
    """Relative accuracy on the values that survive the layer, not on the ones near zero."""
    engine, st, _ = gelu_stages
    rng = np.random.default_rng(103)
    pre_activation = rng.uniform(-GELU_SCALE, GELU_SCALE, SMALL.slot_count)
    got = np.real(engine.decrypt(st.gelu(engine.encrypt(pre_activation / GELU_SCALE))))
    want = numpy_gelu(pre_activation)
    large = np.abs(want) > 1.0
    assert (np.abs(got - want)[large] / np.abs(want)[large]).max() < 1e-3


def test_gelu_scale_is_a_convention_not_a_nicety(gelu_stages):
    """Feed the composite an argument outside [-1, 1] and the degree-31 fit stops being a tanh."""
    engine, st, _ = gelu_stages
    outside = np.full(SMALL.slot_count, 3.0)  # the composite's own argument, not the pre-activation
    tanh = np.real(engine.decrypt(st.he_tanh_for_gelu(engine.encrypt(outside))))
    assert np.abs(tanh).max() > 1.0, "outside its range the composite is not bounded by 1/2"

    inside = np.linspace(-1.0, 1.0, SMALL.slot_count)
    tanh = np.real(engine.decrypt(st.he_tanh_for_gelu(engine.encrypt(inside))))
    assert np.abs(tanh).max() < 0.51
