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
from thorfhe.numeric import (EXP1_COEFFICIENTS, EXP2_COEFFICIENTS, DivisionMixin,
                             InverseSqrtMixin, NumericMixin)

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
    _, st = stages
    with pytest.raises(ValueError, match="power of two"):
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
