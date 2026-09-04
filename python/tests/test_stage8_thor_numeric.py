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
from thorfhe.numeric import EXP1_COEFFICIENTS, EXP2_COEFFICIENTS, NumericMixin

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
