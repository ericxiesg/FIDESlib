"""A seconds-long reproducer for the he_exp blow-up, instead of a 25-minute benchmark.

The GPU probes put the first corruption at `he_exp`: the refreshed scores going in are correct
(range [-10.43, +12.48]) and the polynomial's output comes back at 1e124. Stages 01-06 are exact to
3e-7 and never take this path - they multiply by mask *plaintexts* and by *integers*, while
`evaluate_polynomial` is built from float-scalar multiplies over a lazily-relinearised power basis.

`test_stage2_linear.py` already covers one float-scalar multiply followed by a rescale, and it
passes on the GPU, so the single operation is not the problem. What is untested is the composition:
a degree-15 polynomial whose power basis stays degree 2 between multiplication and relinearisation,
whose terms are aligned down to a common level, and whose constant term arrives as a scalar.

Every check runs twice: once against exact arithmetic, which proves the check says what it means,
and once against the device, which is the one being asked. The two halves skip independently, so
this file is useful with or without the extension built.

Each check isolates one primitive, so whichever fails names what to look at.
"""
import os

import numpy as np
import pytest

from conftest import rand

from thorfhe.clear import ScaleMismatch
from thorfhe.numeric import EXP1_COEFFICIENTS, NumericMixin
from thorfhe.stages import Stages


class Numeric(NumericMixin, Stages):
    """Just the polynomial machinery. `evaluate_polynomial`, `power_basis` and `align` use only
    engine primitives, so the geometry is never touched and may be None."""


@pytest.fixture(scope="module")
def clear_pair():
    from thorfhe.clear import ClearEngine
    from thorfhe.geometry import SMALL as SMALL_GEOMETRY

    engine = ClearEngine(SMALL_GEOMETRY, depth=30)
    return Numeric(engine, SMALL_GEOMETRY), engine


@pytest.fixture
def device_pair(engine):
    return Numeric(engine, None), engine


def _decrypt(engine, ct, length):
    return np.real(np.asarray(engine.decrypt(ct)))[:length]


# ---------------------------------------------------------------- the checks
def _power_basis(numeric, engine):
    """x^2, x^3, x^4, x^8 through square/multiply with lazy relinearisation.

    The first thing `evaluate_polynomial` does, and the only place the port builds a degree-2
    ciphertext and keeps computing with it.
    """
    x = rand(engine, 11, scale=0.4)
    basis = numeric.power_basis(engine.encrypt(x), [2, 3, 4, 8])
    for power in (2, 3, 4, 8):
        got = _decrypt(engine, basis[power], len(x))
        assert np.max(np.abs(got - x[:len(got)] ** power)) < 1e-5, f"x^{power}"


def _float_scalar_on_degree_two(numeric, engine):
    """A float scalar times a ciphertext that has not been relinearised yet.

    multScalar has to scale the third component as well as c0 and c1, or the relinearisation
    afterwards folds in a term that was left behind.
    """
    x = rand(engine, 12, scale=0.4)
    cx = engine.encrypt(x)
    squared = engine.multiply(cx, cx)                       # degree 2, scale Delta^2
    scaled = engine.rescale(engine.multiply(squared, 0.375))
    got = _decrypt(engine, engine.relinearize(scaled), len(x))
    assert np.max(np.abs(got - 0.375 * x[:len(got)] ** 2)) < 1e-5


def _scalar_added_to_degree_two(numeric, engine):
    """A constant added to a degree-2 ciphertext, which addScalar puts into c0 alone.

    The ciphertext is rescaled first: adding a fresh plaintext to an unrescaled product would be a
    scale mismatch, and the whole point of FIXEDMANUAL is that this is the caller's job.
    """
    x = rand(engine, 13, scale=0.4)
    cx = engine.encrypt(x)
    squared = engine.rescale(engine.multiply(cx, cx))       # degree 2, back to Delta^1
    shifted = engine.add(squared, -0.25)
    got = _decrypt(engine, engine.relinearize(shifted), len(x))
    assert np.max(np.abs(got - (x[:len(got)] ** 2 - 0.25))) < 1e-5


def _small_polynomial(numeric, engine):
    """Degree 7: the smallest shape with a giant step."""
    coefficients = np.array([0.5, -1.25, 0.75, 2.0, -0.5, 0.25, 1.5, -0.75])
    x = rand(engine, 14, scale=0.4)
    got = _decrypt(engine, numeric.evaluate_polynomial(engine.encrypt(x), coefficients), len(x))
    assert np.max(np.abs(got - np.polyval(coefficients[::-1], x[:len(got)]))) < 1e-4


def _exp_polynomial(numeric, engine):
    """THOR's degree-15 exponential fit, over the range he_softmax hands it.

    This is the operation the probe caught returning 1e124. The input is the measured refreshed-score
    range divided by the narrow path's scale of 32, so a failure here is the benchmark's failure in
    about a second.
    """
    rng = np.random.default_rng(15)
    x = rng.uniform(-10.43 / 32, 12.48 / 32, engine.slots)
    got = _decrypt(engine, numeric.evaluate_polynomial(engine.encrypt(x), EXP1_COEFFICIENTS), len(x))
    want = np.polyval(np.asarray(EXP1_COEFFICIENTS)[::-1], x[:len(got)])

    assert np.isfinite(got).all(), "the polynomial produced non-finite values"
    assert np.max(np.abs(got)) < 1e3, f"blew up to {np.max(np.abs(got)):.3g}, expected order 1e-2"
    assert np.max(np.abs(got - want)) < 1e-3


CHECKS = [_power_basis, _float_scalar_on_degree_two, _scalar_added_to_degree_two,
          _small_polynomial, _exp_polynomial]
IDS = [check.__name__.lstrip("_") for check in CHECKS]


@pytest.mark.parametrize("check", CHECKS, ids=IDS)
def test_exact(check, clear_pair):
    check(*clear_pair)


@pytest.mark.parametrize("check", CHECKS, ids=IDS)
def test_device(check, device_pair):
    check(*device_pair)


# ---------------------------------------------------------------- at the benchmark's parameters
#: The checks above run at SMALL (log_n 13, depth 12). The benchmark runs he_exp at log_n 16,
#: depth 37, with the ciphertext around level 18 - and a fault that depends on the level or on the
#: modulus chain's length would not show at SMALL. This builds the real thing. It is opt-in because
#: key generation at those parameters is minutes, not milliseconds - but still far less than the
#: 25-minute benchmark it replaces.
BENCH = dict(log_n=16, depth=37, scaling_bits=50, first_mod_bits=55, dnum=4)


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_exp_polynomial_at_benchmark_parameters(device):
    import pyfideslib as pf

    engine = pf.Engine(device, **BENCH)
    numeric = Numeric(engine, None)

    rng = np.random.default_rng(15)
    x = rng.uniform(-10.43 / 32, 12.48 / 32, engine.slots)
    # he_exp sees the scores after a bootstrap and a level_down(3); start where the probe found them.
    cx = engine.encrypt(x)
    cx = engine.level_down(cx, max(engine.level(cx) - 18, 0))
    got = np.real(np.asarray(engine.decrypt(numeric.evaluate_polynomial(cx, EXP1_COEFFICIENTS))))[:len(x)]
    want = np.polyval(np.asarray(EXP1_COEFFICIENTS)[::-1], x[:len(got)])

    assert np.isfinite(got).all(), "the polynomial produced non-finite values"
    assert np.max(np.abs(got)) < 1e3, f"blew up to {np.max(np.abs(got)):.3g}, expected order 1e-2"
    assert np.max(np.abs(got - want)) < 1e-3


# ---------------------------------------------------------------- the contract, in exact arithmetic
def test_the_old_power_basis_order_is_now_caught(clear_pair):
    """The exact mistake that produced 1e124, as a loud contract violation with no device involved.

    `power_basis` used to square *after* relinearising rather than before, so the basis element it
    stored was still degree 2. Multiplying that by another ciphertext is what the device mishandles:
    it drops the third component and returns a plausible wrong answer. Exact arithmetic had no third
    component to lose and stayed happy, which is why this cost several GPU runs to find. Now the
    clear engine refuses it.
    """
    _, engine = clear_pair
    x = engine.encrypt(rand(engine, 21, scale=0.4))
    stale = engine.rescale(engine.square(engine.relinearize(x)))     # relinearise, then square
    assert stale.degree == 2

    with pytest.raises(ScaleMismatch):
        engine.multiply(stale, x)
    with pytest.raises(ScaleMismatch):
        engine.rotate(stale, 1)
    with pytest.raises(ScaleMismatch):
        engine.multiply_1j(stale)


def test_the_current_power_basis_order_is_canonical(clear_pair):
    """And the order it uses now leaves every basis element degree 1, as the docstring promises."""
    numeric, engine = clear_pair
    basis = numeric.power_basis(engine.encrypt(rand(engine, 22, scale=0.4)), [2, 3, 4, 8])
    assert all(basis[power].degree == 1 for power in basis), {k: v.degree for k, v in basis.items()}
