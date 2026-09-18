"""The four primitives `DivisionMixin.he_inv` builds its iteration out of, on the device.

A device run diverges inside this iteration: `07c.denominator` arrives correct (0.2925 against the
clear engine's 0.3036) and `07d.inverse_denominator` leaves at 5.3e16, with the per-iteration probes
putting the first bad step at `iter03_b`. Injecting noise on the clear engine does not reproduce it -
leaking the denominator's empty slots does nothing, leaking `ones` amplifies by 1.158e4 but only into
`a`, and relative noise on the slots that carry data passes through linearly - so the device's `b` is
not a noisier value than ours, it is a different one. That makes it an operation, and these are the
operations.

`test_stage2_linear` already covers `subtract(scalar, ct)`, but only at a fresh level and only at
2.0. What the iteration does is different in two ways that matter: the scalars are small and shrink
(1.00, 0.251, 0.0158, 0.00401 - `iter03`'s is the one where `2/k * delta_b` and `b` are closest in
magnitude), and it happens with few levels left. Both are covered here, and the last test runs the
step itself rather than its pieces, because a compounding error is what the device shows.
"""
import numpy as np
import pytest

pf = pytest.importorskip("pyfideslib", reason="the pyfideslib extension is not built")
from conftest import SMALL, rand

#: `2/k * b.delta` at each iteration, measured from a real `he_inv` at THOR's geometry with
#: epsilon = 2^-11. They shrink, and `iter03`'s 0.0158 sits right on top of `b`'s own largest slot
#: (0.0164 on the clear engine), which is why that step was the first suspect.
SCALARS = [1.00049, 0.250732, 0.0158389, 0.00400755, 0.00435272]

#: The integers `_restore_magnitude` multiplies by, same run. Level-free on both engines, so the
#: value must come back exactly scaled and the level must not move.
FACTORS = [31, 486, 412, 272, 161, 129]

#: Levels consumed before the operation. The iteration runs with few levels left, and a truncated
#: rotation or a scale that only holds near the top would not show at 0.
DROPS = [0, 4, 8, 10]


def _at(engine, ct, drop):
    return engine.level_down(ct, by=drop) if drop else ct


@pytest.mark.parametrize("scalar", SCALARS)
@pytest.mark.parametrize("drop", DROPS)
def test_scalar_minus_ciphertext(engine, scalar, drop):
    """`subtract(s, ct)` is how the Goldschmidt correction is formed, and `b` passes only through it."""
    x = rand(engine, 41, scale=0.5)
    ct = _at(engine, engine.encrypt(x), drop)
    got = engine.decrypt_real(engine.subtract(scalar, ct))
    assert np.max(np.abs(got - (scalar - x))) < 1e-5, (
        f"scalar {scalar} at level {engine.level(ct)}: max |got| {np.abs(got).max():.4g}")


@pytest.mark.parametrize("factor", FACTORS)
@pytest.mark.parametrize("drop", DROPS)
def test_integer_scalar_is_exact_and_level_free(engine, factor, drop):
    """`multiply(ct, int)` routes to `EvalMultByInteger`, which must cost no level and no scale."""
    x = rand(engine, 42, scale=0.5)
    ct = _at(engine, engine.encrypt(x), drop)
    out = engine.multiply(ct, factor)
    assert engine.level(out) == engine.level(ct), "an integer scalar must not consume a level"
    got = engine.decrypt_real(out)
    assert np.max(np.abs(got - factor * x)) < 1e-5 * factor


@pytest.mark.parametrize("drop", DROPS)
def test_doubling_through_the_conjugate(engine, drop):
    """`add(ct, conjugate(ct))` is how the iteration doubles without spending a level."""
    x = rand(engine, 43, scale=0.5)
    ct = _at(engine, engine.encrypt(x), drop)
    out = engine.add(ct, engine.conjugate(ct))
    assert engine.level(out) == engine.level(ct)
    assert np.max(np.abs(engine.decrypt_real(out) - 2 * x)) < 1e-5


def test_a_goldschmidt_step_does_not_compound(engine):
    """The step itself, five times, at the magnitudes `he_inv` reaches.

    Each piece can be within tolerance while the composition is not: the step multiplies by a
    correction formed from a subtraction, and an error there is fed back into the next correction.
    The device's `b` goes from 0.0039 to 219.6 in one step and then squares, so what is being checked
    is that the error stays proportional to the input rather than growing with the iteration.
    """
    b = np.full(engine.slots, 0.0164)
    b[::7] = 0.0                       # the empty slots, where the correction does not cancel
    ct = engine.encrypt(b)
    want = b.copy()

    for step, scalar in enumerate(SCALARS, start=1):
        correction = engine.subtract(scalar, ct)
        ct = engine.rescale(engine.relinearize(engine.multiply(ct, correction)))
        want = want * (scalar - want)
        got = engine.decrypt_real(ct)

        # Magnitude, not precision. `want` decays to about 1e-9 by the last step, and asserting on
        # the difference there would be asserting on the noise floor - a brittle test that fails for
        # the wrong reason. What the device does is grow: 0.0039 to 219.6 in one step and then
        # square. An order of magnitude of headroom catches that and nothing else.
        assert np.abs(got).max() < 10 * np.abs(want).max() + 1e-6, (
            f"step {step} (scalar {scalar}) grew: max |got| {np.abs(got).max():.4g} "
            f"against an expected {np.abs(want).max():.4g}")
        if engine.level(ct) < 2:
            break
