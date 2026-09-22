"""A negative denominator diverges the Goldschmidt iteration, and the floor stops it.

This is the clear-engine half of the `he_inv` investigation, so it needs no device. The device's
bootstrap carries 0.017 of absolute error (10.9 bits, see `test_bootstrap_noise_level`) and `he_inv`
bootstraps its denominator. Outside the support the denominator is exactly zero beforehand, so
afterwards half of those slots are negative - and Goldschmidt started from a positive `ones` cannot
converge to `1/x` for a negative `x`, it grows every step. Injecting that error reproduces the device
exactly: `b` first moves at iteration 3 and is four orders out by iteration 5.
"""
import os

import numpy as np
import pytest

from thorfhe.clear import ClearEngine
from thorfhe.encoding import block_diagonal_masks
from thorfhe.geometry import THOR_BERT
from thorfhe.numeric import PADDING_FLOOR, DivisionMixin, NumericMixin
from thorfhe.stages import Stages


class Division(NumericMixin, DivisionMixin, Stages):
    pass


def _run(with_support, boot_error=0.017, floor=None):
    low, high = block_diagonal_masks(THOR_BERT)
    slots = THOR_BERT.slot_count
    live = np.arange(0, slots, 128)
    support = np.zeros(slots)
    support[live] = 1.0
    denominator = np.zeros(slots)
    denominator[live] = np.linspace(0.05, 0.9, live.size)

    import thorfhe.numeric as numeric
    previous = numeric.PADDING_FLOOR
    if floor is not None:
        numeric.PADDING_FLOOR = floor

    engine = ClearEngine(THOR_BERT, depth=37, bootstrap_level=20)
    stages = Division(engine, THOR_BERT, masks=low, complement_masks=high)
    stages.check_ranges = False
    rng = np.random.default_rng(1)
    exact = Division.bootstrap

    def noisy(self, x, keep_levels=None):
        out = exact(self, x, keep_levels)
        out.slots = out.slots + boot_error * rng.standard_normal(out.slots.size)
        return out

    # The divergence is in `b`, the denominator branch, and it does not reach the returned value:
    # `a` starts from `ones`, which is zero outside the support, so the answer stays confined even
    # while the iteration is falling apart inside. That is why this reads the per-iteration probe
    # rather than the result - and why the bug survived so long behind a plausible-looking output.
    trace = {}
    stages.probe = lambda name, cts, mask=None: trace.__setitem__(
        name, float(np.abs(np.real(engine.decrypt(cts[0]))).max()))

    Division.bootstrap = noisy
    os.environ["THORFHE_DEBUG"] = "1"
    try:
        stages.he_inv(engine.encrypt(denominator), engine.encrypt(support),
                      epsilon=2 ** -11, alpha=0.001,
                      support=support if with_support else None)
    finally:
        os.environ.pop("THORFHE_DEBUG", None)
        Division.bootstrap = exact
        numeric.PADDING_FLOOR = previous
    return max(value for name, value in trace.items() if name.endswith("_b"))


def test_the_bug_reproduces_without_the_floor():
    assert _run(with_support=False) > 1e3


def test_the_floor_stops_it():
    assert _run(with_support=True) < 1e2


def test_the_floor_has_to_sit_inside_the_range_not_at_its_top():
    """1.0 is the top of `[epsilon, 1]` and does not work - the same error lands on the filler.

    A slot filled to 1.0 comes out at 1.017, which is outside the window the schedule was derived
    for, and the iteration diverges from above it instead of from below. This is why PADDING_FLOOR
    is a half and not a one, and it is the kind of choice that looks arbitrary a year later.
    """
    assert _run(with_support=True, floor=1.0) > 1e3
    assert _run(with_support=True, floor=0.25) < 1e2
    assert _run(with_support=True, floor=0.75) < 1e2
    assert 0 < PADDING_FLOOR < 1


# --------------------------------------------------------------------------- the integer lift
def _invert(lift, boot_error=0.0, top=0.2925):
    """1/D over a denominator spanning [2^-6, top], the way layer 0's does."""
    low, high = block_diagonal_masks(THOR_BERT)
    slots = THOR_BERT.slot_count
    live = np.arange(0, slots, 128)
    support = np.zeros(slots)
    support[live] = 1.0
    denominator = np.zeros(slots)
    denominator[live] = np.linspace(2.0 ** -6, top, live.size)

    engine = ClearEngine(THOR_BERT, depth=37, bootstrap_level=20)
    stages = Division(engine, THOR_BERT, masks=low, complement_masks=high)
    rng = np.random.default_rng(1)
    exact = Division.bootstrap

    def noisy(self, x, keep_levels=None):
        out = exact(self, x, keep_levels)
        if boot_error:
            out.slots = out.slots + boot_error * rng.standard_normal(out.slots.size)
        return out

    Division.bootstrap = noisy
    try:
        out, delta, _ = stages.he_inv(engine.encrypt(denominator), engine.encrypt(support),
                                      epsilon=2.0 ** -6, alpha=0.001, support=support, lift=lift)
    finally:
        Division.bootstrap = exact
    got = np.real(engine.decrypt(out))[live] / delta
    return float((np.abs(got - 1.0 / denominator[live]) / (1.0 / denominator[live])).max())


def test_the_lift_does_not_change_the_answer_in_exact_arithmetic():
    """It multiplies before the refresh and divides out of the returned delta: the value is the same."""
    assert _invert(lift=1) < 1e-5
    assert _invert(lift=3) < 1e-4


def test_the_lift_is_what_makes_a_low_denominator_survive_the_bootstrap():
    """0.017 of absolute error against layer 0's epsilon of 2^-6 leaves the bottom of the range noise."""
    assert _invert(lift=1, boot_error=0.017) > 10
    assert _invert(lift=3, boot_error=0.017) < 1.0


def test_a_lift_that_would_push_the_denominator_past_one_is_refused():
    """The range check runs after the lift, so an unsafe one raises instead of quietly diverging."""
    with pytest.raises(Exception):
        _invert(lift=4)


def test_the_lift_is_already_at_its_ceiling_and_the_bootstrap_is_what_is_left():
    """Raising the lift always helps and cannot go past 3, which is not enough at 0.017 of error.

    The denominator `07c` hands the first `he_inv` spans [2^-6, 0.3036] on the real checkpoint - a
    19.4:1 dynamic range. The top caps the lift at 1/0.3036 = 3.29, because above 1 the correction
    `2 - k*b` changes sign and Goldschmidt inverts instead of converging. So `--inverse-lift 3` is
    essentially maximal, and the sweep below says it is also strictly best: at every error level,
    more lift is less error. Dropping the lift to buy headroom would make this worse, not better.

    What that leaves is the bootstrap. At lift 3 the *smallest* carried denominator sits at 0.0468
    against an absolute bootstrap error of 0.017 - a signal-to-noise ratio of 2.8 - and the relative
    error on 1/D comes out proportional to it, about 36x the bootstrap's. That is the whole budget:
    no arrangement of level-free integers fixes a denominator whose bottom is three times its noise.
    """
    top = 0.3036
    assert 1.0 / top == pytest.approx(3.29, abs=0.01), "the cliff is what caps the lift"

    # More lift is less error, at the error the device is documented to have.
    errors = [_invert(lift=lift, boot_error=0.017, top=top) for lift in (1, 2, 3)]
    assert errors[0] > errors[1] > errors[2], f"the lift stopped helping: {errors}"

    # And at the ceiling it is still 61% - the iteration is not usable at 0.017, at any lift.
    assert 0.4 < errors[2] < 0.9
    # Linear in the bootstrap's error, so the precision needed is a number and not a guess.
    assert _invert(lift=3, boot_error=0.001, top=top) < 0.05


def test_the_range_check_runs_before_the_bootstrap_and_so_cannot_see_the_cliff():
    """`_check_inversion_range` guards `lift * denominator`; the iteration gets it bootstrapped.

    The bootstrap's error is absolute and lands after the guard, so the guard's "under 1" and the
    iteration's "under 1" are different numbers. That is why `inv_input_lift{N}` probes after the
    refresh: it is the only place the value the iteration actually receives can be read.

    Here the lift is statically safe - 3 x 0.3036 = 0.911, which the guard passes - and 0.09 of
    bootstrap error still destroys the result, by pushing carried slots across one end or the other.
    """
    top = 0.3036
    assert 3 * top < 1.0, "the lift must be statically safe, or this tests the guard instead"
    assert _invert(lift=3, boot_error=0.09, top=top) > 1e3
