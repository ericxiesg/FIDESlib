"""What the clear engine is supposed to refuse.

It is the FIXEDMANUAL contract checker, so the thing to test is not that it computes - the stage
tests cover that - but that it *stops*. Every case here is one the device either asserts only in a
debug build or does not check at all, which means a gap here is not a wrong answer where the mistake
was made: it is a plausible wrong answer many stages later, and two of these cost this project
several rounds each.
"""
import numpy as np
import pytest

from thorfhe.clear import ClearEngine, ScaleMismatch
from thorfhe.geometry import THOR_BERT


@pytest.fixture
def engine():
    return ClearEngine(THOR_BERT, depth=37, bootstrap_level=20)


def ct(engine, value=0.25, level=0):
    return engine.encrypt(np.full(engine.slots, value), level=level)


def unrescaled(engine, value=0.25):
    """A degree-1 ciphertext still at Delta^2: the state a scalar multiply leaves behind."""
    out = engine.multiply(ct(engine, value), 0.5)
    assert out.scale_exp == 2 and out.degree == 1
    return out


# ---------------------------------------------------------------- scale on the multiply path
def test_ciphertext_product_refuses_an_unrescaled_operand(engine):
    """`Ciphertext::mult` asserts NoiseLevel == 1 for both sides, and asserts are gone in release."""
    with pytest.raises(ScaleMismatch, match="not canonical"):
        engine.multiply(unrescaled(engine), unrescaled(engine))


def test_ciphertext_product_refuses_an_unrescaled_operand_on_either_side(engine):
    for a, b in ((unrescaled(engine), ct(engine)), (ct(engine), unrescaled(engine))):
        with pytest.raises(ScaleMismatch, match="not canonical"):
            engine.multiply(a, b)


def test_plaintext_multiply_refuses_an_unrescaled_ciphertext(engine):
    """FIDESlib holds scaling factors for NoiseLevel 1 and 2 only; a Delta^3 product has none."""
    with pytest.raises(ScaleMismatch, match="not canonical"):
        engine.multiply(unrescaled(engine), np.ones(engine.slots))


def test_plaintext_multiply_refuses_an_unrescaled_ciphertext(engine):
    # multPt on the device asserts NoiseLevel < 2 (Ciphertext.cpp:478); a float scalar
    # (multScalar) does NOT assert and is legal on degree-2, so we test the plaintext path.
    pt = np.ones(engine.slots)
    with pytest.raises(ScaleMismatch, match="not canonical"):
        engine.multiply(unrescaled(engine), pt)


def test_integer_multiply_accepts_an_unrescaled_ciphertext(engine):
    """Deliberately not guarded: `multIntScalar` touches no metadata, and `_restore_magnitude`
    relies on exactly that."""
    out = engine.multiply(unrescaled(engine), 3)
    assert out.scale_exp == 2 and out.level == engine.depth


def test_a_rescale_makes_the_product_legal_again(engine):
    ok = engine.multiply(engine.rescale(unrescaled(engine)), engine.rescale(unrescaled(engine)))
    assert ok.scale_exp == 2 and ok.degree == 2


# ---------------------------------------------------------------- degree through add_inplace
def test_add_inplace_keeps_the_degree_of_the_operand_that_had_one(engine):
    """The device's add builds a c2 when either side carries one, so the result is degree 2.

    Dropping it here is not a bookkeeping nicety: rotate, multiply_1j and bootstrap all discard c2
    without saying so, and the degree is the guard that stops them. This is how `power_basis` once
    got squared without relinearising and made he_exp return 1e124 on the device.
    """
    lazy = engine.rescale(engine.multiply(ct(engine), ct(engine)))
    assert lazy.degree == 2
    target = engine.rescale(engine.multiply(ct(engine), 0.5))
    engine.add_inplace(target, lazy)
    assert target.degree == 2


def test_add_inplace_result_is_still_refused_by_the_key_switching_operations(engine):
    lazy = engine.rescale(engine.multiply(ct(engine), ct(engine)))
    target = engine.rescale(engine.multiply(ct(engine), 0.5))
    engine.add_inplace(target, lazy)
    for operation in (lambda: engine.rotate(target, 16),
                      lambda: engine.conjugate(target),
                      lambda: engine.multiply_1j(target),
                      lambda: engine.bootstrap(target)):
        with pytest.raises(ScaleMismatch, match="degree"):
            operation()


def test_add_inplace_of_two_degree_one_operands_stays_degree_one(engine):
    target, other = ct(engine), ct(engine)
    engine.add_inplace(target, other)
    assert target.degree == 1


# ---------------------------------------------------------------- bootstrap contract
def test_bootstrap_refuses_a_message_at_the_modulus_bound(engine):
    """ModRaise leaves `m + q0*I`; the sine only approximates the reduction near zero."""
    big = engine.encrypt(np.full(engine.slots, 2.0 * engine.message_bound))
    with pytest.raises(ScaleMismatch, match="q0/Delta"):
        engine.bootstrap(big)


def test_bootstrap_accepts_a_message_inside_the_bound(engine):
    ok = engine.bootstrap(ct(engine, engine.message_bound / 8))
    assert ok.level == engine.bootstrap_level and ok.scale_exp == 1


def test_bootstrap_refuses_to_pretend_it_produced_more_levels_than_it_has(engine):
    """`Engine.bootstrap` raises on the shortfall rather than returning fewer levels than asked."""
    with pytest.raises(ScaleMismatch, match="keep_levels"):
        engine.bootstrap(ct(engine), keep_levels=engine.bootstrap_level + 1)


def test_bootstrap_honours_a_keep_levels_it_can_meet(engine):
    out = engine.bootstrap(ct(engine), keep_levels=engine.bootstrap_level - 3)
    assert out.level == engine.bootstrap_level - 3


# ---------------------------------------------------------------- aliasing
def test_relinearize_does_not_alias_its_input(engine):
    lazy = engine.multiply(ct(engine), ct(engine))
    out = engine.relinearize(lazy)
    assert out.slots is not lazy.slots
    out.slots[0] = 12345.0
    assert lazy.slots[0] != 12345.0


# ---------------------------------------------------------------- the noise model
def test_noise_model_is_off_by_default(engine):
    """The default has to stay an exact oracle, or the stage tests cannot compare against it."""
    x = np.arange(engine.slots) / engine.slots
    assert np.array_equal(np.real(engine.decrypt(engine.encrypt(x))), x)
    assert np.array_equal(np.real(engine.decrypt(engine.bootstrap(engine.encrypt(x)))), x)


def test_bootstrap_error_is_set_by_the_bound_not_by_the_value():
    """Why THOR keeps magnitudes up: refreshing a small number costs its significant digits.

    A bootstrap recovers `message_bound` to a fixed number of bits, so its absolute error does not
    shrink with the message. At the benchmark's parameters that error is about 7.6e-6 whether the
    value is 8 or 2e-4 - negligible for the first, and a percent of the second.
    """
    noisy = ClearEngine(THOR_BERT, depth=37, bootstrap_level=20, noise_model=True, seed=7)
    relative = {}
    for value in (8.0, 2e-4):
        got = np.real(noisy.decrypt(noisy.bootstrap(ct(noisy, value))))
        relative[value] = float(np.max(np.abs(got - value))) / value
    assert relative[8.0] < 1e-5, "negligible relative to a large message"
    assert relative[2e-4] > 1e-3, "but not relative to a small one"


def test_noise_model_leaves_the_cheap_operations_far_below_the_bootstrap():
    """The ordering matters more than the figures: everything else is orders below the refresh."""
    noisy = ClearEngine(THOR_BERT, depth=37, bootstrap_level=20, noise_model=True, seed=11)
    x = ct(noisy, 0.25)
    fresh = float(np.max(np.abs(np.real(noisy.decrypt(x)) - 0.25)))
    keyswitch = float(np.max(np.abs(np.real(noisy.decrypt(noisy.rotate(x, 16))) - 0.25)))
    refreshed = float(np.max(np.abs(np.real(noisy.decrypt(noisy.bootstrap(x))) - 0.25)))
    assert fresh < 1e-10 and keyswitch < 1e-10
    assert refreshed > 1e4 * keyswitch
