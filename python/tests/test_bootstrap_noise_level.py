"""Does bootstrap return a canonical ciphertext?

The question is whether the result is at scale Delta or Delta^N, and it is worth asking directly:
several rounds of this project went into inferring a scale degree from a wrong answer downstream.

An earlier version of this file inferred it from `addScalar(1.0)`, on the reasoning that adding a
constant encoded at Delta to a ciphertext at Delta^N would move the value by 1/Delta^(N-1) instead
of by 1. That reasoning does not hold: `Ciphertext::addScalar` passes `this->NoiseLevel` to
`ElemForEvalAddOrSub`, which multiplies the constant by the scaling factor that many times
(Context.cu, the loop over `noise_deg`). The constant is encoded to match, the sum is right either
way, and the check could not fail for the reason it was written to detect.

Two things settle it instead: `noise_level()` reads the field, and adding a *plaintext* - which is
always encoded at Delta^1 - trips the FIXEDMANUAL guard in `addPt` if the ciphertext is not at
Delta^1 too.
"""
import os

import numpy as np
import pytest


BENCH = dict(log_n=16, depth=37, scaling_bits=50, first_mod_bits=55, dnum=4)


def _values(slots):
    x = np.zeros(slots, dtype=complex)
    x[0], x[1], x[2] = 2.0, -3.0, 0.5
    return x


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_bootstrap_returns_a_canonical_ciphertext(device):
    import pyfideslib as pf

    engine = pf.Engine(device, **BENCH)
    x = _values(engine.slots)

    ct = engine.bootstrap(engine.encrypt(x))
    got = np.real(np.asarray(engine.decrypt(ct)))[:3]
    assert np.max(np.abs(got - np.real(x[:3]))) < 1e-3, f"bootstrap changed the value: {got}"

    # The field itself, now that it is readable.
    assert engine.noise_level(ct) == 1, (
        f"bootstrap left the ciphertext at scale degree {engine.noise_level(ct)}; "
        "anything that adds a plaintext to it afterwards is wrong by that many factors of Delta")

    # And the consequence, through the guard: a plaintext is always encoded at Delta^1, so this
    # throws rather than returning a plausible wrong number if the ciphertext is not.
    summed = engine.add(ct, np.ones(engine.slots))
    moved = np.real(np.asarray(engine.decrypt(summed)))[:3] - got
    assert np.max(np.abs(moved - 1.0)) < 1e-3, f"adding a plaintext 1 moved values by {moved}"


def test_noise_level_tracks_a_multiplication(engine):
    """The accessor means what it says, at parameters small enough to run in the normal suite."""
    from conftest import rand

    x = engine.encrypt(rand(engine, 31, scale=0.4))
    assert engine.noise_level(x) == 1
    product = engine.multiply(x, 0.5)
    assert engine.noise_level(product) == 2, "a float scalar multiply costs a scale degree"
    assert engine.noise_level(engine.rescale(product)) == 1, "rescale returns it"
