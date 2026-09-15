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


def bench_params():
    """The engine `thorfhe.bench` actually builds, resolved lazily so this module imports without
    the extension.

    Matching it exactly matters: FIDESlib selects a different Chebyshev coefficient set for the
    bootstrap according to the secret key distribution, so an engine built with the other one is
    not testing the configuration that runs.
    """
    import pyfideslib as pf

    return dict(log_n=16, depth=37, scaling_bits=50, first_mod_bits=55, dnum=3,
                secret_key_dist=pf.SPARSE_TERNARY,
                bootstrap_level_budget=(3, 3),
                rotation_indexes=[1 << i for i in range(15)],
                truncate_keys=True, allow_key_grow=True)


def _values(slots):
    x = np.zeros(slots, dtype=complex)
    x[0], x[1], x[2] = 2.0, -3.0, 0.5
    return x


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_bootstrap_returns_a_canonical_ciphertext(device):
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
    x = _values(engine.slots)

    ct = engine.bootstrap(engine.encrypt(x))
    got = np.real(np.asarray(engine.decrypt(ct)))[:3]
    assert np.max(np.abs(got - np.real(x[:3]))) < 0.05, f"bootstrap changed the value: {got}"

    # The field itself, now that it is readable.
    assert engine.noise_level(ct) == 1, (
        f"bootstrap left the ciphertext at scale degree {engine.noise_level(ct)}; "
        "anything that adds a plaintext to it afterwards is wrong by that many factors of Delta")

    # And the consequence, through the guard: a plaintext is always encoded at Delta^1, so this
    # throws rather than returning a plausible wrong number if the ciphertext is not.
    summed = engine.add(ct, np.ones(engine.slots))
    moved = np.real(np.asarray(engine.decrypt(summed)))[:3] - got
    assert np.max(np.abs(moved - 1.0)) < 0.05, f"adding a plaintext 1 moved values by {moved}"


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
@pytest.mark.xfail(reason="EvalBootstrap on GPU returns scale degree 2 under FIXEDMANUAL; Engine.bootstrap rescales it away (BUG-bootstrap-noise-level-2-20260915.md)")
def test_eval_bootstrap_itself_returns_a_canonical_ciphertext(device):
    """The contract at the C++ boundary, not at the wrapper that works around it.

    `Engine.bootstrap` rescales until the result is canonical, so the test above passes whether or
    not `EvalBootstrap` meets its contract - it measures the workaround. This one calls the binding
    directly, so the underlying behaviour stays pinned and the workaround can be removed the day it
    is fixed rather than living on unexamined.

    It is expected to fail today: the device returns scale degree 2 (see
    `bugs/BUG-bootstrap-noise-level-2-20260915.md`). `approxModReduction` already rescales once
    under FIXEDMANUAL for exactly this purpose, so the intent is not in question - the level it
    costs is, because it is the difference between depth 37 fitting the layer and not.
    """
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
    ct = engine.encrypt(_values(engine.slots))
    assert engine.noise_level(ct) == 1, "a fresh encryption should be canonical"

    raw = engine.cc.EvalBootstrap(ct)
    assert engine.noise_level(raw) == 1, (
        f"EvalBootstrap returned scale degree {engine.noise_level(raw)}, not 1. Engine.bootstrap "
        f"rescales that away at the cost of {engine.noise_level(raw) - 1} level(s), which the level "
        f"plan has to carry: see thorfhe.budget.MEASURED_BOOTSTRAP.")


def test_noise_level_tracks_a_multiplication(engine):
    """The accessor means what it says, at parameters small enough to run in the normal suite."""
    from conftest import rand

    x = engine.encrypt(rand(engine, 31, scale=0.4))
    assert engine.noise_level(x) == 1
    product = engine.multiply(x, 0.5)
    assert engine.noise_level(product) == 2, "a float scalar multiply costs a scale degree"
    assert engine.noise_level(engine.rescale(product)) == 1, "rescale returns it"
