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

    return dict(log_n=16, depth=37, scaling_bits=50, first_mod_bits=55, dnum=4,
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
@pytest.mark.xfail(reason="bootstrap precision is only ~10.9 bits (max abs error 0.017) at dnum=4, "
                          "should be 20-25 bits. This low precision overwhelms he_inv's 2e-4 denominator. "
                          "Root cause: NoiseLevel=3 before approxModReduction's final rescale, indicating "
                          "a missing rescale inside the Chebyshev/double-angle arithmetic.")
def test_bootstrap_returns_a_canonical_ciphertext(device):
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
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


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_bootstrapping_one_ciphertext_twice_gives_the_same_answer(device):
    """Evaluation is deterministic, so the only randomness in a bootstrap is its input's.

    This test used to encrypt twice and compare, which cannot distinguish what it was written to
    distinguish: two encryptions of the same message carry different noise, so a difference between
    their bootstraps says nothing about whether the bootstrap added anything. And a bootstrap has
    nothing to add - key switching uses fixed keys, and NTT, rescale and rotation are deterministic.

    So the question the measured 0.015 raises is not "where does the randomness come from" but "by
    how much is the input's own noise being amplified": a fresh encryption carries about 7e-13 in
    message units, and 0.015 is 2^34 times that. This pins the half of it that is checkable here -
    the same ciphertext in has to give the same answer out - and
    `test_bootstrap_noise_scales_with_the_input_noise` measures the gain.
    """
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
    ct = engine.encrypt(np.zeros(engine.slots, dtype=complex))   # once: the SAME ciphertext twice
    first = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
    second = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))

    spread = float(np.abs(first - second).max())
    assert spread < 1e-9, (
        f"bootstrapping one ciphertext twice gave results differing by {spread:.3g}, against "
        f"{np.abs(first).max():.3g} each. Evaluation has no source of randomness - fixed keys, "
        f"deterministic NTT, rescale and rotation - so this should be bit-identical. If it is not, "
        f"something is reading uninitialised memory.")


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_bootstrap_noise_scales_with_the_input_noise(device):
    """Does the bootstrap amplify what it is given, or add a floor of its own?

    Two inputs holding the same value with different noise: a fresh encryption, and one that has
    been through a thousand rotations (each measured at about 4e-10, so roughly 1e-8 to 1e-6 by the
    time they accumulate). If the output noise tracks the input's, the bootstrap is an amplifier and
    the gain is the thing to explain. If both come back at the same 0.015, it has a floor of its own
    and the input does not matter.

    Reported rather than asserted at a threshold, because which of the two it is has not been
    established yet and a guessed bound would only encode the guess.
    """
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
    zero = np.zeros(engine.slots, dtype=complex)

    fresh = engine.encrypt(zero)
    noisy = engine.encrypt(zero)
    for _ in range(1000):
        noisy = engine.rotate(noisy, 1)

    before = {"fresh": float(np.max(np.abs(np.real(np.asarray(engine.decrypt(fresh)))))),
              "rotated": float(np.max(np.abs(np.real(np.asarray(engine.decrypt(noisy))))))}
    after = {}
    for name, ct in (("fresh", fresh), ("rotated", noisy)):
        out = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
        after[name] = float(np.max(np.abs(out)))

    print(f"\n  input noise   fresh {before['fresh']:.3g}   after 1000 rotations "
          f"{before['rotated']:.3g}  (ratio {before['rotated'] / max(before['fresh'], 1e-300):.3g})")
    print(f"  after bootstrap  fresh {after['fresh']:.3g}   rotated {after['rotated']:.3g}  "
          f"(ratio {after['rotated'] / max(after['fresh'], 1e-300):.3g})")
    print(f"  gain             fresh {after['fresh'] / max(before['fresh'], 1e-300):.3g}   "
          f"rotated {after['rotated'] / max(before['rotated'], 1e-300):.3g}")

    assert after["fresh"] > 0, "a bootstrap of zero should not be exactly zero"


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_which_bootstrap_stage_introduces_the_error(device):
    """Four steps, one of them loses the accuracy. This says which.

    Refreshing a ciphertext of zeros comes back with an error of about 0.015 per slot that is
    deterministic, independent of the message, independent of the input's own noise, and independent
    of q0/Delta - every hypothesis that would explain it has been measured and ruled out, because a
    bootstrap is four steps and from outside it is one.

    Zeros in means every stage should read roughly zero out. Printed rather than asserted: the point
    is to find the step where the number stops being small, and a threshold guessed before knowing
    which step that is would only encode the guess. Once it is known, this becomes an assertion on
    that stage.
    """
    import pyfideslib as pf

    engine = pf.Engine(device, **bench_params())
    ct = engine.encrypt(np.zeros(engine.slots, dtype=complex))

    print(f"\n  input          {np.max(np.abs(np.real(np.asarray(engine.decrypt(ct))))):.3g}")
    for stage, name in ((1, "ModRaise + scale"), (2, "CoeffsToSlots"),
                        (3, "modular reduction"), (4, "SlotsToCoeffs")):
        out = np.real(np.asarray(engine.decrypt(engine.bootstrap_stage(ct, stage))))
        print(f"  after stage {stage} ({name:<18}) max {np.max(np.abs(out)):.3g}  "
              f"p50 {np.quantile(np.abs(out), 0.5):.3g}")
    whole = np.real(np.asarray(engine.decrypt(engine.bootstrap(ct))))
    print(f"  whole bootstrap                    max {np.max(np.abs(whole)):.3g}  "
          f"p50 {np.quantile(np.abs(whole), 0.5):.3g}")


def test_a_single_rotation_is_accurate(engine):
    """CtS and StC are rotations and plaintext multiplies, so a rotation's own error bounds theirs.

    Kept because the bootstrap's 0.015 has to come from somewhere and this rules out the cheapest
    explanation: if one key switch cost anything like that, nothing in the pipeline would work -
    a layer performs several thousand of them.
    """
    from conftest import rand

    x = rand(engine, 77, scale=0.4)
    rotated = np.real(np.asarray(engine.decrypt(engine.rotate(engine.encrypt(x), 1))))
    error = np.max(np.abs(rotated[: len(x)] - np.roll(np.real(x), -1)[: len(x)]))
    assert error < 1e-6, f"one rotation moved the value by {error:.3g}; a key switch should not"


def test_noise_level_tracks_a_multiplication(engine):
    """The accessor means what it says, at parameters small enough to run in the normal suite."""
    from conftest import rand

    x = engine.encrypt(rand(engine, 31, scale=0.4))
    assert engine.noise_level(x) == 1
    product = engine.multiply(x, 0.5)
    assert engine.noise_level(product) == 2, "a float scalar multiply costs a scale degree"
    assert engine.noise_level(engine.rescale(product)) == 1, "rescale returns it"
