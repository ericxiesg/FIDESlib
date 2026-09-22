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


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_conjugate_at_slot_zero_its_own_fixed_point(device):
    """Is `conjugate` right at slot 0, which is the one slot the automorphism maps to itself?

    Why this slot and not another. The device's `07d` has exactly one bad slot, it is slot 0, and it
    is slot 0 on every run - three out of three, plus the two before them. Neither of the two things
    that could place it there does:

      - not the bootstrap's error, which is i.i.d.: two runs of the dense-vector measurement put
        their five worst slots at 31049/1278/8038/4625/2160 and 9644/10144/8841/13672/28282, with no
        overlap and no `%2048 == 0` among them;
      - not the denominator, measured on the clear engine at the same flags: slot 0 holds 0.03757
        and ranks 688th of 8448 carried slots, with 8.1% of them smaller. The actual minimum, 0.02643,
        sits at `%2048 == 436`.

      - and not the input's shape either: bootstrapping a vector with 8448 carried slots and 24320
        exact zeros gives sigma 0.0155 against the dense vector's 0.0156, with the same p50 to three
        figures.

    Structureless noise and a structureless input cannot produce a failure that lands on the same
    slot every time, so what is left is an operation. This test covers conjugate because it sits
    directly above both symptoms - `_restore_magnitude` calls `add(ct, conjugate(ct))` once per
    Goldschmidt iteration, and stage 07's unpack builds its imaginary halves with
    `multiply_1j(subtract(conjugated, merged))`, which is where `07a`'s three worst carried slots
    were: slot 0 of ciphertexts 5, 6 and 7.

    It is worth being clear that this is association, not a mechanism. I first argued slot 0 must be
    conjugation's fixed point, and that is wrong: conjugation acts on slots in place, sending v_j to
    conj(v_j), so as a permutation it fixes every slot and singles out none. (In the *coefficient*
    domain the automorphism does fix index 0, which is a real special case in a kernel - but an error
    there lands on every slot equally, not on one.) So this checks an operator that is upstream of
    both symptoms; it does not predict that it will fail.

    The clear engine cannot answer it either way, because its conjugate is numpy's.

    Checks the identity on a real message (conj is identity), the doubling `_restore_magnitude`
    actually performs, and a complex message (conj flips the imaginary part), each at the levels the
    iteration reaches - and reports slot 0 against every other slot rather than a single maximum,
    because one wrong slot in 32768 does not move a max-norm that a healthy tail already sets.
    """
    import pyfideslib as pf

    from test_bootstrap_noise_level import bench_params

    engine = pf.Engine(device, **bench_params())
    rng = np.random.default_rng(11)
    # The magnitudes `b` reaches in the iteration, not a unit-scale vector: the question is whether
    # slot 0 is wrong, and a relative error shows only against the value it is relative to.
    real = rng.uniform(0.0264, 0.3036, engine.slots) * 3.0
    complexed = real + 1j * rng.uniform(-0.3, 0.3, engine.slots)

    failures = []
    for name, message, expected in (
            ("conjugate(real) == real", real, real),
            ("real + conj(real) == 2*real", real, 2 * real),
            ("conjugate(complex)", complexed, np.conj(complexed)),
    ):
        for drop in (0, 8, 16):
            ct = _at(engine, engine.encrypt(message), drop)
            out = engine.conjugate(ct)
            if name.startswith("real +"):
                out = engine.add(ct, out)
            got = np.asarray(engine.decrypt(out))[:message.size]
            error = np.abs(got - expected)
            others = np.delete(error, 0)
            print(f"\n{name}, {drop} levels down (level {engine.level(ct)}):"
                  f"\n  slot 0     {error[0]:.6g}"
                  f"\n  every other slot: max {others.max():.6g}  p50 {np.median(others):.6g}"
                  f"\n  slot 0 / p50 of the rest: {error[0] / max(np.median(others), 1e-30):.1f}x")
            # A slot that is merely noisy sits inside the spread of the others. One that is wrong
            # does not, and 100x the median of 32767 healthy slots is not a tail.
            if error[0] > 100 * max(np.median(others), 1e-12):
                failures.append(f"{name} at drop {drop}: slot 0 is {error[0]:.4g} against a median "
                                f"{np.median(others):.4g} over the other {others.size} slots")

    assert not failures, "conjugate is wrong at its fixed point:\n  " + "\n  ".join(failures)
