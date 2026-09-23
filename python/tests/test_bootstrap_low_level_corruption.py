"""Four experiments on the slot-0 corruption, meant to be run as one batch.

Where this stands. A bootstrapped ciphertext multiplied at a low level corrupts a fixed set of
slots - 0, 9891, 20170 - at about 13000x the median error, and it needs both conditions: fresh at
any level is clean, bootstrapped at a high level is clean, and the CPU backend is clean in all
four. The corrupted positions are identical between a real softmax denominator and
`rng.uniform(seed=23)`, so it is positional rather than data-dependent. Eliminated by measurement
so far: bootstrap error magnitude (sigma = 0.0156, six clean local injections), its slot
correlation, its dependence on input shape, the denominator's own structure at slot 0, the lift
cliff, aliasing between the two `_times` calls, `_restore_magnitude` (inert at iteration 1), any
index-0 branch in multiply/relinearize/rescale (source review), a non-bijective automorph scatter
(enumerated all 65536 slots), and `truncate_keys` (still 12452x with full keys).

The four tests below split what is left. They are independent - run them all, report all four, and
do not stop at the first interesting one.

    PYFIDESLIB_BENCH_PARAMS=1 python3 -m pytest -s -q tests/test_bootstrap_low_level_corruption.py

`-s` is required; these print rather than assert, except where a specific claim is being pinned.
"""
import os

import numpy as np
import pytest

pf = pytest.importorskip("pyfideslib", reason="the pyfideslib extension is not built")

BENCH = pytest.mark.skipif(
    not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")

#: layer 0's epsilon lifted by 3, and the k the first Goldschmidt step uses.
K = 2 / (1 + 2.0 ** -6 * 3)


def _engine(device, **overrides):
    from test_bootstrap_noise_level import bench_params

    params = dict(bench_params())
    params.update(overrides)
    return pf.Engine(device, **params)


def _denominator(engine):
    """The magnitudes `he_inv` hands the iteration, same draw as the reproducing test."""
    return np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)


def _slot0_ratio(engine, product, expected):
    """Slot 0's error against the median of every other slot. The median, not the maximum.

    One wrong slot does not move a max-norm that a healthy tail already sets, which is how the
    conjugate test came back green while slot 0 was 12x the median.
    """
    got = np.real(np.asarray(engine.decrypt(product)))[:expected.size]
    error = np.abs(got - expected)
    others = np.delete(error, 0)
    return error[0] / max(np.median(others), 1e-30), error, got


def _times(engine, left, right):
    return engine.rescale(engine.relinearize(engine.multiply(left, right)))


# --------------------------------------------------------------------------------- experiment 1
@BENCH
def test_1_raw_bootstrap_without_the_rescale_loop(device):
    """Is it the bootstrap, or the rescale loop `pyfideslib` runs after it?

    `Engine.bootstrap` does not just call `EvalBootstrap`. Under FIXEDMANUAL it then rescales in a
    loop until `noise_level` comes back to 1, because the GPU bootstrap returns scale degree 2 -
    the correction added in 72dc818. So every "bootstrapped ciphertext" in every experiment so far
    has actually been *bootstrapped and then rescaled one or more times*, and the two have never
    been separated.

    This runs `cc.EvalBootstrap` directly, with no correction, and reports the scale degree it
    really comes back at alongside the slot-0 ratio.

      - raw is clean and corrected is not -> the rescale loop is implicated, not the bootstrap;
      - both are broken -> the bootstrap's own output is, and the loop is innocent.
    """
    engine = _engine(device)
    b0 = _denominator(engine)

    raw = engine.cc.EvalBootstrap(engine.encrypt(b0))
    degree_raw = engine.noise_level(raw)
    level_raw = engine.level(raw)

    corrected = engine.bootstrap(engine.encrypt(b0))
    degree_fixed = engine.noise_level(corrected)
    level_fixed = engine.level(corrected)

    print(f"\n[1] raw EvalBootstrap:  level {level_raw}  scale degree {degree_raw}"
          f"\n    after the loop:     level {level_fixed}  scale degree {degree_fixed}"
          f"\n    the loop spent {level_raw - level_fixed} level(s)")

    for name, ct in (("raw (no rescale loop)", raw), ("corrected (current path)", corrected)):
        boot = np.real(np.asarray(engine.decrypt(ct)))[:b0.size]
        try:
            product = _times(engine, ct, engine.subtract(2 / K, ct))
        except Exception as error:          # a degree-2 operand may be refused outright
            print(f"    {name}: _times refused it - {type(error).__name__}: {error}")
            continue
        ratio, err, _ = _slot0_ratio(engine, product, boot * (2 / K - boot))
        print(f"    {name}: slot 0 / p50 = {ratio:.1f}x   slot 0 error {err[0]:.6g}"
              f"   worst {', '.join(str(int(i)) for i in np.argsort(-err)[:5])}")


# --------------------------------------------------------------------------------- experiment 2
@BENCH
def test_2_metadata_of_a_bootstrapped_versus_a_dropped_ciphertext(device):
    """Do the two differ in any state we can actually read, at the same nominal level?

    The whole symptom is that a bootstrapped ciphertext behaves differently from a fresh one at the
    same level. `level` and `noise_level` are the two pieces of FIXEDMANUAL state exposed to
    Python. If they disagree, that is the bug and no further search is needed. If they agree, the
    difference is in state Python cannot see - limb layout, allocation width, the live length of
    the `limbptr` / `DIGITlimbptr` tables - and the next step is a C++-side print.
    """
    engine = _engine(device)
    b0 = _denominator(engine)

    boot = engine.bootstrap(engine.encrypt(b0))
    target = engine.level(boot)
    fresh = engine.encrypt(b0)
    dropped = engine.level_down(fresh, by=engine.level(fresh) - target)

    print(f"\n[2] at nominal level {target}:"
          f"\n    bootstrapped: level {engine.level(boot):3d}  noise_level {engine.noise_level(boot)}"
          f"\n    dropped:      level {engine.level(dropped):3d}  noise_level {engine.noise_level(dropped)}")

    same = (engine.level(boot) == engine.level(dropped)
            and engine.noise_level(boot) == engine.noise_level(dropped))
    verdict = ("identical in everything Python can read; look at the C++ limb state" if same
               else "THEY DIFFER - this is the bug")
    print(f"    -> {verdict}")

    # And the two products side by side, so the asymmetry is visible in one place.
    for name, ct in (("bootstrapped", boot), ("dropped", dropped)):
        values = np.real(np.asarray(engine.decrypt(ct)))[:b0.size]
        ratio, err, _ = _slot0_ratio(
            engine, _times(engine, ct, engine.subtract(2 / K, ct)), values * (2 / K - values))
        print(f"    {name}: slot 0 / p50 = {ratio:.1f}x   slot 0 error {err[0]:.6g}")


# --------------------------------------------------------------------------------- experiment 3
@BENCH
def test_3_which_bootstrap_stage_first_deviates_at_slot_zero(device):
    """Bisect the bootstrap itself. `bootstrap_stage` stops after ModRaise / CtS / EvalMod / StC.

    A bootstrap is four steps and from outside it is one, which is why its accuracy has nowhere to
    be pinned. Refreshing a vector and stopping at each stage says which step first treats slot 0
    differently from its neighbours - and whether any does at all, which would move the fault out
    of the bootstrap entirely and into what happens to its output afterwards.

    The intermediates are not usable ciphertexts - level, scale and slot layout are mid-flight - so
    this reports slot 0 against the spread of the other slots rather than against an expected
    value, which needs no reference.
    """
    engine = _engine(device)
    b0 = _denominator(engine)

    print("\n[3] slot 0 against the other slots, after each bootstrap stage:")
    for stage, what in enumerate(("ModRaise", "CoeffsToSlots", "EvalMod", "SlotsToCoeffs"), start=1):
        try:
            out = engine.bootstrap_stage(engine.encrypt(b0), stage)
        except Exception as error:
            print(f"    stage {stage} ({what}): raised {type(error).__name__}: {error}")
            continue
        values = np.abs(np.real(np.asarray(engine.decrypt(out)))[:b0.size])
        others = np.delete(values, 0)
        print(f"    stage {stage} ({what:13s}): slot 0 {values[0]:12.6g}   "
              f"others p50 {np.median(others):12.6g}  max {others.max():12.6g}   "
              f"ratio {values[0] / max(np.median(others), 1e-30):8.1f}x")


# --------------------------------------------------------------------------------- experiment 4
@BENCH
def test_4_at_which_level_does_the_corruption_start(device):
    """Sweep the level. The answer is a number of limbs, and that number is the clue.

    Clean at level 36, broken at level 19/20, and nothing in between has been measured. If the
    transition is sharp and lands on a particular limb count, that count is worth comparing against
    the structural boundaries: `alpha = ceil(L/dnum)` limbs per key-switch digit, so the number of
    live digits `beta = ceil(cur_limbs/alpha)` steps down at multiples of alpha, and the DIGIT
    metadata prefix shortens with it. Key truncation is built on exactly that prefix property
    (LimbPartition.cu:289-303), and our key-switch kernel dereferences three pointer levels deep
    (ElemenwiseBatchKernels.cu:271-273).

    If the corruption switches on at a beta boundary, that is a strong pointer at the tables. If it
    degrades smoothly instead, it is numerical and the tables are innocent.
    """
    engine = _engine(device)
    b0 = _denominator(engine)

    boot = engine.bootstrap(engine.encrypt(b0))
    top = engine.level(boot)
    print(f"\n[4] bootstrapped ciphertext starts at level {top}; dropping and multiplying:")
    print(f"    {'level':>6}  {'slot 0 / p50':>14}  {'slot 0 error':>14}   worst three")

    for drop in range(0, top - 1, max(1, (top - 1) // 12)):
        ct = engine.level_down(boot, by=drop) if drop else boot
        values = np.real(np.asarray(engine.decrypt(ct)))[:b0.size]
        try:
            product = _times(engine, ct, engine.subtract(2 / K, ct))
        except Exception as error:
            print(f"    {engine.level(ct):6d}  raised {type(error).__name__}: {error}")
            continue
        ratio, err, _ = _slot0_ratio(engine, product, values * (2 / K - values))
        worst = ", ".join(str(int(i)) for i in np.argsort(-err)[:3])
        print(f"    {engine.level(ct):6d}  {ratio:14.1f}  {err[0]:14.6g}   {worst}")
