"""Stage stability measured from *one* ciphertext, which is the only way it means anything.

`test_stage_stability.py` and `test_modraise_components.py` both call `engine.encrypt(b0)` inside
the run loop, so every run starts from a different ciphertext. That makes the ModRaise result differ
between runs by construction, and it is not a device bug.

CKKS encryption is randomised: `c1 = a` is uniform and `c0 = -a*s + Delta*m + e`. At level 0 the pair
satisfies `c0 + c1*s = Delta*m + e (mod q0)`, so over the integers

    c0 + c1*s = Delta*m + e + q0 * I

for a polynomial `I` determined by the actual coefficients - that is, by the random `a`. ModRaise does
not compute anything new; it reinterprets the same pair modulo a much larger Q, and what that decrypts
to is `Delta*m + e + q0*I`, with the `q0*I` term dominating. A fresh `a` per run gives a fresh `I` per
run. Stripping `q0*I` back off is exactly what EvalMod is for, and why a bootstrap has an EvalMod at
all.

So "100% of slots unstable after stage 1" is what a *correct* ModRaise must produce when the input is
re-encrypted, and a stage-1 output that was stable across re-encryptions would be the thing worth
investigating.

The question the reports meant to ask - does the device compute the same stage twice the same way -
needs the ciphertext held fixed. `EvalBootstrap` does not mutate its argument: it copy-constructs
(`api/CryptoContext.cpp:1785`) and `CiphertextImpl`'s copy constructor deep-copies the device
ciphertext through `CopyDeviceCiphertext` (`api/Ciphertext.cpp:34`). One `encrypt`, four
`bootstrap_stage` calls on it, is therefore both valid and cheap.
"""
import os

import numpy as np
import pytest

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                           reason="set PYFIDESLIB_BENCH_PARAMS=1")

N_RUNS = 4
STAGES = ((1, "ModRaise"), (2, "CoeffsToSlots"), (3, "EvalMod"), (4, "SlotsToCoeffs"))


def _engine(device, **ov):
    from test_bootstrap_noise_level import bench_params
    p = dict(bench_params())
    p.update(ov)
    return pf.Engine(device, **p)


def _message(engine):
    return np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)


def _decrypt_real(engine, ct, n):
    return np.real(np.asarray(engine.decrypt(ct)))[:n]


@BENCH
def test_each_stage_is_deterministic_from_one_ciphertext(device):
    """The measurement the earlier reports intended: same input, same stage, four times.

    Threshold is exact equality rather than a tolerance. Every stage is deterministic arithmetic on
    fixed inputs - INTT, grow, broadcast, NTT, rotations, plaintext multiplies - so on the same
    ciphertext the same stage has to return the same bits. A tolerance would hide precisely the
    small races the reports were hunting.
    """
    engine = _engine(device)
    b0 = _message(engine)
    ct = engine.encrypt(b0)          # once, on purpose

    failures = []
    for stage, name in STAGES:
        runs = np.array([_decrypt_real(engine, engine.bootstrap_stage(ct, stage), b0.size)
                         for _ in range(N_RUNS)])
        spread = np.max(np.abs(runs - runs[0]), axis=0)
        differing = int(np.sum(spread > 0.0))
        print(f"\n[stage {stage} {name:13s}] slots differing from run 0: {differing:5d}/{runs.shape[1]}"
              f"  max deviation {np.max(spread):.4g}  slot0={runs[0][0]:.8g}")
        if differing:
            worst = int(np.argmax(spread))
            print(f"    worst slot {worst}: " + " ".join(f"{runs[r][worst]:.10g}" for r in range(N_RUNS)))
            failures.append(f"stage {stage} ({name}): {differing} slots, max dev {np.max(spread):.4g}")

    assert not failures, "the device did not reproduce its own result:\n  " + "\n  ".join(failures)


@BENCH
def test_reencrypting_changes_modraise_and_that_is_correct(device):
    """The control for the test above: re-encrypt per run and stage 1 *must* move.

    This reproduces the earlier reports' number and shows what it measures. It is written as an
    assertion in the opposite direction on purpose - if re-encrypting stopped changing the ModRaise
    output, `q0*I` would not be varying with `a`, and something would be wrong with either the
    encryption randomness or ModRaise.
    """
    engine = _engine(device)
    b0 = _message(engine)

    runs = np.array([_decrypt_real(engine, engine.bootstrap_stage(engine.encrypt(b0), 1), b0.size)
                     for _ in range(N_RUNS)])
    spread = np.std(runs, axis=0)
    moving = int(np.sum(spread > 1e-6))
    print(f"\n[stage 1, re-encrypted each run] slots moving(>1e-6): {moving}/{runs.shape[1]}"
          f"  median spread {np.median(spread):.4g}")
    print("    this is q0*I following the random `a`, not a device defect")

    assert moving > runs.shape[1] // 2, (
        "re-encrypting barely changed the ModRaise output; q0*I should follow the random `a`")


@BENCH
def test_encrypt_decrypt_roundtrip_is_stable_but_the_ciphertexts_are_not(device):
    """Why the earlier reports read encrypt as deterministic: they measured the round trip.

    `encrypt` then `decrypt` returns the message to within the encryption noise, far below the 1e-6
    threshold used, so the round trip looks stable. The ciphertexts underneath are different every
    time, and that is what ModRaise then exposes.
    """
    engine = _engine(device)
    b0 = _message(engine)

    runs = np.array([_decrypt_real(engine, engine.encrypt(b0), b0.size) for _ in range(N_RUNS)])
    spread = np.std(runs, axis=0)
    print(f"\n[encrypt->decrypt] max spread {np.max(spread):.4g} (threshold used by the earlier "
          f"reports: 1e-6)")

    # Stable as a round trip...
    assert np.max(spread) < 1e-6
    # ...and yet not bit-identical, which is the whole point: the noise is there, just small.
    assert np.max(spread) > 0.0, (
        "every encryption returned the identical value; if this ever fires, the encryption "
        "randomness is not being drawn and the ModRaise result above would be meaningless")
