"""Check that bootstrap resets NoiseLevel to 1 on GPU.

If bootstrap leaves NoiseLevel at N > 1, then addScalar(1.0) encodes the
constant at scale Delta but the ciphertext is at scale Delta^N, so the
addition is wrong by Delta^(N-1).  For N=1 the decoded value increases by
exactly 1.0; for N>1 it increases by 1/Delta^(N-1) which is essentially 0.

This test encrypts a known value, bootstraps it, adds 1.0, and checks
whether the result increased by ~1.0.
"""
import os
import numpy as np
import pytest

from thorfhe.numeric import NumericMixin
from thorfhe.stages import Stages


class Numeric(NumericMixin, Stages):
    pass


BENCH = dict(log_n=16, depth=37, scaling_bits=50, first_mod_bits=55, dnum=4)


@pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                    reason="set PYFIDESLIB_BENCH_PARAMS=1 to build an engine at the benchmark's parameters")
def test_bootstrap_resets_noise_level(device):
    import pyfideslib as pf

    engine = pf.Engine(device, **BENCH)
    slots = engine.slots

    x = np.zeros(slots, dtype=complex)
    x[0] = 2.0
    x[1] = -3.0
    x[2] = 0.5

    ct = engine.encrypt(x)
    ct = engine.bootstrap(ct)

    before = np.real(np.asarray(engine.decrypt(ct)))[:3]
    print(f"\nAfter bootstrap: {before}")

    ct_plus_1 = engine.add(ct, 1.0)
    after = np.real(np.asarray(engine.decrypt(ct_plus_1)))[:3]
    print(f"After addScalar(1.0): {after}")

    diff = after - before
    print(f"Difference: {diff}")
    print(f"Expected: [1.0, 1.0, 1.0]")

    assert np.max(np.abs(diff - 1.0)) < 1e-6, (
        f"addScalar(1.0) moved values by {diff}, not 1.0 — "
        f"bootstrap likely left NoiseLevel != 1"
    )
