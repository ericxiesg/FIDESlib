"""Sweep the bootstrap correction factor, which we have never set and OpenFHE picks badly for us.

`EvalBootstrapSetup(..., correctionFactor=0)` makes OpenFHE derive the factor from a curve fitted
per scaling technique. The fit has two branches and they were calibrated separately: the FLEXIBLE*
branch is the one `ckks-bootstrapping-precision.cpp` was measured on, and FIXEDMANUAL - which is what
we run - takes the other. At depth 37 ours returns

    round(-0.1516 * 47 + 14.284) = 7, clamped to [6, 13] -> 7

while thor-openfhe's own `full_bootstrap_probe.cpp` passes **12** explicitly.

The factor is the 2^k amplification applied either side of ModRaise, so it is a first-order term in
bootstrap accuracy - and our measured floor is a Gaussian sigma = 0.0156 = 32 * 2^-11, i.e. about
10.9 bits where 20-25 is expected. A factor five below what the reference implementation uses is the
cheapest explanation still standing, and this is the cheapest test of it: one engine per value, one
bootstrap of a known vector, report the error.

Nothing here asserts a threshold. The point is the shape of the curve across k - if accuracy is flat
in k, the correction factor is not the problem and this candidate is closed; if it improves toward
12, we have both a cause and a fix.
"""
import os

import numpy as np
import pytest

from thorfhe.clear import effective_bootstrap_precision_bits

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                           reason="set PYFIDESLIB_BENCH_PARAMS=1")

# 0 is OpenFHE's own choice (7 here) and is the control; 12 is thor-openfhe's.
FACTORS = (0, 7, 9, 10, 11, 12, 13)


def _message(slots, seed=23):
    return np.random.default_rng(seed).uniform(0.0814, 0.9812, slots)


@BENCH
@pytest.mark.parametrize("factor", FACTORS)
def test_bootstrap_error_versus_correction_factor(device, factor):
    """One engine per factor. Prints max and RMS error of a bootstrap against the input."""
    from test_bootstrap_noise_level import bench_params

    params = dict(bench_params())
    params["bootstrap_correction_factor"] = factor
    engine = pf.Engine(device, **params)

    x = _message(engine.slots)
    ct = engine.encrypt(x)
    out = engine.bootstrap(ct)
    got = np.real(np.asarray(engine.decrypt(out)))[:x.size]

    err = np.abs(got - x)
    rms = float(np.sqrt(np.mean(err ** 2)))
    # Raw bits and the normalised figure both: a bootstrap reproduces q0/Delta to some precision,
    # so the raw number moves with the moduli even when the implementation does not. See
    # thorfhe.clear.effective_bootstrap_precision_bits.
    raw = -np.log2(max(rms, 1e-300))
    eff = effective_bootstrap_precision_bits(max(rms, 1e-300), params["scaling_bits"],
                                             params["first_mod_bits"])
    print(f"\n[correction factor {factor:2d}] max {np.max(err):.6g}  rms {rms:.6g}  "
          f"raw {raw:.1f} bits  effective {eff:.1f} bits "
          f"(CPU reaches 38.1 on these params; ClearEngine assumes 22)")

    # Only a sanity bound: a bootstrap that returns noise this large is not bootstrapping at all,
    # and the number above is then not a precision measurement worth comparing across the sweep.
    assert np.max(err) < 1.0, "bootstrap output is unrelated to its input at this correction factor"
