"""Stage 4: bootstrap on complex full-slot input + output level contract (THOR: refresh to a fixed level)."""
import os

import numpy as np
import pytest

import pyfideslib as pf

DEVICES = [d.strip() for d in os.environ.get("PYFIDESLIB_DEVICES", "cpu,cuda:0").split(",") if d.strip()]
BOOT = dict(log_n=int(os.environ.get("PYFIDESLIB_BOOT_LOGN", "13")), depth=int(os.environ.get("PYFIDESLIB_BOOT_DEPTH", "25")),
            scaling_bits=50, first_mod_bits=55, dnum=3)


@pytest.fixture(scope="module", params=DEVICES)
def boot_engine(request):
    return pf.Engine(request.param, bootstrap_level_budget=(3, 3), secret_key_dist=pf.SPARSE_TERNARY, **BOOT)


def test_bootstrap_complex_and_keep_levels(boot_engine):
    e = boot_engine
    rng = np.random.default_rng(7)
    z = rng.uniform(-1, 1, e.slots) + 1j * rng.uniform(-1, 1, e.slots)
    ct = e.encrypt(z, level=e.depth - 1)  # one level left
    out = e.bootstrap(ct, keep_levels=10)
    assert e.level(out) == 10
    err = np.max(np.abs(e.decrypt(out) - z))
    print("bootstrap max err", err, "keys grown", e.cc.GetGrownKeyCount())
    assert err < 1e-2
    assert e.cc.GetGrownKeyCount() == 0
