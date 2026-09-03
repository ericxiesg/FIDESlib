"""Stage 4: bootstrap on complex full-slot input + output level contract (THOR: refresh to a fixed level)."""
import os

import numpy as np
import pytest

import pyfideslib as pf

DEVICES = [d.strip() for d in os.environ.get("PYFIDESLIB_DEVICES", "cpu,cuda:0").split(",") if d.strip()]
BOOT = dict(log_n=int(os.environ.get("PYFIDESLIB_BOOT_LOGN", "13")), depth=int(os.environ.get("PYFIDESLIB_BOOT_DEPTH", "25")),
            scaling_bits=59, first_mod_bits=60, dnum=3)


@pytest.fixture(scope="module", params=DEVICES)
def boot_engine(request):
    # allow_key_grow: this test *measures* GetBootstrapKeyLevelPlan rather than relying on it, so a key used
    # above its plan must be reloaded and counted (asserted 0 below) instead of aborting the bootstrap.
    return pf.Engine(request.param, bootstrap_level_budget=(3, 3), secret_key_dist=pf.SPARSE_TERNARY,
                     allow_key_grow=True, **BOOT)


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
    # 0 == the bootstrap level plan was exact; a non-zero count names the offending keys on stderr.
    assert e.cc.GetGrownKeyCount() == 0
