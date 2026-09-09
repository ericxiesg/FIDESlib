"""Stage 4: bootstrap on complex full-slot input + output level contract (THOR: refresh to a fixed level)."""
import os

import numpy as np
import pytest

pf = pytest.importorskip("pyfideslib", reason="the pyfideslib extension is not built")

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

    # How many levels a bootstrap leaves is not a constant: it moves with the level budget, the
    # secret key distribution, and the level its diagonals were encoded at. Ask what it gives, then
    # test the contract against that rather than against a number baked in here.
    out = e.bootstrap(ct)
    native = e.level(out)
    assert native > 0
    err = np.max(np.abs(e.decrypt(out) - z))
    print("bootstrap max err", err, "level", native, "keys grown", e.cc.GetGrownKeyCount())
    assert err < 1e-2
    # 0 == the bootstrap level plan was exact; a non-zero count names the offending keys on stderr.
    assert e.cc.GetGrownKeyCount() == 0

    # keep_levels below what it produced is honoured exactly ...
    assert e.level(e.bootstrap(ct, keep_levels=native - 1)) == native - 1
    # ... and above it is refused, rather than quietly handing back fewer levels than asked for
    with pytest.raises(ValueError, match="keep_levels"):
        e.bootstrap(ct, keep_levels=native + 1)
