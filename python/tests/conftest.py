import os

import pytest

try:
    import pyfideslib as pf
except ImportError as exc:  # the pybind11 extension is not built
    pf = None
    _pyfideslib_error = exc

DEVICES = [d.strip() for d in os.environ.get("PYFIDESLIB_DEVICES", "cpu,cuda:0").split(",") if d.strip()]

# Small parameters for the functional stages (fast on CPU). Stage 4 (bootstrap) uses its own engine.
# `secret_key_dist` is explicit because the default is not what the benchmark runs. `Engine` defaults
# to UNIFORM_TERNARY, which is the conservative choice and the one OpenFHE makes; `thorfhe.bench`
# passes SPARSE_TERNARY, which is THOR's. They are not interchangeable for anything that bootstraps:
# FIDESlib picks the Chebyshev coefficient set and the number of double-angle iterations from this
# (3 and depth 10 for sparse, 6 and depth 13 for uniform), so a suite left on the default tests a
# different approximation from the one that runs - and it is bootstrap accuracy that is in question.
SMALL = dict(log_n=13, depth=12, scaling_bits=50, first_mod_bits=55, dnum=3,
             secret_key_dist=None)   # filled in below, once pyfideslib is known to be importable

if pf is not None:
    SMALL["secret_key_dist"] = pf.SPARSE_TERNARY
else:
    SMALL.pop("secret_key_dist")


@pytest.fixture(scope="session", params=DEVICES)
def device(request):
    """Skips rather than errors when the extension is absent, so the numpy-only suites still run."""
    if pf is None:
        pytest.skip(f"pyfideslib is not built here: {_pyfideslib_error}")
    return request.param


# rotation index -> maximum remaining level the key is ever used at (THOR's create_fixed_rotation_key table).
# A declared level >= depth-1 keeps the key complete; 2 and -3 are deliberately truncated so the tests cover
# both a truncated key used inside its plan and one used outside it.
ROTATIONS = {1: SMALL["depth"], 2: 5, -3: 3, 16: SMALL["depth"]}


@pytest.fixture(scope="session")
def engine(device):
    return pf.Engine(device, rotation_indexes=ROTATIONS, **SMALL)


def rand(engine, seed, scale=1.0, complex_=False):
    import numpy as np

    rng = np.random.default_rng(seed)
    x = rng.uniform(-scale, scale, engine.slots)
    if complex_:
        x = x + 1j * rng.uniform(-scale, scale, engine.slots)
    return x
