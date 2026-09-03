import os

import pytest

import pyfideslib as pf

DEVICES = [d.strip() for d in os.environ.get("PYFIDESLIB_DEVICES", "cpu,cuda:0").split(",") if d.strip()]

# Small parameters for the functional stages (fast on CPU). Stage 4 (bootstrap) uses its own engine.
SMALL = dict(log_n=13, depth=12, scaling_bits=50, first_mod_bits=55, dnum=3)


@pytest.fixture(scope="session", params=DEVICES)
def device(request):
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
