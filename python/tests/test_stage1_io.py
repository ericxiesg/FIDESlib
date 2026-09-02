"""Stage 1: context creation, keys, encode/encrypt/decrypt round trip (numpy bridge)."""
import numpy as np

from conftest import rand


def test_roundtrip_real(engine):
    x = rand(engine, 1)
    ct = engine.encrypt(x)
    y = engine.decrypt_real(ct)
    assert np.max(np.abs(y - x)) < 1e-6


def test_roundtrip_complex(engine):
    x = rand(engine, 2, complex_=True)
    ct = engine.encrypt(x)
    y = engine.decrypt(ct)
    assert np.max(np.abs(y - x)) < 1e-6


def test_levels(engine):
    ct = engine.encrypt(rand(engine, 3))
    assert engine.level(ct) == engine.depth
    ct2 = engine.encrypt(rand(engine, 3), level=4)
    assert engine.level(ct2) == engine.depth - 4
