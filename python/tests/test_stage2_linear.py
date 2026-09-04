"""Stage 2: add/sub/scalar, plaintext mult + rescale, rotate (level-truncated keys), conjugate, mult-by-i,
level_down, integer scalar (level-free)."""
import numpy as np
import pytest

pf = pytest.importorskip("pyfideslib", reason="the pyfideslib extension is not built")
from conftest import SMALL, rand


def test_add_sub_scalar(engine):
    x, y = rand(engine, 10), rand(engine, 11)
    cx, cy = engine.encrypt(x), engine.encrypt(y)
    assert np.max(np.abs(engine.decrypt_real(engine.add(cx, cy)) - (x + y))) < 1e-6
    assert np.max(np.abs(engine.decrypt_real(engine.subtract(cx, cy)) - (x - y))) < 1e-6
    assert np.max(np.abs(engine.decrypt_real(engine.add(cx, 0.5)) - (x + 0.5))) < 1e-6
    assert np.max(np.abs(engine.decrypt_real(engine.subtract(2.0, cx)) - (2.0 - x))) < 1e-6


def test_mult_plaintext_and_rescale(engine):
    x, w = rand(engine, 12), rand(engine, 13)
    cx = engine.encrypt(x)
    prod = engine.multiply(cx, w)  # ndarray -> encoded at the ciphertext level, FIXEDMANUAL: no auto rescale
    prod = engine.rescale(prod)
    assert engine.level(prod) == engine.depth - 1
    assert np.max(np.abs(engine.decrypt_real(prod) - x * w)) < 1e-5


def test_mult_float_scalar_consumes_level(engine):
    cx = engine.encrypt(rand(engine, 14))
    y = engine.rescale(engine.multiply(cx, 0.25))
    assert engine.level(y) == engine.depth - 1


def test_mult_int_is_level_free(engine):
    x = rand(engine, 15, scale=0.1)
    cx = engine.encrypt(x)
    y = engine.multiply(cx, 7)
    assert engine.level(y) == engine.depth
    assert np.max(np.abs(engine.decrypt_real(y) - 7 * x)) < 1e-5


def test_rotate(engine):
    # indexes 1 and 16 are declared for the top level, so their keys are complete
    x = rand(engine, 16)
    cx = engine.encrypt(x)
    for d in (1, 16):
        assert np.max(np.abs(engine.decrypt_real(engine.rotate(cx, d)) - np.roll(x, -d))) < 1e-6


def test_rotate_truncated_key_inside_plan(engine):
    # keys for 2 and -3 were declared at max remaining levels 5 and 3: use them there and below
    for d, declared in ((2, 5), (-3, 3)):
        x = rand(engine, 17 + d)
        for level in (declared, declared - 2):
            cx = engine.encrypt(x, level=engine.depth - level)
            assert engine.level(cx) == level
            assert np.max(np.abs(engine.decrypt_real(engine.rotate(cx, d)) - np.roll(x, -d))) < 1e-6
    assert engine.cc.GetGrownKeyCount() == 0


def test_rotate_truncated_key_above_plan_raises(engine):
    # Using a key above its declared level means the level plan is wrong: it must be reported, not papered over.
    # OpenFHE has no truncated keys, so this is a GPU-only contract.
    if not engine.on_gpu:
        pytest.skip("level-truncated keys are a GPU-backend feature")
    cx = engine.encrypt(rand(engine, 21))  # level 12 > the 5 declared for index 2
    with pytest.raises(RuntimeError, match="truncated to level"):
        engine.rotate(cx, 2)
    assert engine.cc.GetGrownKeyCount() == 0


def test_rotate_truncated_key_grows_when_allowed(device):
    # Opt-in fallback: the key is reloaded from the OpenFHE context and rebuilt at the higher level.
    e = pf.Engine(device, rotation_indexes={2: 5}, allow_key_grow=True, **SMALL)
    x = rand(e, 22)
    cx = e.encrypt(x)
    assert e.level(cx) == e.depth
    assert np.max(np.abs(e.decrypt_real(e.rotate(cx, 2)) - np.roll(x, -2))) < 1e-6
    assert e.cc.GetGrownKeyCount() == (1 if e.on_gpu else 0)
    # once grown the key is complete, so a second rotation neither reloads nor loses precision
    assert np.max(np.abs(e.decrypt_real(e.rotate(cx, 2)) - np.roll(x, -2))) < 1e-6
    assert e.cc.GetGrownKeyCount() == (1 if e.on_gpu else 0)


def test_conjugate_and_mult_by_i(engine):
    z = rand(engine, 18, complex_=True)
    cz = engine.encrypt(z)
    assert np.max(np.abs(engine.decrypt(engine.conjugate(cz)) - np.conj(z))) < 1e-6
    iz = engine.multiply_1j(cz)
    assert engine.level(iz) == engine.depth
    assert np.max(np.abs(engine.decrypt(iz) - 1j * z)) < 1e-6
    # THOR split: real = (z + conj)/1, imag = i*(conj - z)  (both doubled)
    re2 = engine.add(cz, engine.conjugate(cz))
    im2 = engine.multiply_1j(engine.subtract(engine.conjugate(cz), cz))
    assert np.max(np.abs(engine.decrypt_real(re2) - 2 * z.real)) < 1e-6
    assert np.max(np.abs(engine.decrypt_real(im2) - 2 * z.imag)) < 1e-6


def test_level_down(engine):
    x = rand(engine, 19)
    cx = engine.encrypt(x)
    y = engine.level_down(cx, 3)
    assert engine.level(y) == engine.depth - 3
    assert np.max(np.abs(engine.decrypt_real(y) - x)) < 1e-6
