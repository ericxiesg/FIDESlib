"""Stage 5: light plaintexts — THOR's compact weight storage.

The contract is that a light plaintext expanded at level L is *the same plaintext* OpenFHE's encoder
produces at that level, so `multiply(ct, light)` and `multiply(ct, ndarray)` must agree to rounding.
Since the CPU backend rebuilds the RNS towers with OpenFHE and the CUDA backend expands them with its
own kernel + NTT, running this on both devices is what pins the coefficient ordering down.
"""
import numpy as np
import pytest

from conftest import rand


def test_encode_expand_roundtrip(engine):
    x = rand(engine, 30)
    light = engine.encode_to_light_plaintext(x)
    # 0.5 MiB at N = 2^16 regardless of depth; a plain encoding is (depth + 1) times that.
    assert light.nbytes() == engine.slots * 2 * 8
    pt = engine.expand_light_plaintext(light, engine.depth)
    if engine.on_gpu:
        # A device-expanded plaintext has no host copy to decode; test_multiply_matches_dense_encoding
        # is what checks its contents on CUDA.
        assert pt.loaded
    else:
        assert np.max(np.abs(np.real(pt.GetCKKSPackedValue()[: engine.slots]) - x)) < 1e-8


def test_multiply_matches_dense_encoding(engine):
    x, w = rand(engine, 31), rand(engine, 32)
    cx = engine.encrypt(x)
    light = engine.encode_to_light_plaintext(w)

    dense = engine.rescale(engine.multiply(cx, w))
    compact = engine.rescale(engine.multiply(cx, light))

    # Same encoder, same coefficients: the two products must agree far below the CKKS noise floor.
    assert np.max(np.abs(engine.decrypt_real(compact) - engine.decrypt_real(dense))) < 1e-9
    assert np.max(np.abs(engine.decrypt_real(compact) - x * w)) < 1e-5


def test_add_matches_dense_encoding(engine):
    x, w = rand(engine, 33), rand(engine, 34)
    cx = engine.encrypt(x)
    light = engine.encode_to_light_plaintext(w)
    assert np.max(np.abs(engine.decrypt_real(engine.add(cx, light)) - (x + w))) < 1e-6


def test_expands_at_the_ciphertext_level(engine):
    """One light plaintext, several levels: the expansion follows the ciphertext (THOR reuses masks)."""
    x, w = rand(engine, 35), rand(engine, 36)
    light = engine.encode_to_light_plaintext(w)
    for consumed in (0, 3, 7):
        cx = engine.encrypt(x, level=consumed)
        out = engine.rescale(engine.multiply(cx, light))
        assert engine.level(out) == engine.depth - consumed - 1
        assert np.max(np.abs(engine.decrypt_real(out) - x * w)) < 1e-5


def test_complex_message(engine):
    z = rand(engine, 37, complex_=True)
    cz = engine.encrypt(rand(engine, 38, complex_=True))
    light = engine.encode_to_light_plaintext(z)
    assert np.max(np.abs(engine.decrypt(engine.add(cz, light)) - (engine.decrypt(cz) + z))) < 1e-6


def test_file_roundtrip(engine, tmp_path):
    w = rand(engine, 39)
    light = engine.encode_to_light_plaintext(w, level=5)
    path = tmp_path / "weight.flpt"
    engine.write_light_plaintext(light, path)
    # 32-byte header + N int64 coefficients: this is the size THOR's ~220k weight files are budgeted at.
    assert path.stat().st_size == light.nbytes() + 32

    back = engine.read_light_plaintext(path)
    assert back.slots == light.slots
    assert back.level_hint == light.level_hint
    assert back.scale == light.scale
    assert np.array_equal(back.coefficients(), light.coefficients())

    cx = engine.encrypt(rand(engine, 40))
    assert np.max(
        np.abs(engine.decrypt_real(engine.multiply(cx, back)) - engine.decrypt_real(engine.multiply(cx, light)))
    ) < 1e-9


def test_cache_is_bounded(engine):
    engine.clear_light_plaintext_cache()
    engine.cc.light_plaintext_cache_capacity = 2
    try:
        lights = [engine.encode_to_light_plaintext(rand(engine, 41 + i)) for i in range(4)]
        cx = engine.encrypt(rand(engine, 45))
        for light in lights:
            engine.multiply(cx, light)
        assert engine.cc.GetLightPlaintextCacheSize() <= 2
    finally:
        engine.cc.light_plaintext_cache_capacity = 64
        engine.clear_light_plaintext_cache()


def test_message_too_large_for_one_tower_is_rejected(engine):
    # Delta is 2^50 and q0 ~2^55, so a message of magnitude 2^12 scales to coefficients well past q0/2:
    # there is no single-tower representation. Either our cross-tower check or OpenFHE's own encoder
    # overflow check has to reject it -- what must not happen is a silently wrong plaintext.
    big = np.full(engine.slots, float(2**12))
    with pytest.raises(RuntimeError):
        engine.encode_to_light_plaintext(big)
