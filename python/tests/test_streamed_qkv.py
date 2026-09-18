"""Stages 03-05 with the rotated copies streamed instead of held.

`stage_02_make_rotated_copies` returns `pack * len(x)` ciphertexts - 64 at THOR's geometry, near full
level - and stages 03, 04 and 05 all read from that one array, so it stays alive across the three.
It is the largest single item in a layer's working set. Streaming turns "hold 64 copies, form one
`(out, diag)` sum at a time" into "hold one copy and all 24 sums", and the copies are then made once
per projection rather than shared.

Measured here at THOR's geometry, depth 37, one projection: **84 ciphertexts / 2743 MiB held against
50 / 1652**. The 1.07 GiB is the margin `--extra-rotation-keys 6` needs, which `bench budget` puts at
2.7 GiB headroom against the 3 GiB key generation wants for scratch.
"""
import numpy as np
import pytest

from thorfhe import (SMALL, THOR_BERT, ClearEngine, block_diagonal_masks, encode_activations,
                     encode_bias, encode_weight)
from thorfhe.stages import Stages
from thorfhe.workingset import tracking_engine


def build(geometry, depth, engine=None):
    low, high = block_diagonal_masks(geometry)
    engine = ClearEngine(geometry, depth=depth) if engine is None else engine
    return engine, Stages(engine, geometry, masks=low, complement_masks=high)


def sample(geometry, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(geometry.dim, geometry.features)) * 0.1,
            rng.normal(size=(geometry.features, geometry.features)) * 0.05,
            rng.normal(size=(geometry.features,)) * 0.1)


@pytest.mark.parametrize("geometry,depth", [(SMALL, 14), (THOR_BERT, 37)],
                         ids=["small", "bert"])
def test_streaming_the_copies_computes_the_same_projection(geometry, depth):
    """Same arithmetic in a different order, so the result has to be identical, not merely close."""
    x, w, b = sample(geometry, 7)
    weight, bias = encode_weight(geometry, w), encode_bias(geometry, b)

    results, levels = {}, {}
    for stream in (False, True):
        engine, stages = build(geometry, depth)
        stages.stream_qkv = stream
        cts = np.array([engine.encrypt(m) for m in encode_activations(geometry, x)], dtype=object)
        _, complexified = stages.stage_01_complexify_x(cts, layer_index=0)
        source = complexified if stream else stages.stage_02_make_rotated_copies(complexified)
        out = stages.stage_03_query(source, weight, bias)
        results[stream] = [np.asarray(engine.decrypt(ct)) for ct in out]
        levels[stream] = [engine.level(ct) for ct in out]

    worst = max(np.abs(a - b).max() for a, b in zip(results[False], results[True]))
    assert worst < 1e-12, f"streaming moved the projection by {worst:.3g}"
    assert levels[False] == levels[True], (
        f"streaming changed the level schedule: {levels[False]} -> {levels[True]}")


def test_streaming_holds_less():
    """The point of the exercise, priced by level rather than counted.

    A ciphertext at level 31 costs six times one at level 5, so the moment a stage holds the most
    ciphertexts is not the moment it holds the most memory - the count is reported too, but the
    assertion is on bytes.
    """
    x, w, _ = sample(THOR_BERT, 7)
    weight = encode_weight(THOR_BERT, w)

    held = {}
    for stream in (False, True):
        engine, workingset = tracking_engine(THOR_BERT, depth=37)
        _, stages = build(THOR_BERT, 37, engine=engine)
        cts = np.array([engine.encrypt(m) for m in encode_activations(THOR_BERT, x)], dtype=object)
        _, complexified = stages.stage_01_complexify_x(cts, layer_index=0)
        workingset.at("projection")
        if stream:
            stages.pcmm_streamed(weight, stages.iter_rotated_copies(complexified))
        else:
            rotated = stages.stage_02_make_rotated_copies(complexified)
            stages.pcmm(weight, rotated)
            del rotated
        held[stream] = workingset.per_label["projection"]

    saved = (held[False].nbytes - held[True].nbytes) / 2 ** 20
    assert saved > 800, (
        f"streaming saved {saved:.0f} MiB, which is not enough to be worth making the copies three "
        f"times - {held[False].count} ciphertexts at {held[False].nbytes / 2 ** 20:.0f} MiB against "
        f"{held[True].count} at {held[True].nbytes / 2 ** 20:.0f}")


def test_the_streamed_index_mapping_matches_pcmm():
    """`pcmm` reads `prepared[(pack * out + j) % in_dim]`; the stream has to invert that exactly.

    Getting it wrong permutes which weight diagonal meets which copy, which is still a linear map and
    still produces plausible numbers - so it is asserted against `pcmm` itself rather than by eye.
    """
    x, w, _ = sample(SMALL, 11)
    weight = encode_weight(SMALL, w)
    engine, stages = build(SMALL, 14)
    cts = np.array([engine.encrypt(m) for m in encode_activations(SMALL, x)], dtype=object)
    _, complexified = stages.stage_01_complexify_x(cts, layer_index=0)

    direct = stages.pcmm(weight, stages.stage_02_make_rotated_copies(complexified))
    streamed = stages.pcmm_streamed(weight, stages.iter_rotated_copies(complexified))
    for index, (a, b) in enumerate(zip(direct, streamed)):
        difference = np.abs(np.asarray(engine.decrypt(a)) - np.asarray(engine.decrypt(b))).max()
        assert difference < 1e-12, f"output {index} differs by {difference:.3g}"


def test_iter_rotated_copies_yields_what_the_array_form_holds():
    """Same ciphertexts, same positions - the generator only changes when they exist."""
    x, _, _ = sample(SMALL, 13)
    engine, stages = build(SMALL, 14)
    cts = np.array([engine.encrypt(m) for m in encode_activations(SMALL, x)], dtype=object)
    _, complexified = stages.stage_01_complexify_x(cts, layer_index=0)

    array = stages.stage_02_make_rotated_copies(complexified)
    streamed = dict(stages.iter_rotated_copies(complexified))
    assert sorted(streamed) == list(range(len(array)))
    for index, expected in enumerate(array):
        got = np.asarray(engine.decrypt(streamed[index]))
        assert np.abs(got - np.asarray(engine.decrypt(expected))).max() < 1e-12, f"copy {index}"
