"""`compact`: every option that trades recomputation for held memory, and the promise that it is exact.

The options are listed in `layer.COMPACT_OPTIONS` and all of them are off by default. What they have
in common is that none changes a value - they change only what is alive while a value is produced -
so the tests here are equalities, not tolerances.

`THORFHE_DEBUG=1` makes a compact layer say what it turned on.
"""
import os

import numpy as np
import pytest

from thorfhe import SMALL, THOR_BERT, ClearEngine
from thorfhe.encoding import encode_activations
from thorfhe.layer import COMPACT_OPTIONS, EncoderLayer, encode_layer, debug_enabled


def weights_for(geometry, seed):
    rng = np.random.default_rng(seed)
    f = geometry.features
    return {"query.weight": rng.normal(0, .04, (f, f)), "query.bias": rng.normal(0, .01, f),
            "key.weight": rng.normal(0, .04, (f, f)), "key.bias": rng.normal(0, .01, f),
            "value.weight": rng.normal(0, .04, (f, f)), "value.bias": rng.normal(0, .01, f),
            "attention.output.dense.weight": rng.normal(0, .04, (f, f)),
            "attention.output.dense.bias": rng.normal(0, .01, f),
            "attention.output.LayerNorm.weight": rng.normal(1, .1, f),
            "attention.output.LayerNorm.bias": rng.normal(0, .1, f),
            "intermediate.dense.weight": rng.normal(0, .04, (4 * f, f)),
            "intermediate.dense.bias": rng.normal(0, .01, 4 * f),
            "output.dense.weight": rng.normal(0, .04, (f, 4 * f)),
            "output.dense.bias": rng.normal(0, .01, f),
            "output.LayerNorm.weight": rng.normal(1, .1, f),
            "output.LayerNorm.bias": rng.normal(0, .1, f)}


def test_compact_sets_every_option_it_lists():
    """The flag and the list cannot drift apart: one is the documentation of the other."""
    engine = ClearEngine(THOR_BERT, depth=20)
    plain, compact = EncoderLayer(engine), EncoderLayer(engine, compact=True)
    owners = lambda layer: (layer.attention, layer.dense, layer.norm, layer.feedforward)  # noqa: E731

    assert "stream_qkv" in COMPACT_OPTIONS
    assert all(not owner.stream_qkv for owner in owners(plain)), "off by default"
    assert all(owner.stream_qkv for owner in owners(compact)), "and on for every stage owner"
    assert compact.compact and not plain.compact


def test_lazy_weights_encode_what_the_encoders_produce():
    """The cheap half of `compact`, checked on one field instead of a layer.

    `encode_layer` eagerly is 9.7 GiB at THOR's geometry, and it cannot be run at `SMALL` at all -
    the feed-forward encoders are not defined there. So this compares one lazily-built field against
    the encoders called directly, which is what the thunk does, and does it without building the
    other seven.
    """
    weights = weights_for(THOR_BERT, 23)
    lazy = encode_layer(weights, 0, lazy=True)

    from thorfhe import encode_bias, encode_weight
    want_weight = encode_weight(THOR_BERT, weights["query.weight"])
    want_bias = encode_bias(THOR_BERT, weights["query.bias"])
    got_weight, got_bias = lazy.query

    assert got_weight.shape == want_weight.shape
    for index in np.ndindex(want_weight.shape):
        assert np.array_equal(np.asarray(got_weight[index]), np.asarray(want_weight[index])), index
    for index in np.ndindex(np.asarray(want_bias).shape):
        assert np.array_equal(np.asarray(got_bias[index]), np.asarray(want_bias[index])), index


def test_lazy_weights_reject_a_field_that_is_not_one():
    """A typo has to say so rather than resolve to something empty."""
    lazy = encode_layer(weights_for(THOR_BERT, 23), 0, lazy=True)
    with pytest.raises(AttributeError, match="not one of a layer's weights"):
        lazy.querry


def test_streaming_changes_nothing_a_layer_computes():
    """The half that reorders arithmetic, on a whole layer at the only geometry that can run one.

    The attention stages are transcribed from `he.py` for four output ciphertexts, so this cannot be
    done at `SMALL` - it is two layer forwards at THOR's geometry, which is why both sides encode
    lazily. What differs between them is `stream_qkv` alone.

    Exact, not close: streaming reorders the same additions, so any drift at all is a bug in the
    index mapping rather than a tolerance to widen.
    """
    rng = np.random.default_rng(17)
    weights = weights_for(THOR_BERT, 17)
    activations = rng.normal(0, 1.0, (THOR_BERT.dim, THOR_BERT.features))

    outputs = {}
    for stream in (False, True):
        engine = ClearEngine(THOR_BERT, depth=37, bootstrap_level=20)
        engine.bootstrap_message_margin = 1e9
        layer = EncoderLayer(engine, compact=stream, refresh_after_dense=True,
                             binary_rotations=True)
        for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
            owner.check_ranges = False
        packed = np.array([engine.encrypt(m)
                           for m in encode_activations(THOR_BERT, activations)],
                          dtype=object)
        result = layer.forward(packed, encode_layer(weights, 0, lazy=True),
                               layer.padding_mask(THOR_BERT.dim), 0)
        outputs[stream] = np.stack([np.asarray(engine.decrypt(ct)) for ct in result])
        del engine, layer, packed, result

    difference = float(np.max(np.abs(outputs[False] - outputs[True])))
    magnitude = float(np.max(np.abs(outputs[False])))
    assert difference / magnitude < 1e-12, (
        f"streaming moved the layer's output by {difference:.3g} against {magnitude:.3g}")


def test_the_rotation_plan_is_made_on_the_configuration_that_runs():
    """`plan_rotations` takes `compact`, because the basis is chosen from measured frequencies.

    `factored_basis` picks its extra keys greedily from how often each index is actually rotated by,
    and streaming makes the QKV copies three times instead of once. A basis chosen on the plain
    profile and used by a compact run would spend keys where the run does not rotate - which costs
    time, not correctness, so nothing else would report it.

    Measured at THOR's geometry with six extra keys the greedy lands on the same 21 either way, and
    the rotation count goes 3345 -> 3465. That the answer is currently the same is not a reason to
    derive it from the wrong profile.
    """
    import inspect

    from thorfhe.he import plan_rotations

    assert "compact" in inspect.signature(plan_rotations).parameters, (
        "plan_rotations must be able to model a compact run, or the basis is chosen for a rotation "
        "profile the run does not have")


@pytest.mark.parametrize("value,expected", [("1", True), ("", False), (None, False)])
def test_debug_is_read_per_call(monkeypatch, value, expected):
    """So a test can turn it on, and so a long run cannot be stuck with what the environment had."""
    if value is None:
        monkeypatch.delenv("THORFHE_DEBUG", raising=False)
    else:
        monkeypatch.setenv("THORFHE_DEBUG", value)
    assert debug_enabled() is expected


def test_debug_reports_what_compact_turned_on(monkeypatch, capsys):
    monkeypatch.setenv("THORFHE_DEBUG", "1")
    EncoderLayer(ClearEngine(THOR_BERT, depth=20), compact=True)
    assert "[compact]" in capsys.readouterr().out

    monkeypatch.delenv("THORFHE_DEBUG")
    EncoderLayer(ClearEngine(THOR_BERT, depth=20), compact=True)
    assert "[compact]" not in capsys.readouterr().out
