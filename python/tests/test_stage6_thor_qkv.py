"""Stage 6: THOR stages 01-05 (complexify, rotated copies, QKV block-diagonal product).

Three layers of checking, cheapest first:

1. the packing is right - the numpy slot pipeline reproduces ``x @ w.T + 2 * b`` exactly, at BERT's
   real geometry and at a small one, which is what pins down the encoders and the rotation directions;
2. the level/scale schedule is right - the clear engine runs in strict FIXEDMANUAL mode and would
   raise if a stage added operands at different scales;
3. the FHE port is right - the same stage code over ``pyfideslib.Engine`` agrees with the clear run.

Layer 3 is the slow one, so it uses ``thorfhe.SMALL``: 4096 slots, still 128 tokens, and the same
``pack != n_slot`` structure THOR has.
"""
import inspect

import numpy as np
import pytest

import thorfhe
from thorfhe import (SMALL, THOR_BERT, ClearEngine, LightWeights, ScaleMismatch, Stages,
                     block_diagonal_masks, decode_linear_output, encode_activations, encode_bias,
                     encode_weight, encrypt_activations, plan_rotation_keys)
from thorfhe.layer import ACTIVATION_SCALE

DEPTH = 12
LOG_N = 13


def sample(geometry, seed):
    rng = np.random.default_rng(seed)
    return (rng.normal(size=(geometry.dim, geometry.features)) * 0.1,
            rng.normal(size=(geometry.features, geometry.features)) * 0.05,
            rng.normal(size=(geometry.features,)) * 0.1)


def run_clear(geometry, x, w, b, depth=DEPTH, layer_index=0):
    engine = ClearEngine(geometry, depth=depth)
    low, high = block_diagonal_masks(geometry)
    stages = Stages(engine, geometry, masks=low, complement_masks=high)

    cts = np.array([engine.encrypt(m) for m in encode_activations(geometry, x)], dtype=object)
    _, x_cplx = stages.stage_01_complexify_x(cts, layer_index=layer_index)
    rotated = stages.stage_02_make_rotated_copies(x_cplx)
    out = stages.stage_03_query(rotated, encode_weight(geometry, w), encode_bias(geometry, b))
    return engine, out


# ---------------------------------------------------------------- 1. the packing
def test_every_non_linearity_is_told_the_same_amplitude():
    """The three non-linearities each divide the carrier out of their argument, and it is one number.

    Everything else in the layer is linear and carries the amplitude through untouched, so getting it
    wrong cancels - except here, where it silently computes a different function of a different input.
    The network has exactly three such places and this enumerates them:

    * GELU, which takes `carrier` as a parameter (`thorfhe.feedforward`);
    * `tanh` in the pooler, where it is folded into the weight, because the halved bias means there is
      no single factor left to undo afterwards (`encode_weight_pooler`);
    * the softmax, where it rides in the plaintext `SOFTMAX_SCALES` - and where, precisely because
      nothing had to agree with anything, the constant was wrong by four for every layer and then by
      two again for layer 2.

    They were three separate declarations of 2.0 until `ACTIVATION_SCALE` became one.
    """
    from thorfhe.encoding import encode_weight_pooler
    from thorfhe.feedforward import FeedForwardStages
    from thorfhe.layer import DEFAULT_SOFTMAX_SCALE, SOFTMAX_SCALES

    assert FeedForwardStages.carrier == ACTIVATION_SCALE, "GELU divides out a different amplitude"
    assert (inspect.signature(encode_weight_pooler).parameters["carrier"].default
            == ACTIVATION_SCALE), "the pooler is encoded for a different amplitude"
    assert SOFTMAX_SCALES == {}, "a layer's softmax carries its own scale"
    assert 2 * ACTIVATION_SCALE ** 2 * DEFAULT_SOFTMAX_SCALE == 1 / np.sqrt(THOR_BERT.n_out), (
        "the softmax scale no longer matches the amplitude it is derived from - `he_softmax` sees "
        "`2 * s**2 * (q.k) * scale` and BERT's score is `(q.k) / sqrt(head_dim)`")


@pytest.mark.parametrize("geometry", [SMALL, THOR_BERT], ids=["small", "bert"])
@pytest.mark.parametrize("entry", [1.0, ACTIVATION_SCALE], ids=["at-1", "at-activation-scale"])
def test_qkv_computes_xw_plus_bias(geometry, entry):
    """A QKV stage is a linear layer; decode its output and it must be exactly that layer.

    The two terms scale differently with the amplitude the layer is entered at, and that is the whole
    point of running this at two of them. `encode_weight` halves the weights to pay for the
    `y + conj(y)` that makes the result real, and THOR's `encode_b` does not halve the bias - so the
    bias is added *between* the halving and the doubling and comes out at `2b` whatever `entry` is,
    while `x @ w.T` comes out at `entry` times itself.

    Pinned at one amplitude this reads `x @ w.T + 2b`, which is a true statement about a pipeline
    entered at 1 and a misleading one about the pipeline that runs: layers are entered at
    `ACTIVATION_SCALE`, where the same stage gives `2 * (x @ w.T + b)`, i.e. twice BERT's own `q`.
    Reading it as the first cost `SOFTMAX_SCALES` a factor of four - see
    `test_stage_06_hands_the_softmax_berts_own_score`.
    """
    x, w, b = sample(geometry, seed=3)
    engine, out = run_clear(geometry, entry * x, w, b)
    got = decode_linear_output(geometry, [engine.decrypt(o) for o in out])
    assert np.abs(got - (entry * (x @ w.T) + 2 * b)).max() < 1e-12


@pytest.mark.parametrize("kwargs,message", [
    (dict(n_blocks=12), "does not fit in n_slot"),
    (dict(features=50), "divisible by both"),
    (dict(pack=16), "divisible by pack"),
    (dict(dim=12), "power of two"),
])
def test_geometry_rejects_inconsistent_shapes(kwargs, message):
    """The shape invariants that still hold; `features // n_out == n_blocks` was a QKV coincidence."""
    base = dict(dim=128, pack=4, n_slot=8, n_blocks=6, features=48, n_in=16, n_out=8)
    with pytest.raises(ValueError, match=message):
        thorfhe.Geometry(**{**base, **kwargs})


def test_unused_slots_stay_empty():
    """Only ``n_blocks`` of every ``n_slot`` carry data; the padding must not pick anything up."""
    x, w, b = sample(SMALL, seed=4)
    engine, out = run_clear(SMALL, x, w, b)
    padding = (np.arange(SMALL.slot_count) % SMALL.n_slot) >= SMALL.n_blocks
    for ct in out:
        assert np.abs(engine.decrypt(ct)[padding]).max() < 1e-12


# ---------------------------------------------------------------- 2. the level schedule
def test_level_schedule():
    """Stages 01-05 cost 8 levels: THOR's deliberate level_down(6) plus one each for pcmm's two rescales."""
    x, w, b = sample(SMALL, seed=5)
    engine, out = run_clear(SMALL, x, w, b)
    assert [engine.level(o) for o in out] == [DEPTH - 8] * SMALL.n_output_ciphertexts


def test_scale_discipline_is_enforced():
    """The clear engine is strict FIXEDMANUAL: he.py's mask-and-subtract would be caught here.

    ``x - mask * x`` is exactly the operation ``Stages.rotate_internal`` had to be rewritten to avoid,
    and it fails twice over: first on the scale, and then - once the product is rescaled - on the level.
    """
    engine = ClearEngine(SMALL, depth=DEPTH)
    ct = engine.encrypt(np.ones(SMALL.slot_count))
    masked = engine.multiply(ct, np.ones(SMALL.slot_count))  # now at Delta^2, same level
    with pytest.raises(ScaleMismatch, match="scale"):
        engine.subtract(ct, masked)
    with pytest.raises(ScaleMismatch, match="level"):
        engine.subtract(ct, engine.rescale(masked))
    engine.subtract(engine.level_down(ct, 1), engine.rescale(masked))  # matched on both: fine


def test_rotation_plan_covers_every_rotation():
    """plan_rotation_keys must name every index the stages ask for, at a high enough level."""
    plan = plan_rotation_keys(SMALL, depth=DEPTH, scope="qkv")
    x, w, b = sample(SMALL, seed=6)
    engine, _ = run_clear(SMALL, x, w, b)
    assert engine.rotations_used <= set(plan)
    for delta, level in engine.rotation_levels.items():
        assert plan[delta] >= level


# ---------------------------------------------------------------- 3. the FHE port
@pytest.fixture(scope="module")
def thor_engine(device):
    # imported here, not at module scope: the checks above are numpy-only and must run without the
    # compiled extension.
    pf = pytest.importorskip("pyfideslib", reason="the pyfideslib extension is not built")
    plan = plan_rotation_keys(SMALL, depth=DEPTH, scope="qkv")
    engine = pf.Engine(device, log_n=LOG_N, depth=DEPTH, scaling_bits=50, first_mod_bits=55, dnum=3,
                       rotation_indexes=plan)
    assert engine.slots == SMALL.slot_count
    return engine


def test_fhe_stages_match_clear(thor_engine):
    """The same stage code on the FHE engine and on the numpy mirror must agree to CKKS noise."""
    geometry = SMALL
    x, w, b = sample(geometry, seed=7)

    clear_engine, clear_out = run_clear(geometry, x, w, b)

    weights = LightWeights(thor_engine, geometry)
    stages = weights.stages()
    cts = encrypt_activations(thor_engine, geometry, x)
    _, x_cplx = stages.stage_01_complexify_x(cts, layer_index=0)
    rotated = stages.stage_02_make_rotated_copies(x_cplx)
    out = stages.stage_03_query(rotated, weights.weight(w), weights.bias(b))

    assert [thor_engine.level(o) for o in out] == [clear_engine.level(o) for o in clear_out]
    for fhe_ct, clear_ct in zip(out, clear_out):
        got = thor_engine.decrypt(fhe_ct)
        assert np.max(np.abs(got - clear_engine.decrypt(clear_ct))) < 1e-5

    decoded = decode_linear_output(geometry, [thor_engine.decrypt(o) for o in out])
    assert np.abs(decoded - (x @ w.T + 2 * b)).max() < 1e-5


def test_weight_files_round_trip(thor_engine, tmp_path):
    """The on-disk layout THOR streams weights from, in the light-plaintext format."""
    geometry = SMALL
    x, w, b = sample(geometry, seed=10)

    path = thorfhe.qkv_path(tmp_path, "stage_03", 0)
    written = thorfhe.write_linear(thor_engine, geometry, path, w, b)
    assert written == geometry.n_output_ciphertexts * (geometry.diag_count * geometry.n_in_complex + 1)
    assert sum(f.stat().st_size for f in path.iterdir()) == thorfhe.bytes_on_disk(geometry)

    weight, bias = thorfhe.read_linear(thor_engine, geometry, path)
    stages = LightWeights(thor_engine, geometry).stages()
    cts = encrypt_activations(thor_engine, geometry, x)
    _, x_cplx = stages.stage_01_complexify_x(cts, layer_index=0)
    out = stages.stage_03_query(stages.stage_02_make_rotated_copies(x_cplx), weight, bias)

    decoded = decode_linear_output(geometry, [thor_engine.decrypt(o) for o in out])
    assert np.abs(decoded - (x @ w.T + 2 * b)).max() < 1e-5


def test_fhe_key_and_value_reuse_the_same_input(thor_engine):
    """Stages 03-05 differ only in their weights: one set of rotated copies feeds all three."""
    geometry = SMALL
    x, wq, bq = sample(geometry, seed=8)
    _, wv, bv = sample(geometry, seed=9)

    weights = LightWeights(thor_engine, geometry)
    stages = weights.stages()
    cts = encrypt_activations(thor_engine, geometry, x)
    _, x_cplx = stages.stage_01_complexify_x(cts, layer_index=0)
    rotated = stages.stage_02_make_rotated_copies(x_cplx)

    q = stages.stage_03_query(rotated, weights.weight(wq), weights.bias(bq))
    v = stages.stage_05_value(rotated, weights.weight(wv), weights.bias(bv))

    assert np.abs(decode_linear_output(geometry, [thor_engine.decrypt(o) for o in q])
                  - (x @ wq.T + 2 * bq)).max() < 1e-5
    assert np.abs(decode_linear_output(geometry, [thor_engine.decrypt(o) for o in v])
                  - (x @ wv.T + 2 * bv)).max() < 1e-5
