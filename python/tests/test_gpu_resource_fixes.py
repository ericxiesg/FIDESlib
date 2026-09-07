"""The three things a GPU run of a full layer tripped over, pinned so they cannot come back.

All three were reported from a GV100 (`bugs/GPU-benchmark-bugs-20260907.md`) and all three are
invisible on the clear engine: a rotation key that is never generated, a plaintext that is re-encoded
instead of reused, and a grid of ciphertexts that is held instead of streamed. None of them changes a
single decrypted value, which is why they need tests of their own rather than an accuracy assertion.
"""
import numpy as np
import pytest

from thorfhe import SMALL, ClearEngine, Stages, block_diagonal_masks, plan_rotation_keys

DEPTH = 12


def small_stages(depth=DEPTH, engine=None):
    engine = ClearEngine(SMALL, depth=depth) if engine is None else engine
    low, high = block_diagonal_masks(SMALL)
    return engine, Stages(engine, SMALL, masks=low, complement_masks=high)


# ---------------------------------------------------------------- rotation by zero
def test_a_zero_rotation_never_reaches_the_engine():
    """`EvalRotate(x, 0)` needs a key OpenFHE has no index for, and rotating by nothing needs none."""
    engine, stages = small_stages()
    ct = engine.encrypt(np.arange(SMALL.slot_count, dtype=float))

    assert stages.rotate(ct, 0) is ct
    assert stages.rotate(ct, SMALL.slot_count) is ct       # the same rotation, spelled differently
    assert engine.rotations_used == set()

    moved = stages.rotate(ct, 1)
    assert moved is not ct
    assert engine.rotations_used == {SMALL.slot_count - 1}


def test_the_key_plan_never_asks_for_index_zero():
    """A 123 MiB rotation key for a no-op, and one the GPU cannot generate."""
    plan = plan_rotation_keys(SMALL, depth=DEPTH, scope="qkv")
    assert plan
    assert 0 not in plan


def test_the_key_plan_refuses_a_starved_level_budget():
    """Stages 01-05 cost 8 levels. Below that, dropping the negatives would hide missing keys."""
    with pytest.raises(ValueError, match="below level 0"):
        plan_rotation_keys(SMALL, depth=6, scope="qkv")


def test_the_key_plan_rejects_an_unknown_scope():
    with pytest.raises(ValueError, match="scope must be"):
        plan_rotation_keys(SMALL, depth=DEPTH, scope="everything")


# ---------------------------------------------------------------- plaintext reuse
class CountingEngine(ClearEngine):
    """A clear engine that also offers the light-plaintext API, so `Stages.plaintext` engages."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encodes = 0

    def encode_to_light_plaintext(self, message, level=None):
        self.encodes += 1
        return np.asarray(message)


def test_a_mask_is_encoded_once_however_often_it_is_used():
    """A GV100 run reported 378 plaintexts holding 6.4 GB - one per multiply, not one per mask."""
    engine, stages = small_stages(engine=CountingEngine(SMALL, depth=DEPTH))
    ct = engine.encrypt(np.ones(SMALL.slot_count))
    mask = (np.arange(SMALL.slot_count) % SMALL.n_slot < 2).astype(float)

    for _ in range(10):
        stages.multiply(ct, mask)
    assert engine.encodes == 1

    # keyed by content, not identity: the port rebuilds several masks at their call site
    for _ in range(10):
        stages.multiply(ct, mask.copy())
    assert engine.encodes == 1

    stages.multiply(ct, 1.0 - mask)
    assert engine.encodes == 2


def test_the_clear_engine_still_gets_its_arrays():
    """`plaintext` has to be a no-op without the light-plaintext API, or the numpy path breaks."""
    engine, stages = small_stages()
    mask = np.ones(SMALL.slot_count)
    assert stages.plaintext(mask) is mask
    assert stages.plaintext(3.0) == 3.0


# ---------------------------------------------------------------- pcmm working set
def test_pcmm_streams_the_grid_it_used_to_hold():
    """`out_dim * diag_dim` ciphertexts at 50 MB each is what exhausted a 32 GB card.

    Liveness is awkward to sample - `ClearCiphertext` has `__slots__` and so is not weakref-able - but
    the property that matters is structural and exact: `pcmm` must never materialise the grid. Making
    the grid builder fail proves it, and comparing against the grid separately proves the streaming
    version still computes the same thing.
    """
    engine, stages = small_stages(depth=40)
    out_dim, diag_dim = SMALL.n_output_ciphertexts, SMALL.diag_count

    rng = np.random.default_rng(4)
    inputs = np.array([engine.encrypt(rng.normal(size=SMALL.slot_count))
                       for _ in range(SMALL.n_in_complex)], dtype=object)
    weights = np.empty((out_dim, diag_dim, SMALL.n_in_complex), dtype=object)
    for index in np.ndindex(weights.shape):
        weights[index] = rng.normal(size=SMALL.slot_count)

    streamed = stages.pcmm(weights, inputs)

    # the grid transcription still exists and still agrees, so the rewrite is a scheduling change
    grid = stages.parallel_diagonal_pc_mult(weights, inputs)
    assert grid.shape == (out_dim, diag_dim)
    folded = np.empty((out_dim,), dtype=object)
    for out_index in range(out_dim):
        temp = stages.level_down(grid[out_index, 0], by=1)
        for diag_index in range(1, diag_dim):
            temp = stages.add(temp, stages.rotate_internal(grid[out_index, diag_index],
                                                           SMALL.n_blocks - diag_index))
        folded[out_index] = temp
    for a, b in zip(streamed, folded):
        assert np.abs(engine.decrypt(a) - engine.decrypt(b)).max() < 1e-12
        assert engine.level(a) == engine.level(b)

    # ... and it is a scheduling change that never builds the grid
    def refuse(*args, **kwargs):
        raise AssertionError("pcmm must not materialise the (out_dim, diag_dim) grid")

    stages.parallel_diagonal_pc_mult = refuse
    stages.pcmm(weights, inputs)
