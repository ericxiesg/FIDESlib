"""The three things a GPU run of a full layer tripped over, pinned so they cannot come back.

All three were reported from a GV100 (`bugs/GPU-benchmark-bugs-20260907.md`) and all three are
invisible on the clear engine: a rotation key that is never generated, a plaintext that is re-encoded
instead of reused, and a grid of ciphertexts that is held instead of streamed. None of them changes a
single decrypted value, which is why they need tests of their own rather than an accuracy assertion.
"""
import numpy as np
import pytest

from thorfhe import (SMALL, ClearEngine, ScaleMismatch, Stages, block_diagonal_masks,
                     plan_rotation_keys)
from thorfhe import budget as budget_model

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
    """Stages 01-05 cost 8 levels, so 6 cannot run and the planner must say so rather than
    return a plan that is short of keys.

    The engine now catches this first, at the operation that would go below level 0, which is the
    only place that stays sound once rotations share indices - see
    ``test_a_negative_level_is_an_error_not_a_silent_result``.
    """
    with pytest.raises((ValueError, ScaleMismatch), match="level"):
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


# ---------------------------------------------------------------- binary rotations
def test_binary_rotations_agree_with_direct_ones():
    """The decomposition must be exact: rotations compose additively and cost no levels."""
    low, high = block_diagonal_masks(SMALL)
    direct_engine = ClearEngine(SMALL, depth=20)
    binary_engine = ClearEngine(SMALL, depth=20)
    direct = Stages(direct_engine, SMALL, masks=low, complement_masks=high)
    binary = Stages(binary_engine, SMALL, masks=low, complement_masks=high, binary_rotations=True)

    values = np.arange(SMALL.slot_count, dtype=float)
    for delta in (1, 5, 16, 100, 1234, -7, -2048, SMALL.slot_count - 1, SMALL.slot_count):
        want = direct_engine.decrypt(direct.rotate(direct_engine.encrypt(values), delta))
        got = binary_engine.decrypt(binary.rotate(binary_engine.encrypt(values), delta))
        assert np.abs(got - want).max() == 0.0, delta

    assert all(index & (index - 1) == 0 for index in binary_engine.rotations_used)


def test_rotation_steps_are_the_binary_expansion():
    _, stages = small_stages()
    assert stages.rotation_steps(1) == [1]
    assert stages.rotation_steps(2048) == [2048]
    assert stages.rotation_steps(2049) == [1, 2048]
    assert stages.rotation_steps(7) == [1, 2, 4]
    for index in (3, 100, 1234, 4095):
        assert sum(stages.rotation_steps(index)) == index


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


# ---------------------------------------------------------------- level starvation
def test_a_negative_level_is_an_error_not_a_silent_result():
    """The only sound place to catch level starvation.

    ``plan_rotation_keys`` used to infer it from negative *rotation* levels, which stops working the
    moment two rotations share an index - the plan keeps the maximum level per index, so a use deep
    in the layer is masked by a shallow one. Under ``binary_rotations`` every index is shared, so the
    check silently passed configurations that cannot run.
    """
    engine = ClearEngine(SMALL, depth=1)
    ct = engine.encrypt(np.ones(SMALL.slot_count))
    assert engine.level_down(ct, 1).level == 0            # exactly empty is still legal
    with pytest.raises(ScaleMismatch, match="level -1"):
        engine.level_down(ct, 2)


def test_a_lenient_engine_still_allows_it():
    """Strictness is opt-out, so an exploratory run can deliberately go past the budget."""
    engine = ClearEngine(SMALL, depth=1, strict=False)
    ct = engine.encrypt(np.ones(SMALL.slot_count))
    assert engine.level_down(ct, 5).level == -4


# ---------------------------------------------------------------- memory model
def test_the_budget_model_reproduces_the_reported_key_size():
    """248 MiB a key at depth 50 / dnum 4, and K=11 recovered from the same log line."""
    k = budget_model.special_prime_count(log_n=16, depth=50, dnum=4,
                                         measured_key_mib=12152, keys=49)
    assert k == 11
    assert round(budget_model.key_bytes(log_n=16, level=50, special_primes=k,
                                        dnum=4) / budget_model.MIB) == 248


def test_an_unmeasured_level_budget_refuses_to_give_a_total():
    """A rosy total for a configuration nobody has run costs another twenty-minute OOM."""
    measured = budget_model.estimate(depth=50, dnum=4, rotation_keys=15, level_budget=(3, 3))
    assert measured.predictable
    assert measured.total > 32 * budget_model.GIB          # round 3 did not fit, and this says so

    unmeasured = budget_model.estimate(depth=50, dnum=4, rotation_keys=15, level_budget=(5, 5))
    assert not unmeasured.predictable
    assert unmeasured.total is None
    assert "UNKNOWN" in unmeasured.format(32 * budget_model.GIB)


def test_the_measured_level_budgets_reproduce_their_runs():
    """(4,4) at depth 51 was reported as 6882 MiB of plaintexts and 17452 MiB of keys.

    The model has to land on that, and then say the configuration does not fit - it was measured on a
    run that got through key generation and died at the first ciphertext copy.
    """
    got = budget_model.estimate(depth=51, dnum=4, rotation_keys=15, level_budget=(4, 4),
                                special_primes=12)
    resident = (got.rotation_keys + got.bootstrap_keys + got.bootstrap_plaintexts) / budget_model.MIB
    assert abs(resident - (17452 + 6882)) < 60           # within a quarter of a percent
    assert got.total > 32 * budget_model.GIB             # ... and still does not fit
    assert budget_model.bootstrap_depth(level_budget=(4, 4)) == 18
    assert budget_model.bootstrap_depth(level_budget=(5, 5)) is None
