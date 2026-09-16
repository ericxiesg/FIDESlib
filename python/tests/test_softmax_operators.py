"""The operators stages 06-08 are built from, tested one at a time.

Stages 06, 07 and 08 each have a test that checks the whole stage against the plaintext model, and
those are the tests that matter for correctness. What they cannot do is say *where* a stage went
wrong, and the softmax has now cost several rounds of exactly that question. Each operator here has a
contract stated in its own docstring and, until now, no test of its own - only the end-to-end ones,
which means a fault in any of them surfaces as "softmax is wrong".

Everything here runs on the clear engine, so these are not GPU stubs: they are exact assertions on
layout and algebra, and they fail in milliseconds on a laptop.
"""
import numpy as np
import pytest

from thorfhe import THOR_BERT, ClearEngine, block_diagonal_masks
from thorfhe.attention import attention_rotate_masks, ccmm_masks, make_copies_masks, transpose_masks
from thorfhe.numeric import DeltaCiphertext
from thorfhe.softmax import Softmax

G = THOR_BERT
DEPTH = 60


@pytest.fixture(scope="module")
def masks():
    return (block_diagonal_masks(G), transpose_masks(G), make_copies_masks(G),
            attention_rotate_masks(G), ccmm_masks(G))


def used_slots():
    return ((np.arange(G.slot_count) % G.n_slot) < G.n_blocks).astype(float)


@pytest.fixture
def stages(masks):
    (low, high), transpose, copies, attention, ccmm = masks
    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    return engine, Softmax(engine, G, masks=low, complement_masks=high, transpose=transpose,
                           copies=copies, attention=attention, ccmm=ccmm,
                           ones=engine.encrypt(used_slots()))


# ---------------------------------------------------------------- the denominator
def test_sum_over_groups_totals_every_diagonal_into_every_group(stages):
    """The softmax denominator: sum the score ciphertexts, then fold the groups onto each other.

    The layout makes this the sum over all `dim` diagonals of a row - 8 ciphertexts carrying `pack`
    diagonals each, times `slot_count / group_size` groups - and the fold has to leave that total in
    *every* group, because `_broadcast_softmax` reads it from all of them.

    Untested until now, and directly implicated: the device run's denominator came back at 7.0e-5
    against the 4.9e-4 `he_inv` requires, and a fold that dropped terms would look exactly like that.
    """
    engine, st = stages
    rng = np.random.default_rng(3)
    terms = []
    expected = np.zeros(G.slot_count)
    for _ in range(2 * G.n_output_ciphertexts):
        values = rng.uniform(0.1, 1.0, G.slot_count) * used_slots()
        terms.append(engine.encrypt(values))
        expected += values

    got = np.real(engine.decrypt(st._sum_over_groups(terms)))

    groups = G.slot_count // G.group_size
    within = np.arange(G.slot_count) % G.group_size
    for group in range(groups):
        window = slice(group * G.group_size, (group + 1) * G.group_size)
        # every group holds the total over all groups, position by position within the group
        folded = expected.reshape(groups, G.group_size).sum(axis=0)
        assert np.allclose(got[window], folded), f"group {group} does not hold the total"
    assert within.size == G.slot_count   # the reshape above assumes a contiguous group layout


def test_sum_over_groups_loses_nothing(stages):
    """The total over the used slots has to survive the fold: it is what is being inverted."""
    engine, st = stages
    terms = [engine.encrypt(used_slots() * 0.25) for _ in range(2 * G.n_output_ciphertexts)]
    got = np.real(engine.decrypt(st._sum_over_groups(terms)))
    per_group = used_slots().reshape(G.slot_count // G.group_size, G.group_size).sum(axis=0)
    assert np.allclose(got[: G.group_size], 0.25 * len(terms) * per_group)


# ---------------------------------------------------------------- the delta bookkeeping
def test_restore_magnitude_leaves_the_value_it_represents_unchanged(stages):
    """`DeltaCiphertext` means the value is `ciphertext / delta`; restoring magnitude must not move it.

    That is the whole point of the pair: delta shrinks quadratically through Goldschmidt, so the
    ciphertext is scaled back up by a conjugate doubling and an integer multiply - both level-free -
    and the represented value has to come out identical. It is exact arithmetic here, so `identical`
    means to floating-point, not to a tolerance.
    """
    engine, st = stages
    rng = np.random.default_rng(5)
    values = rng.uniform(-1.0, 1.0, G.slot_count) * used_slots()

    for delta in (1.0, 1e-2, 1e-4, 3.9e-3):
        a = DeltaCiphertext(engine.encrypt(values * delta), delta)
        b = DeltaCiphertext(engine.encrypt(values * delta * 0.5), delta)
        before = np.real(engine.decrypt(a.ciphertext)) / a.delta

        a_out, b_out = st._restore_magnitude(a, b)
        after = np.real(engine.decrypt(a_out.ciphertext)) / a_out.delta

        assert np.allclose(before, after, rtol=1e-12, atol=1e-14), (
            f"delta={delta:g}: restoring the magnitude moved the value by "
            f"{np.max(np.abs(after - before)):.3g}")


def test_restore_magnitude_actually_restores_the_magnitude(stages):
    """And it has to do something: the ciphertext must come back up, not just stay put."""
    engine, st = stages
    delta = 1e-4
    values = used_slots() * 0.5
    a = DeltaCiphertext(engine.encrypt(values * delta), delta)
    b = DeltaCiphertext(engine.encrypt(values * delta), delta)

    small = np.max(np.abs(np.real(engine.decrypt(a.ciphertext))))
    a_out, _ = st._restore_magnitude(a, b)
    large = np.max(np.abs(np.real(engine.decrypt(a_out.ciphertext))))

    assert large > 10 * small, f"magnitude went {small:.3g} -> {large:.3g}, which is not a restore"


# ---------------------------------------------------------------- streamed vs array accumulation
def test_streamed_accumulation_equals_the_array_form(stages):
    """`_accumulate_streamed` exists to save 1.8 GiB; it has to compute the same thing.

    The array form builds all `in_dim` diagonals and then consumes them; the streamed form consumes
    each as it is produced, in an order that is not the natural one. The loop only accumulates, so
    order cannot matter - but that is the claim, and this is the test of it.
    """
    engine, st = stages
    rng = np.random.default_rng(7)
    in_dim = 2 * G.pack
    left = [engine.encrypt(rng.uniform(-0.5, 0.5, G.slot_count)) for _ in range(2)]
    diagonals = [engine.encrypt(rng.uniform(-0.5, 0.5, G.slot_count)) for _ in range(in_dim)]

    array_form = st._accumulate_product(left, list(diagonals), in_dim)
    natural = list(enumerate(diagonals))
    streamed = st._accumulate_streamed(left, reversed(natural))    # deliberately the wrong order

    assert array_form.shape == streamed.shape
    for index in np.ndindex(array_form.shape):
        one, other = array_form[index], streamed[index]
        assert (one is None) == (other is None), f"{index}: one form accumulated and the other did not"
        if one is not None:
            assert np.allclose(np.asarray(engine.decrypt(one)), np.asarray(engine.decrypt(other))), (
                f"{index}: the streamed and array forms disagree")


def test_streamed_accumulation_refuses_a_diagonal_at_the_wrong_level(stages):
    """It cannot scan ahead for a common level, so it holds the rest to the first one - and checks.

    A violation would otherwise be a FIXEDMANUAL scale mismatch several stages downstream, which is
    the failure mode this whole engine exists to convert into a message at the point of the mistake.
    """
    engine, st = stages
    rng = np.random.default_rng(11)
    left = [engine.encrypt(rng.uniform(-0.5, 0.5, G.slot_count)) for _ in range(2)]
    diagonals = [engine.encrypt(rng.uniform(-0.5, 0.5, G.slot_count)) for _ in range(2 * G.pack)]
    diagonals[3] = engine.level_down(diagonals[3], by=1)

    with pytest.raises(ValueError, match="not.*like the first"):
        st._accumulate_streamed(left, enumerate(diagonals))


# ---------------------------------------------------------------- level alignment
def test_align_brings_operands_to_one_level_without_changing_them(stages):
    """`align` is used wherever two routes meet; dropping levels is exact, so values must not move."""
    engine, st = stages
    rng = np.random.default_rng(13)
    values = [rng.uniform(-1.0, 1.0, G.slot_count) for _ in range(3)]
    cts = [engine.level_down(engine.encrypt(v), by=drop)
           for v, drop in zip(values, (0, 2, 5))]

    aligned = st.align(*cts)

    levels = {engine.level(ct) for ct in aligned}
    assert levels == {DEPTH - 5}, f"align left levels {levels}, not all at the lowest"
    for value, ct in zip(values, aligned):
        assert np.allclose(np.real(engine.decrypt(ct)), value), "align changed a value"


def test_align_is_a_no_op_when_the_levels_already_agree(stages):
    engine, st = stages
    cts = [engine.encrypt(np.full(G.slot_count, 0.25)) for _ in range(3)]
    aligned = st.align(*cts)
    assert {engine.level(ct) for ct in aligned} == {DEPTH}


# ---------------------------------------------------------------- the Newton refinement
def test_update_inv_D_squares_the_probabilities(stages):
    """One halving of the temperature: `softmax(2x)` is proportional to `softmax(x)` squared.

    The arithmetic that gets there is not obvious from the code. `inv_D` carries `delta / D` rather
    than `1 / D` (the `DeltaCiphertext` convention), and `k` is chosen as `1 / (2 * delta)`, so

        scaled = 2 * exp * (delta / D) * k ~ exp / D

    which is the probability itself - and squaring it is the halving. Everything downstream of the
    first `he_inv` depends on this identity and nothing tested it.
    """
    engine, st = stages
    rng = np.random.default_rng(23)
    used = used_slots()
    terms = []
    total = np.zeros(G.slot_count)
    for _ in range(2 * G.n_output_ciphertexts):
        values = rng.uniform(0.2, 1.0, G.slot_count) * used
        terms.append(values)
        total += values

    groups = G.slot_count // G.group_size
    denominator = np.tile(total.reshape(groups, G.group_size).sum(axis=0), groups)
    delta = 1.0 / 512
    inv_D = engine.encrypt(np.where(used, delta / np.where(denominator > 0, denominator, 1.0), 0.0))

    squared, _, _, _ = st.update_inv_D(
        [engine.encrypt(v) for v in terms], [used] * len(terms), inv_D,
        delta=delta, precision=0.5, alpha=0.1)

    carried = used > 0
    for index, values in enumerate(terms):
        probability = np.zeros(G.slot_count)
        probability[carried] = values[carried] / denominator[carried]
        got = np.real(engine.decrypt(squared[index]))
        assert np.allclose(got[carried], probability[carried] ** 2, rtol=2e-2), (
            f"ciphertext {index}: the squared probabilities are off by "
            f"{np.max(np.abs(got[carried] - probability[carried] ** 2)):.3g}")


def test_update_inv_D_confines_the_final_inverse_to_the_first_group(stages):
    """`final=True` keeps only the first group, which is where `_broadcast_softmax` starts reading."""
    engine, st = stages
    used = used_slots()
    terms = [np.where(used > 0, 0.5, 0.0) for _ in range(2 * G.n_output_ciphertexts)]
    inv_D = engine.encrypt(np.where(used > 0, 1.0 / 512 / 64, 0.0))

    for final, expect_beyond in ((False, True), (True, False)):
        _, inverse, _, _ = st.update_inv_D(
            [engine.encrypt(v) for v in terms], [used] * len(terms), inv_D,
            delta=1.0 / 512, precision=0.5, alpha=0.1, final=final)
        beyond = np.real(engine.decrypt(inverse))[G.group_size:]
        assert np.any(np.abs(beyond) > 0) == expect_beyond, (
            f"final={final}: slots past the first group should "
            f"{'carry' if expect_beyond else 'be empty'}")
