"""A rotation basis has to reach every index, and reach it the same way twice.

Both properties are silent when broken. A decomposition whose steps do not sum to the index rotates
the wrong way and decrypts to a plausible-looking wrong answer; a decomposition that depends on
iteration order makes the key plan and the run disagree, and the missing key surfaces later still.
"""
import random

import pytest

from thorfhe.rotation import (RotationBasis, basis_bytes, binary_indices, factored_basis,
                              key_levels, rotation_cost)

MODULUS = 32768


def test_every_index_is_reachable_from_the_powers_of_two():
    basis = RotationBasis(binary_indices(MODULUS), MODULUS)
    for index in range(MODULUS):
        assert sum(basis.steps(index)) % MODULUS == index


def test_extra_keys_do_not_break_reachability():
    basis = RotationBasis(binary_indices(MODULUS) + [3, 2288, 4320, 32512], MODULUS)
    for index in range(MODULUS):
        assert sum(basis.steps(index)) % MODULUS == index


def test_zero_takes_no_steps():
    basis = RotationBasis(binary_indices(MODULUS), MODULUS)
    assert basis.steps(0) == ()
    assert basis.steps(MODULUS) == ()


def test_every_step_is_a_key_that_exists():
    basis = RotationBasis(binary_indices(MODULUS) + [2288], MODULUS)
    keys = set(basis.indices)
    for index in range(MODULUS):
        assert set(basis.steps(index)) <= keys


def test_decomposition_does_not_depend_on_the_order_keys_were_given():
    one = RotationBasis(binary_indices(MODULUS) + [2288, 3], MODULUS)
    other = RotationBasis([3, 2288] + binary_indices(MODULUS), MODULUS)
    assert all(one.steps(i) == other.steps(i) for i in range(MODULUS))


def test_an_index_with_its_own_key_costs_one_rotation():
    basis = RotationBasis(binary_indices(MODULUS) + [2288], MODULUS)
    assert basis.steps(2288) == (2288,)


def test_a_sum_of_two_keys_costs_two_rotations_not_eight():
    basis = RotationBasis(binary_indices(MODULUS) + [2032, 30736], MODULUS)
    index = (2032 + 30736) % MODULUS
    assert len(basis.steps(index)) <= 2


def test_factored_basis_spends_its_keys_on_the_busiest_indices():
    counts = {2288: 127, 2544: 63, 7: 1}
    basis = factored_basis(counts, MODULUS, extra_keys=2)
    extra = set(basis.indices) - set(binary_indices(MODULUS))
    assert extra == {2288, 2544}
    assert rotation_cost(basis, counts) < rotation_cost(
        RotationBasis(binary_indices(MODULUS), MODULUS), counts)


def test_factored_basis_with_no_extra_keys_is_the_binary_basis():
    basis = factored_basis({2288: 127}, MODULUS, extra_keys=0)
    assert set(basis.indices) == set(binary_indices(MODULUS))


def test_key_levels_cover_every_step_at_the_level_that_needs_it():
    basis = RotationBasis(binary_indices(MODULUS) + [2288], MODULUS)
    levels = {2288: 20, 3: 9}
    plan = key_levels(basis, levels)
    for index, level in levels.items():
        for step in basis.steps(index):
            assert plan[step] >= level


@pytest.mark.parametrize("extra", [0, 1, 4, 6, 9])
def test_more_keys_never_cost_more_rotations(extra):
    counts = {2288: 127, 2544: 63, 4320: 63, 3: 40, 12704: 17}
    basis = factored_basis(counts, MODULUS, extra_keys=extra)
    binary = RotationBasis(binary_indices(MODULUS), MODULUS)
    assert rotation_cost(basis, counts) <= rotation_cost(binary, counts)


def test_a_basis_is_always_truthy_even_when_empty():
    """`Stages.rotate` reads a falsy `binary_rotations` as "one key per index, rotate directly"."""
    assert RotationBasis([], MODULUS)
    assert RotationBasis(binary_indices(MODULUS), MODULUS)


# ------------------------------------------------------------------ meeting in the middle
def test_a_sum_of_three_keys_costs_three_rotations_not_fifteen():
    """The reason `max_steps` exists: without it an index three keys away falls back to binary.

    `2288 + 6608 + 14480 = 23376` has seven bits set, so the pairwise basis spends seven rotations
    on an index that is three keys away. That is not a corner case - it is the shape of the layer:
    `attention._accumulate` rotates over a two-dimensional grid, so its own indices are sums of two,
    and everything that lands one key off the grid was paying its popcount.
    """
    keys = binary_indices(MODULUS) + [2288, 6608, 14480]
    index = (2288 + 6608 + 14480) % MODULUS
    assert len(RotationBasis(keys, MODULUS, max_steps=2).steps(index)) == bin(index).count("1") == 7
    assert RotationBasis(keys, MODULUS, max_steps=4).steps(index) == (2288, 6608, 14480)


@pytest.mark.parametrize("max_steps", [2, 3, 4, 5])
def test_deeper_search_still_lands_on_the_index(max_steps):
    basis = RotationBasis(binary_indices(MODULUS) + [2288, 6608, 30720], MODULUS, max_steps)
    for index in range(0, MODULUS, 37):
        steps = basis.steps(index)
        assert sum(steps) % MODULUS == index
        assert all(step in basis.indices for step in steps)


@pytest.mark.parametrize("max_steps", [3, 4, 5, 6])
def test_deeper_search_never_costs_more_than_the_pairwise_one(max_steps):
    keys = binary_indices(MODULUS) + [2288, 6608, 10416, 14480, 30720, 32763]
    shallow = RotationBasis(keys, MODULUS, max_steps=2)
    deep = RotationBasis(keys, MODULUS, max_steps)
    for index in range(0, MODULUS, 53):
        assert len(deep.steps(index)) <= len(shallow.steps(index))


def test_max_steps_below_two_is_still_two():
    """A pair is what a basis reaches directly; `max_steps=0` would mean the binary fallback only."""
    assert RotationBasis([1, 2, 4, 8], 16, max_steps=0).max_steps == 2


def _grid():
    """THOR's own rotation shape in miniature: a two-dimensional grid of indices, equally used."""
    counts = {(2288 * a + 6608 * b) % MODULUS: 10 for a in range(4) for b in range(4)}
    counts.pop(0, None)
    return counts


def test_the_greedy_scores_candidates_the_way_the_run_will_spend_them():
    """Choosing keys under a decomposition the run does not use picks the wrong keys.

    On THOR's own measured demand this is 4237 rotations against 3465 - the same six extra keys'
    worth of memory, a fifth of the rotations, and a different six keys.
    """
    counts = _grid()
    shallow = factored_basis(counts, MODULUS, 2, max_steps=2)
    deep = factored_basis(counts, MODULUS, 2, max_steps=4)
    assert rotation_cost(deep, counts) < rotation_cost(shallow, counts)


# ------------------------------------------------------------------ pricing the keys
def test_basis_bytes_prices_through_the_keys_that_are_used_not_the_keys_that_exist():
    levels = {2288: 30, 6608: 4}
    basis = RotationBasis(binary_indices(MODULUS) + [2288, 6608], MODULUS)
    used = key_levels(basis, levels)
    assert basis_bytes(basis, levels, lambda level: level + 1) == sum(l + 1 for l in used.values())
    # the powers of two that nothing reaches for are not paid for
    assert len(used) < len(basis.indices)


def _mixed():
    """The grid plus scattered indices at scattered levels - the shape the real layer asks for.

    Unlike the bare grid, this one gets *more* expensive as keys are added, which is the direction
    the budget exists for. (The grid alone gets cheaper, see the test below.)
    """
    rng = random.Random(7)
    counts = _grid()
    counts.update({rng.randrange(1, MODULUS): 3 for _ in range(60)})
    counts.pop(0, None)
    return counts, {index: rng.randrange(0, 31) for index in counts}


def test_a_priced_greedy_stays_inside_its_budget():
    counts, levels = _mixed()
    cost = lambda level: level + 1
    floor = basis_bytes(RotationBasis(binary_indices(MODULUS), MODULUS), levels, cost)
    free = basis_bytes(factored_basis(counts, MODULUS, 6, levels=levels, key_cost=cost),
                       levels, cost)
    budget = (floor + free) // 2
    assert floor < budget < free, "the budget has to bite for this to be testing anything"
    capped = factored_basis(counts, MODULUS, 6, levels=levels, key_cost=cost, budget=budget)
    assert basis_bytes(capped, levels, cost) <= budget
    assert rotation_cost(capped, counts) < rotation_cost(
        RotationBasis(binary_indices(MODULUS), MODULUS), counts)


def test_an_extra_key_can_make_the_key_set_smaller():
    """Keys are truncated, so what a key costs depends on the level of whatever reaches for it.

    Every index here is used at level 30, so with the powers of two alone each one drags its whole
    binary decomposition up to 30. Give the grid its own keys and most of those powers stop being
    reached at all - the key set gets *cheaper* as it gets larger, which is why the budget is priced
    through `key_levels` rather than counted in keys.
    """
    counts = _grid()
    levels = dict.fromkeys(counts, 30)
    cost = lambda level: level + 1
    floor = RotationBasis(binary_indices(MODULUS), MODULUS)
    assert (basis_bytes(factored_basis(counts, MODULUS, 6), levels, cost)
            < basis_bytes(floor, levels, cost))


def test_pricing_does_not_change_what_a_basis_reaches():
    counts = _grid()
    levels = dict.fromkeys(counts, 30)
    basis = factored_basis(counts, MODULUS, 4, levels=levels, key_cost=lambda l: l + 1)
    for index in range(0, MODULUS, 61):
        assert sum(basis.steps(index)) % MODULUS == index
