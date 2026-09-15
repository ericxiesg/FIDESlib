"""A rotation basis has to reach every index, and reach it the same way twice.

Both properties are silent when broken. A decomposition whose steps do not sum to the index rotates
the wrong way and decrypts to a plausible-looking wrong answer; a decomposition that depends on
iteration order makes the key plan and the run disagree, and the missing key surfaces later still.
"""
import pytest

from thorfhe.rotation import (RotationBasis, binary_indices, factored_basis, key_levels,
                              rotation_cost)

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
