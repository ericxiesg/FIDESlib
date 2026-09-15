"""How a rotation index is reached from the keys we can afford to keep on the card.

A rotation by an arbitrary index needs its own key. One encoder layer asks for 210 distinct indices,
which is 28 GiB of level-truncated keys at depth 37 - on a 32 GB card that leaves nothing for the
bootstrap keys, so it is not an option (:mod:`thorfhe.budget` prices it). The usual escape is to keep
only the powers of two and reach every other index as the sum of its set bits, which is 15 keys and
2.5 GiB, at the cost of about eight rotations per index: 1802 rotations become 8138.

That trade is much better than binary, because the indices this layer asks for are not arbitrary. The
heaviest site is ``attention._accumulate``, whose rotation for diagonal ``in_index`` is

    2048 * j - 16 * in_index,    in_index = pack * block + j

which for ``pack = 16`` is exactly ``2032 * j - 256 * block``: a full two-dimensional grid. Every one
of those 127 indices is a sum of two elements drawn from 16 + 8 values, so two keys' worth of chain
reaches what binary needs eight for. Rather than hard-code that structure - it would silently rot the
day a geometry changes - :func:`factored_basis` *measures* it: it takes the indices a dry run actually
asked for, with how often, and greedily adds the key that removes the most rotations, stopping at a
key count the card has room for.

Six extra keys (21 in total, 3.5 GiB) halve the rotation count. Anything past nine stops fitting.
"""
from __future__ import annotations

import collections


class RotationBasis:
    """The set of rotation keys that will exist, and how each index decomposes into them.

    ``steps`` is the *only* implementation of the decomposition: the key plan is derived from it and
    so is the run, so the two cannot disagree about which keys are needed. That matters because a
    missing key is not a crash - OpenFHE has no index for it and the failure surfaces much later.
    """

    def __init__(self, indices, modulus: int):
        self.modulus = int(modulus)
        #: sorted so `steps` is deterministic: two runs must pick the same decomposition.
        self.indices = tuple(sorted({int(i) % self.modulus for i in indices} - {0}))
        self._set = frozenset(self.indices)
        self._cache: dict[int, tuple[int, ...]] = {}

    def __repr__(self):
        return f"RotationBasis({len(self.indices)} keys, modulus={self.modulus})"

    def __len__(self):
        return len(self.indices)

    def __bool__(self):
        # `Stages.rotate` reads its `binary_rotations` as a flag: falsy means "every index has its
        # own key, rotate directly". A basis is never that, even an empty one - without `__bool__`,
        # `__len__` would make an empty basis silently take the one-key-per-index path and ask the
        # GPU for 210 keys that were never built.
        return True

    def steps(self, index: int) -> tuple[int, ...]:
        """``index`` as a sequence of rotations drawn from this basis, applied left to right.

        One step when the index is itself a key, two when it splits across a pair of keys, and the
        binary decomposition otherwise - which is why the powers of two are always in the basis:
        without them the fallback would need a key that does not exist.
        """
        index = int(index) % self.modulus
        if index == 0:
            return ()
        hit = self._cache.get(index)
        if hit is None:
            self._cache[index] = hit = self._decompose(index)
        return hit

    def _decompose(self, index: int) -> tuple[int, ...]:
        if index in self._set:
            return (index,)
        for first in self.indices:
            second = (index - first) % self.modulus
            if second in self._set:
                return (first, second)
        return tuple(1 << bit for bit in range(index.bit_length()) if index >> bit & 1)


def binary_indices(modulus: int) -> list[int]:
    """The powers of two below ``modulus``: the fallback every basis has to contain."""
    return [1 << bit for bit in range(max(0, (modulus - 1).bit_length()))]


def rotation_cost(basis: RotationBasis, counts) -> int:
    """How many rotations a run costs under ``basis``, given ``{index: times used}``."""
    return sum(times * len(basis.steps(index)) for index, times in counts.items())


def factored_basis(counts, modulus: int, extra_keys: int = 6,
                   candidates=None) -> RotationBasis:
    """The powers of two plus ``extra_keys`` indices chosen to remove the most rotations.

    ``counts`` is ``{rotation index: how many times the layer uses it}`` from a dry run on the clear
    engine. Greedy rather than optimal: the exact problem is a set cover, and the greedy curve is
    already flat by the point the card runs out of room, so the difference cannot be spent.
    """
    if extra_keys <= 0:
        return RotationBasis(binary_indices(modulus), modulus)

    counts = {int(i) % modulus: int(n) for i, n in counts.items() if int(i) % modulus}
    pool = sorted(set(counts) if candidates is None else
                  {int(c) % modulus for c in candidates} | set(counts))

    chosen: list[int] = []
    powers = set(binary_indices(modulus))
    for _ in range(extra_keys):
        best = None
        for candidate in pool:
            if candidate in powers or candidate in chosen:
                continue
            cost = rotation_cost(
                RotationBasis(binary_indices(modulus) + chosen + [candidate], modulus), counts)
            if best is None or cost < best[0]:
                best = (cost, candidate)
        if best is None:
            break
        chosen.append(best[1])
    return RotationBasis(binary_indices(modulus) + chosen, modulus)


def key_levels(basis: RotationBasis, levels) -> dict[int, int]:
    """``{basis key: highest level it is used at}``, ready for ``SetRotationKeyLevels``.

    Both steps of a two-step chain happen at the level of the index that asked for it, so a key's
    level is the highest level of any index whose decomposition uses it.
    """
    plan: dict[int, int] = {}
    for index, level in levels.items():
        for step in basis.steps(index):
            plan[step] = max(plan.get(step, 0), level)
    return plan
