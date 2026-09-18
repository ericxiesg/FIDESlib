"""How a rotation index is reached from the keys we can afford to keep on the card.

A rotation by an arbitrary index needs its own key. One encoder layer asks for 210 distinct indices,
which is 28 GiB of level-truncated keys at depth 37 - on a 32 GB card that leaves nothing for the
bootstrap keys, so it is not an option (:mod:`thorfhe.budget` prices it). The usual escape is to keep
only the powers of two and reach every other index as the sum of its set bits, which is 15 keys and
2.5 GiB, at the cost of about four rotations per index: 1922 rotations become 8258.

That trade is much better than binary, because the indices this layer asks for are not arbitrary. The
heaviest site is ``attention._accumulate``, whose rotation for diagonal ``in_index`` is

    2048 * j - 16 * in_index,    in_index = pack * block + j

which for ``pack = 16`` is exactly ``2032 * j - 256 * block``: a full two-dimensional grid. Every one
of those 127 indices is a sum of two elements drawn from 16 + 8 values, so two keys' worth of chain
reaches what binary needs eight for. Rather than hard-code that structure - it would silently rot the
day a geometry changes - :func:`factored_basis` *measures* it: it takes the indices a dry run actually
asked for, with how often, and greedily adds the key that removes the most rotations, stopping at a
key count the card has room for.

Two things matter as much as which keys are kept.

**How far an index may be reached.** Reaching an index by a *pair* of keys covers the grid but not
what sits beside it, and everything else fell back to its binary expansion - seven or ten rotations
for an index three keys away. ``max_steps`` lets :class:`RotationBasis` meet in the middle instead,
and the greedy is scored through the same decomposition the run will spend, so it picks different
keys. On THOR's measured demand, with the same six extra keys and the same 3.5 GiB:

    max_steps=2   4237 rotations     max_steps=4   3465     max_steps=6   3461

Four is the default. Six is worth four rotations and costs four times the planning.

**What a key costs.** Keys are level-truncated, so a key reached for at level 30 is several times the
size of one reached for at level 3 - and because truncation is driven by :func:`key_levels`, adding a
key can make the key set *smaller*, by taking an index off a chain of powers of two it was dragging
up to its own level. Given ``levels`` and a ``key_cost``, the greedy ranks candidates by rotations
removed per byte and honours a ``budget``; on THOR that buys 0.09 GiB for about 180 rotations, which
is the trade to make only when the card is the thing that does not fit.

Measured at THOR's geometry, depth 37, one layer, ``--compact``:

    one key per index   210 keys   28 GiB    1922 rotations
    binary               15 keys   2.5 GiB   8258
    binary + 6           21 keys   3.5 GiB   3465
    binary + 9           24 keys   3.9 GiB   3153

Anything past nine stops fitting.
"""
from __future__ import annotations

import collections


class RotationBasis:
    """The set of rotation keys that will exist, and how each index decomposes into them.

    ``steps`` is the *only* implementation of the decomposition: the key plan is derived from it and
    so is the run, so the two cannot disagree about which keys are needed. That matters because a
    missing key is not a crash - OpenFHE has no index for it and the failure surfaces much later.
    """

    def __init__(self, indices, modulus: int, max_steps: int = 4):
        self.modulus = int(modulus)
        #: how many keys a single index may be reached through before the binary fallback takes over.
        #: Two is what a pair of keys covers directly; above that the chain is found by meeting in the
        #: middle, which costs nothing at run time (`steps` is cached) and is where most of the saving
        #: is: the grid the attention accumulates over is two-dimensional, so its indices are sums of
        #: two, but the *residues* around it are not, and those were falling all the way back to their
        #: binary decomposition - fifteen rotations for an index four keys away.
        self.max_steps = max(2, int(max_steps))
        #: sorted so `steps` is deterministic: two runs must pick the same decomposition.
        self.indices = tuple(sorted({int(i) % self.modulus for i in indices} - {0}))
        self._set = frozenset(self.indices)
        self._cache: dict[int, tuple[int, ...]] = {}
        self._reach: dict[int, tuple[int, ...]] | None = None

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
        binary = tuple(1 << bit for bit in range(index.bit_length()) if index >> bit & 1)
        if self.max_steps <= 2:
            return binary
        deep = self._meet(index)
        return binary if deep is None or len(deep) >= len(binary) else deep

    def _half_reach(self) -> dict[int, tuple[int, ...]]:
        """``{value: shortest chain reaching it}`` for chains of at most half ``max_steps`` steps.

        Half, because :meth:`_meet` puts two of these together - the table is what makes the search
        affordable, and it is built once per basis rather than once per index.
        """
        if self._reach is None:
            reach: dict[int, tuple[int, ...]] = {0: ()}
            frontier = dict(reach)
            for _ in range((self.max_steps + 1) // 2):
                nxt = {}
                for value, chain in frontier.items():
                    for step in self.indices:
                        moved = (value + step) % self.modulus
                        if moved not in reach:
                            nxt[moved] = reach[moved] = chain + (step,)
                frontier = nxt
            self._reach = reach
        return self._reach

    def _meet(self, index: int) -> tuple[int, ...] | None:
        reach = self._half_reach()
        best = None
        for value, chain in reach.items():
            tail = reach.get((index - value) % self.modulus)
            if tail is not None and (best is None or len(chain) + len(tail) < len(best)):
                best = chain + tail
        return best


def binary_indices(modulus: int) -> list[int]:
    """The powers of two below ``modulus``: the fallback every basis has to contain."""
    return [1 << bit for bit in range(max(0, (modulus - 1).bit_length()))]


def rotation_cost(basis: RotationBasis, counts) -> int:
    """How many rotations a run costs under ``basis``, given ``{index: times used}``."""
    return sum(times * len(basis.steps(index)) for index, times in counts.items())


def _with(basis: RotationBasis, extra) -> RotationBasis:
    return RotationBasis(list(basis.indices) + list(extra), basis.modulus, basis.max_steps)


def basis_bytes(basis: RotationBasis, levels, key_cost) -> int:
    """What ``basis`` costs to hold, given ``{index: highest level}`` and a per-level price.

    Priced through :func:`key_levels`, not through ``basis.indices``, because the keys are truncated:
    a key is only as large as the highest level any index that uses it is used at. That is also why
    adding a key can make the key set *smaller*. An index at level 30 that falls back to its binary
    decomposition drags all eight of its powers of two up to level 30; give it a pair of keys of its
    own and those powers keep whatever level the rest of the layer asked of them.
    """
    return sum(key_cost(level) for level in key_levels(basis, levels).values())


def factored_basis(counts, modulus: int, extra_keys: int = 6,
                   candidates=None, *, levels=None, key_cost=None, budget=None,
                   max_steps: int = 4) -> RotationBasis:
    """The powers of two plus up to ``extra_keys`` indices chosen to remove the most rotations.

    ``counts`` is ``{rotation index: how many times the layer uses it}`` from a dry run on the clear
    engine. Greedy rather than optimal: the exact problem is a set cover, and the greedy curve is
    already flat by the point the card runs out of room, so the difference cannot be spent.

    Given ``levels`` (``{index: highest level}``) and ``key_cost`` (``level -> bytes``) the choice
    becomes a budget one: candidates are ranked by rotations removed **per byte added** rather than by
    rotations removed, a candidate that pays for itself in truncation is taken ahead of any that does
    not, and ``budget`` caps the bytes the key set is allowed to reach. Ranking by rotations alone
    spends the whole allowance on whichever key happens to be used at depth, when two keys used below
    the bootstrap level can cost less together and remove more.
    """
    if extra_keys <= 0:
        return RotationBasis(binary_indices(modulus), modulus, max_steps)

    counts = {int(i) % modulus: int(n) for i, n in counts.items() if int(i) % modulus}
    pool = sorted(set(counts) if candidates is None else
                  {int(c) % modulus for c in candidates} | set(counts))
    priced = levels is not None and key_cost is not None
    if priced:
        levels = {int(i) % modulus: int(l) for i, l in levels.items() if int(i) % modulus}

    def build(chosen):
        return RotationBasis(binary_indices(modulus) + list(chosen), modulus, max_steps)

    chosen: list[int] = []
    powers = set(binary_indices(modulus))
    current = build(chosen)
    cost = rotation_cost(current, counts)
    size = basis_bytes(current, levels, key_cost) if priced else 0
    for _ in range(extra_keys):
        best = None
        for candidate in pool:
            if candidate in powers or candidate in chosen:
                continue
            trial = build(chosen + [candidate])
            saved = cost - rotation_cost(trial, counts)
            if saved <= 0:
                continue
            if not priced:
                score = (saved, 0)
            else:
                grown = basis_bytes(trial, levels, key_cost)
                if budget is not None and grown > budget:
                    continue
                added = grown - size
                # A key that shrinks the set is strictly better than any key that grows it, however
                # much the latter removes: it wins on both axes. Rank those among themselves by
                # rotations removed, and everything else by rotations removed per byte.
                score = ((1, saved) if added <= 0 else (0, saved / added))
            if best is None or score > best[0]:
                best = (score, candidate, trial)
        if best is None:
            break
        chosen.append(best[1])
        current = best[2]
        cost = rotation_cost(current, counts)
        if priced:
            size = basis_bytes(current, levels, key_cost)
    return current


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
