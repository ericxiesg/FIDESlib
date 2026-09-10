"""THOR's attention plumbing: the transpose, the diagonal broadcast, and their masks.

These are the data-movement halves of stage 06. Unlike stages 01-05 they are written for THOR's shape
only - four output ciphertexts of sixteen groups - because that is how ``he.py`` writes them: the mask
families and the accumulator permutations are literal tables, not formulas, and generalising them is
guesswork rather than transcription. :func:`require_thor_shape` says so out loud instead of quietly
computing something wrong at another geometry.

Both operations are exact permutations of the input (up to one deliberate factor of a half), which is
what makes them cheap to test: :mod:`thorfhe.attention` states what each one does as a formula and the
tests check that formula slot by slot, rather than eyeballing an end-to-end error.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .stages import Stages


def require_thor_shape(g: Geometry) -> None:
    """These stages are transcriptions of tables written for four ciphertexts of ``pack`` groups."""
    if g.n_output_ciphertexts != 4:
        raise NotImplementedError(
            f"the attention stages are transcribed from he.py for 4 output ciphertexts, not "
            f"{g.n_output_ciphertexts}; the mask families and accumulator permutations are literal "
            "tables in THOR and have no derived form yet")


def _pad(g: Geometry, values) -> np.ndarray:
    """THOR builds masks as short lists and lets the encoder zero-pad them to the slot count."""
    out = np.zeros((g.slot_count,))
    out[: len(values)] = values
    return out


def transpose_masks(g: Geometry):
    """THOR ``pre_encode_masks``, the ``transpose`` family: ``(mask0, mask1, mask2, mask3)``."""
    require_thor_shape(g)
    pack, n_out, dim, n_slot = g.pack, g.n_out, g.dim, g.n_slot
    m0, m1, m2, m3 = {}, {}, {}, {}
    for index in range(4):
        n = (index * pack - pack) % n_out + pack
        m0[index] = _pad(g, [1] * (n_slot * (n_out + n)))
        m1[index] = _pad(g, [0] * (g.slot_count - n_slot * (n_out - n)) + [1] * (n_slot * (n_out - n)))
        for upper_diag in range(index * pack + 1, (index + 1) * pack):
            lower_diag = n_out - upper_diag
            m2[upper_diag] = _pad(g, [0] * (g.group_size * (pack - upper_diag % pack))
                                  + [1] * ((dim - lower_diag) * n_slot))
            m3[upper_diag] = _pad(g, [0] * (g.group_size * (pack - upper_diag % pack - 1))
                                  + [0] * ((dim - lower_diag) * n_slot)
                                  + [1] * (n_slot * lower_diag))
    return m0, m1, m2, m3


def make_copies_masks(g: Geometry):
    """THOR's ``make_copies`` masks: the two half-group selectors and the ``pack // 2`` chunk masks.

    The chunks carry a factor of 1/4 that, with the ``y + conj(y)`` at the end, leaves the broadcast
    diagonals at half the input - see :func:`AttentionStages.make_copies`.
    """
    require_thor_shape(g)
    slot = np.arange(g.slot_count)
    low = ((slot % (2 * g.group_size)) < g.group_size).astype(float)
    high = ((slot % (2 * g.group_size)) >= g.group_size).astype(float)
    chunks = {i: ((slot // (2 * g.group_size)) == i).astype(float) * 0.25
              for i in range(g.slot_count // (2 * g.group_size))}
    return low, high, chunks


def attention_rotate_masks(g: Geometry) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """``rotate_internal`` masks for the ``attention`` mode: the window is a whole group, not a token."""
    within = np.arange(g.slot_count) % g.group_size
    low = {d: (within < g.n_slot * d).astype(float) for d in range(1, g.dim)}
    high = {d: (within >= g.n_slot * d).astype(float) for d in range(1, g.dim)}
    return low, high


class AttentionStages(Stages):
    """Stages 01-05 plus the attention data movement of stage 06.

    ``masks``/``complement_masks`` are the block-diagonal family inherited from :class:`Stages`; the
    attention families are passed separately because they select on a different window.
    """

    def __init__(self, engine, geometry: Geometry, masks=None, complement_masks=None,
                 transpose=None, copies=None, attention=None, **kwargs):
        super().__init__(engine, geometry, masks=masks, complement_masks=complement_masks, **kwargs)
        require_thor_shape(geometry)
        #: ``(mask0, mask1, mask2, mask3)`` from :func:`transpose_masks`.
        self.transpose = transpose
        #: ``(low, high, chunks)`` from :func:`make_copies_masks`.
        self.copies = copies
        #: ``(low, high)`` from :func:`attention_rotate_masks`.
        self.attention = attention

    # ---------------------------------------------------------------- helpers
    def rotate_internal_attention(self, x, delta: int):
        """``rotate_internal`` with the window a whole group: token ``t`` moves to ``t + delta``.

        Same shape as :meth:`Stages.rotate_internal` - mask and complement, then one rescale - but the
        wrap is at ``group_size`` rather than at the ``n_blocks`` used slots of one token.
        """
        low, high = self.attention
        left = self.g.n_slot * delta
        right = self.g.group_size - left
        rotated = self.add(self.rotate(self.multiply(high[delta], x), -left),
                           self.rotate(self.multiply(low[delta], x), right))
        return self.rescale(rotated)

    # ---------------------------------------------------------------- stage 06 data movement
    def transpose_upper_to_lower(self, upper):
        """Transpose the packed per-head matrices, so a ciphertext-ciphertext product can form ``Q K^T``.

        Exactly a permutation of the entries: with ``d = f % n_out`` and ``b = f // n_out``,

            ``out[ct][group, tau, b] = k[t, f]``
            ``ct * pack + group = (t - d) mod n_out``
            ``tau = d + n_out * ((d > t mod n_out) XOR (t // n_out))``

        The result is canonical (rescaled), which ``he.py`` leaves to desilofhe.
        """
        g = self.g
        mask0, mask1, mask2, mask3 = self.transpose
        lower_temp = np.empty((4, 2), dtype=object)

        for index in range(4):
            rotated = self.rotate(upper[index], -(((g.n_out - g.pack * index) % g.n_out) * g.n_slot))
            prepared = self.prepare_for_multiply(rotated)
            lower_temp[(4 - index) % 4][0] = self.multiply(mask0[index], prepared)
            lower_temp[(4 - index) % 4][1] = self.multiply(mask1[index], prepared)

        for index in range(4):
            for upper_diag in range(g.pack * index + 1, g.pack * (index + 1)):
                lower_diag = g.n_out - upper_diag
                delta = -(lower_diag * g.n_slot
                          + (((upper_diag % (3 * g.pack)) * 2) % g.pack) * g.group_size)
                prepared = self.prepare_for_multiply(self.rotate(upper[index], delta))
                self.add_inplace(lower_temp[(3 - index) % 4][0], self.multiply(mask2[upper_diag], prepared))
                self.add_inplace(lower_temp[(3 - index) % 4][1], self.multiply(mask3[upper_diag], prepared))

        lower = np.empty((4,), dtype=object)
        for index in range(4):
            merged = self.add(lower_temp[index][0], self.rotate(lower_temp[index][1], g.group_size))
            lower[index] = self.rescale(merged)
        return lower

    def make_copies(self, x):
        """Broadcast every diagonal of ``x`` across all groups, halved.

        Exactly ``copies[l][group, t, b] = x[t, n_out * b + (l + t) mod n_out] / 2`` for every group,
        so ``copies[l]`` is the ciphertext a ciphertext-ciphertext product needs for input diagonal
        ``l``. Costs two levels: the chunk mask and the half-group mask.
        """
        copies = np.empty((self.g.pack * x.shape[0],), dtype=object)
        for position, copy in self.iter_copies(x):
            copies[position] = copy
        return copies

    def iter_copies(self, x):
        """The same diagonals as :meth:`make_copies`, yielded as ``(position, ciphertext)`` as they
        are built rather than returned as a filled array.

        At THOR's geometry that array is 64 ciphertexts near full level - about 1.8 GiB at depth 37,
        and over half of the layer's largest measured working set. They are built all at once and then
        used one at a time, so holding the array is the cost, not the arithmetic. A consumer that takes
        them in the order they are produced holds a handful instead.

        Each round of the outer loop produces two blocks, ``index`` and ``index + n // 2``, because the
        real and imaginary halves of one complex ciphertext are two different diagonals. That is why
        this yields positions rather than a flat sequence: the production order is not the natural
        order, and it is the consumer that has to bend.
        """
        g = self.g
        low, high, chunks = self.copies
        n = x.shape[0]

        for index in range(n // 2):
            merged = self.add(x[index], self.multiply_1j(x[index + n // 2]))
            prepared = self.prepare_for_multiply(merged)

            for chunk_index in range(len(chunks)):
                selected = self.rescale(self.multiply(chunks[chunk_index], prepared))
                spread = self.interval_sum(selected, 2 * g.group_size)

                first = self.rescale(self.multiply(low, spread))
                second = self.rescale(self.multiply(high, spread))
                self.add_inplace(first, self.rotate(first, g.group_size))
                self.add_inplace(second, self.rotate(second, -g.group_size))

                for offset, half in ((0, first), (1, second)):
                    conjugated = self.conjugate(half)
                    yield (index * g.pack + 2 * chunk_index + offset,
                           self.add(half, conjugated))
                    yield ((index + n // 2) * g.pack + 2 * chunk_index + offset,
                           self.multiply_1j(self.subtract(conjugated, half)))


def ccmm_masks(g: Geometry) -> dict[int, dict[int, np.ndarray]]:
    """THOR ``pre_encode_masks``, the ``ccmm`` family: how a rotated product is split four ways.

    Mask 1 is the complement of mask 0 when ``n`` is a multiple of ``pack``, and mask 3 is literally
    ``1 - m0 - m1 - m2`` otherwise, so the four always partition the slot vector. ``he.py`` forms the
    last piece by subtracting the others from the un-rescaled product instead; that mixes two levels,
    so the port multiplies by the mask THOR already stores.
    """
    require_thor_shape(g)
    slot = np.arange(g.slot_count)
    masks: dict[int, dict[int, np.ndarray]] = {0: {}, 1: {}, 2: {}, 3: {}}
    for n in range(1, g.dim):
        j = n % g.pack
        high = (slot % g.group_size) >= (g.group_size - g.n_slot * n)
        m0 = np.ones(g.slot_count)
        m0[high] = 0
        m1 = np.zeros(g.slot_count)
        m1[high] = 1
        if j == 0:
            masks[0][n], masks[1][n] = m0, m1
            continue
        m0[: g.group_size * j] = 0
        m1[-g.group_size:] = 0
        if j > 1:
            m1[: g.group_size * (j - 1)] = 0
        m2 = np.ones(g.slot_count)
        m2[high] = 0
        m2[g.group_size * j:] = 0
        masks[0][n], masks[1][n], masks[2][n] = m0, m1, m2
        masks[3][n] = np.ones(g.slot_count) - m0 - m1 - m2
    return masks


class AttentionScore(AttentionStages):
    """Stage 06: the ciphertext-ciphertext product that forms the attention scores.

    Computes, exactly, ``S[b] = Q_b K_b^T`` for each of the ``n_blocks`` heads, packed by diagonals::

        out[ct][group, tau, b] = S[b][tau, (ct * pack + group + tau) mod dim]

    over ``2 * n_output_ciphertexts`` ciphertexts, since the score matrix is ``dim x dim`` per head
    while the projections were ``dim x n_out``.

    **This diverges from he.py, deliberately.** ``he.py``'s accumulator table for the
    ``in_index % pack == 0`` case sends the contribution to columns 2-3 when ``i == 0`` and to columns
    0-1 otherwise. The rule its own ``in_index % pack != 0`` branch uses - columns 2-3 exactly when the
    source index wrapped, ``i - in_index // pack < 0`` - is the one that makes the result equal
    ``Q K^T``; the two agree for ``in_index // pack == 1`` and disagree for 2 and 3, which corrupts two
    of the sixty-four inner-product terms in half the output ciphertexts. See docs/thor_port.md.
    """

    def __init__(self, engine, geometry: Geometry, ccmm=None, **kwargs):
        super().__init__(engine, geometry, **kwargs)
        #: ``{column: {n: mask}}`` from :func:`ccmm_masks`.
        self.ccmm = ccmm

    def accumulator_column(self, out_index: int, offset: int, j: int, out_dim: int) -> int:
        """Which pair of accumulator columns a contribution lands in.

        Columns 2-3 when the source index wrapped an odd number of times, 0-1 otherwise. Stage 08's
        tables are what show this is the general rule: there ``out_dim`` is 2 and offsets reach -4, so
        double wraps occur and land back in columns 0-1 - which those tables do, in every entry.

        ``he.py`` uses this rule everywhere except the ``j == 0`` branch of the *score*, where it keys
        on ``out_index == 0``; that agrees only when ``in_index // pack == 1``. Overridable so a test
        can pin the difference down.
        """
        return 2 * ((offset // out_dim) % 2)

    def _key_as_complex(self, k):
        """Fold the transposed key and its group-rotated copy into one complex ciphertext."""
        g = self.g
        lower = self.transpose_upper_to_lower(k)
        out = np.empty((4,), dtype=object)
        for index in range(4):
            rotated = self.rotate_internal_attention(lower[index], g.n_out)
            # rotate_internal_attention spent a level, so the un-rotated half has to follow it down.
            out[index] = self.add(self.level_down(lower[index], 1), self.multiply_1j(rotated))
        return out

    def _accumulate_product(self, left, diagonals, in_dim: int, *, consume: bool = False):
        """The shared body of both ciphertext-ciphertext products.

        ``left`` is the complex left operand - one ciphertext per output - and ``diagonals`` is
        ``in_dim`` broadcast diagonals.

        ``consume`` says the caller will not look at ``diagonals`` again, so each one can be dropped
        the moment its iteration is over. At THOR's geometry there are 64 of them, about 38 MiB each at
        depth 37, so holding the lot is 2.4 GiB against the 3.7 GiB a GV100 has left after key
        generation. Dropping one does not hand memory back to the driver - the destructor parks its
        polynomials in the context's auxiliary pool - but the next ciphertext then *reuses* them
        instead of asking the allocator for more, which is what stops the high-water mark climbing
        through the loop. Returns the ``len(left) x 4`` accumulator, still degree-2 and
        at scale Delta^2. The score product has four outputs and the context product two, which is why
        the width is taken from the operand rather than fixed on the class.
        """
        # The two operands reach this point by very different routes - the values come straight from the
        # projection, the weights through a softmax - so they have to be brought to a common level.
        target = min(min(self.engine.level(ct) for ct in left),
                     min(self.engine.level(ct) for ct in diagonals))

        def in_order():
            for in_index in range(in_dim):
                yield in_index, diagonals[in_index]
                # last use of this diagonal: let it go, so the next iteration's ciphertexts
                # come out of the auxiliary pool rather than off the top of the heap
                if consume:
                    diagonals[in_index] = None

        return self._accumulate(left, in_order(), target)

    def _accumulate_streamed(self, left, produce):
        """:meth:`_accumulate_product` against a generator of ``(in_index, ciphertext)`` pairs.

        The loop below only ever accumulates, so the order the diagonals arrive in does not matter -
        which means a producer does not have to build them all before the first one can be used. That
        is worth about 1.8 GiB at stage 06, where :meth:`iter_copies` produces 64 near-full-level
        ciphertexts in an order that is not the order they are consumed in.

        The one thing the streamed form cannot do is scan every diagonal for the common level before
        starting, so it takes the first one's and holds the rest to it. Every producer here builds its
        diagonals by one uniform path, so that is already true; it is checked rather than assumed
        because a violation would otherwise surface as a silent FIXEDMANUAL scale mismatch much later.
        """
        stream = iter(produce)
        try:
            first = next(stream)
        except StopIteration:
            raise ValueError("_accumulate_streamed needs at least one diagonal") from None

        level = self.engine.level(first[1])
        target = min(min(self.engine.level(ct) for ct in left), level)

        def checked():
            yield first
            for in_index, diagonal in stream:
                if self.engine.level(diagonal) != level:
                    raise ValueError(
                        f"streamed diagonal {in_index} is at level {self.engine.level(diagonal)}, "
                        f"not {level} like the first: a streamed product cannot align to a level it "
                        "has not seen yet")
                yield in_index, diagonal

        return self._accumulate(left, checked(), target)

    def _accumulate(self, left, stream, target):
        """The accumulation loop itself, over ``(in_index, diagonal)`` pairs in any order."""
        g = self.g
        out_dim = len(left)

        def aligned(ct):
            surplus = self.engine.level(ct) - target
            return self.level_down(ct, surplus) if surplus > 0 else ct

        # `left` is two or four ciphertexts, so align it once. The diagonals are `in_dim` of them - 128
        # at THOR's geometry - and levelling those eagerly holds a second copy of all of them while the
        # caller still holds the originals: about 5 GiB at depth 37, which is what a GV100 runs out of
        # here. Each diagonal is used in exactly one iteration below, so align it there instead and let
        # it go, which keeps one alive rather than 128.
        left = [aligned(ct) for ct in left]

        accumulator = np.full((out_dim, 4), None, dtype=object)

        def accumulate(index, value):
            if accumulator[index] is None:
                accumulator[index] = value
            else:
                self.add_inplace(accumulator[index], value)

        for in_index, raw in stream:
            # in_index 0 needs no rotation and no split; level_down aligns it with the rescaled rest,
            # so its term lands at the same level as everyone else's and may arrive at any point.
            if in_index == 0:
                zeroth = aligned(raw)
                for i in range(out_dim):
                    accumulate((i, 0), self.level_down(self.multiply(left[i], zeroth), by=1))
                continue

            block, j = divmod(in_index, g.pack)
            rotation = g.group_size * j - g.n_slot * in_index

            pieces = np.full((out_dim, 4), None, dtype=object)
            diagonal = aligned(raw)
            for i in range(out_dim):
                product = self.multiply(self.rotate(left[i], rotation), diagonal)
                rescaled = self.rescale(product)
                for column in ((0, 1) if j == 0 else (0, 1, 2, 3)):
                    pieces[i, column] = self.multiply(self.ccmm[column][in_index], rescaled)

            for i in range(out_dim):
                sources = [(i - block, 0)] if j == 0 else [(i - 1 - block, 2), (i - block, 0)]
                for offset, first_column in sources:
                    column = self.accumulator_column(i, offset, j, out_dim)
                    accumulate((i, column), pieces[offset % out_dim, first_column])
                    accumulate((i, column + 1), pieces[offset % out_dim, first_column + 1])
        return accumulator

    def _fold_accumulator(self, accumulator):
        """Relinearise, realign the halves and fold the conjugate pair into one complex ciphertext."""
        g = self.g
        out_dim = accumulator.shape[0]
        merged = np.empty((out_dim,), dtype=object)
        for i in range(out_dim):
            parts = [self.relinearize(accumulator[i, c]) for c in range(4)]
            parts[1] = self.rotate(parts[1], g.group_size)
            parts[3] = self.rotate(parts[3], g.group_size)
            parts[2] = self.multiply_1j(self.conjugate(self.add(parts[2], parts[3])))
            merged[i] = self.add(self.add(parts[0], parts[1]), parts[2])
        return merged

    def stage_06_attention_score(self, q, k):
        # make_copies' output is built here and read nowhere else, so it never has to exist as an
        # array: streaming it costs one diagonal alive instead of 64.
        accumulator = self._accumulate_streamed(self._key_as_complex(k), self.iter_copies(q))
        merged = self._fold_accumulator(accumulator)

        half = len(merged)
        output = np.empty((2 * half,), dtype=object)
        for i in range(half):
            conjugated = self.conjugate(merged[i])
            output[i] = self.rescale(self.add(merged[i], conjugated))
            output[i + half] = self.rescale(self.multiply_1j(self.subtract(conjugated, merged[i])))
        return output


#: how many ciphertexts the attention *context* product spans, against the score product's four.
CONTEXT_OUTPUTS = 2


class AttentionContext(AttentionScore):
    """Stage 08: the second ciphertext-ciphertext product, attention weights times values.

    Same machinery as the score, with the operands swapped in shape: ``out_dim`` is 2 rather than 4 and
    the inner dimension is ``dim`` (one term per key token) rather than ``n_out``. Computes

        ``context[b][tau, d] = sum_j A[b][tau, j] * V[j, n_out*b + d]``

    packed the way the QKV projections were, so stage 10's dense layer can consume it directly.

    ``weights`` is ``dim`` broadcast diagonals of the per-head attention matrix - what
    ``he_softmax`` produces, and the same shape :meth:`make_copies` produces for the score.
    """

    def _values_as_complex(self, v):
        """Pack the four real value ciphertexts into two complex ones."""
        return np.array([self.add(v[i], self.multiply_1j(v[i + CONTEXT_OUTPUTS]))
                         for i in range(CONTEXT_OUTPUTS)], dtype=object)

    def stage_08_attention_context(self, v, weights, *, consume: bool = False):
        """``consume`` releases each softmax diagonal as it is used - see :meth:`_accumulate_product`.

        There are ``dim`` of them, twice what stage 06 holds and the largest single working set in the
        layer, so it is worth taking when the caller has no further use for them. It is off by default
        because the obvious other reader is a trace: ``--per-stage`` decrypts the softmax output after
        this returns.
        """
        accumulator = self._accumulate_product(self._values_as_complex(v), weights, self.g.dim,
                                               consume=consume)
        merged = self._fold_accumulator(accumulator)
        return np.array([self.rescale(m) for m in merged], dtype=object)
