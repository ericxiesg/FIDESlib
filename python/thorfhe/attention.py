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
                 transpose=None, copies=None, attention=None):
        super().__init__(engine, geometry, masks=masks, complement_masks=complement_masks)
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

    def interval_sum(self, x, interval: int):
        """Fold the slot vector onto itself in steps of ``interval``, so every window holds the total."""
        out = x
        for step in range(int(np.log2(self.g.slot_count / interval))):
            out = self.add(out, self.rotate(out, -interval * 2 ** step))
        return out

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
        g = self.g
        low, high, chunks = self.copies
        n = x.shape[0]
        copies = np.empty((g.pack * n,), dtype=object)

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
                    real = self.add(half, conjugated)
                    imag = self.multiply_1j(self.subtract(conjugated, half))
                    copies[index * g.pack + 2 * chunk_index + offset] = real
                    copies[(index + n // 2) * g.pack + 2 * chunk_index + offset] = imag
        return copies
