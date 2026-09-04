"""THOR's feed-forward stages (12 and 14).

The expansion is (4 * features, features), four times too tall for one block diagonal, so THOR splits
it into four pieces and stacks them two at a time. Each ``rep`` therefore carries twelve block rows
where the attention stages carry six - laid out as **two windows of six per token**, at
``FF_SLOT_INDICES`` - and the partial products are recombined with ``block_diag_2``: a window of six on
a stride of eight, rather than twelve on sixteen.

That window is a property of the product, not of the packing, which is why
:meth:`~thorfhe.stages.Stages.pcmm` takes it as an argument rather than reading it off the geometry.
"""
from __future__ import annotations

import numpy as np

from .encoding import block_diagonal_masks
from .geometry import FEEDFORWARD_STRIDE, FEEDFORWARD_WINDOW, Geometry
from .numeric import GeluMixin, NumericMixin
from .stages import Stages


def feedforward_masks(g: Geometry):
    """The ``block_diag_2`` mask family: six blocks wrapping in a stride of eight."""
    return block_diagonal_masks(g, stride=FEEDFORWARD_STRIDE, width=FEEDFORWARD_WINDOW)


def input_mask(g: Geometry) -> np.ndarray:
    """THOR ``masks["intermediate_dense"]``: the six blocks a LayerNorm output occupies."""
    return (np.arange(g.slot_count) % g.n_slot < FEEDFORWARD_WINDOW).astype(float)


class FeedForwardStages(GeluMixin, NumericMixin, Stages):
    """Stages 12, 13 and 14 over the ``Stages`` primitive surface."""

    def prepare_feedforward_input(self, x):
        """Pack the LayerNorm output into ``pack * n`` rotated copies with both windows filled.

        The eight real ciphertexts become four complex ones, masked to the six blocks they occupy and
        then duplicated into the second window (``rotate(temp, stride)``) so a rep's twelve block rows
        have something to multiply. Then the usual rotated copies.
        """
        g = self.g
        half = x.shape[0] // 2
        mask = input_mask(g)

        prepared = np.empty((g.pack * half,), dtype=object)
        for index in range(half):
            merged = self.add(x[index], self.multiply_1j(x[index + half]))
            masked = self.rescale(self.multiply(mask, merged))
            base = self.add(masked, self.rotate(masked, FEEDFORWARD_STRIDE))
            prepared[g.pack * index] = base
            for step in range(1, g.pack):
                prepared[g.pack * index + step] = self.rotate(prepared[g.pack * index + step - 1],
                                                              -g.group_size)
        return prepared

    def stage_12_intermediate_dense(self, x, weight, bias, masks=None):
        """The 4x expansion: two reps of a ``block_diag_2`` product, biased and made real."""
        low, high = feedforward_masks(self.g) if masks is None else masks
        prepared = self.prepare_feedforward_input(x)

        out = np.empty(weight.shape[:2], dtype=object)
        for rep in range(weight.shape[0]):
            products = self.pcmm(weight[rep], prepared, window=FEEDFORWARD_WINDOW,
                                 masks=low, complements=high)
            for column in range(len(products)):
                biased = self.add(products[column], bias[rep][column])
                out[rep, column] = self.add(biased, self.conjugate(biased))
        return out

    def stage_13_gelu(self, x):
        """GELU on the expansion output - eight bootstraps for sixteen ciphertexts.

        The two reps are folded into one complex ciphertext before the bootstrap, halving its cost. The
        halving before it and the doubling implicit in ``temp + conj(temp)`` cancel exactly, so the
        values that come out are the ones that went in; the halving is there for the bootstrap's own
        input bound, not for the arithmetic.
        """
        out = np.empty(x.shape, dtype=object)
        for column in range(x.shape[1]):
            merged = self.add(x[0, column], self.multiply_1j(x[1, column]))
            merged = self.bootstrap(self.rescale(self.multiply(merged, 0.5)))
            conj = self.conjugate(merged)
            out[0, column] = self.gelu(self.add(merged, conj))
            out[1, column] = self.gelu(self.multiply_1j(self.subtract(conj, merged)))
        return out

    def prepare_output_dense_input(self, x, rep):
        """One rep's ``pack * n`` rotated copies. No masking: stage 13 leaves both windows filled."""
        g = self.g
        half = x.shape[1] // 2
        prepared = np.empty((g.pack * half,), dtype=object)
        for index in range(half):
            prepared[g.pack * index] = self.add(x[rep, index], self.multiply_1j(x[rep, index + half]))
            for step in range(1, g.pack):
                prepared[g.pack * index + step] = self.rotate(prepared[g.pack * index + step - 1],
                                                              -g.group_size)
        return prepared

    def stage_14_output_dense(self, x, weight, bias, masks=None):
        """The 4x contraction back to six blocks: two reps of a ``block_diag_2`` product, then folded.

        Each rep contracts two of the four horizontal pieces, one per window, and ``rotate(temp, -8)``
        sums the windows. That rotation pulls the *next* token's low window into slots 8..13, so the
        output is only meaningful on slots 0..5 - which is all LayerNorm reads, since its ``value_mask``
        zeroes the rest before anything else happens. THOR builds a mask here for the same purpose and
        then never applies it; this is why it can get away with that.
        """
        low, high = feedforward_masks(self.g) if masks is None else masks

        wx = np.empty(weight.shape[:2], dtype=object)
        for rep in range(weight.shape[0]):
            wx[rep] = self.pcmm(weight[rep], self.prepare_output_dense_input(x, rep),
                                window=FEEDFORWARD_WINDOW, masks=low, complements=high)

        out = np.empty((wx.shape[1],), dtype=object)
        for column in range(wx.shape[1]):
            temp = self.add(wx[0, column], wx[1, column])
            temp = self.add(temp, self.rotate(temp, -FEEDFORWARD_STRIDE))
            temp = self.add(temp, bias[column])
            out[column] = self.add(temp, self.conjugate(temp))
        return out
