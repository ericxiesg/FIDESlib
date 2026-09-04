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
from .stages import Stages


def feedforward_masks(g: Geometry):
    """The ``block_diag_2`` mask family: six blocks wrapping in a stride of eight."""
    return block_diagonal_masks(g, stride=FEEDFORWARD_STRIDE, width=FEEDFORWARD_WINDOW)


def input_mask(g: Geometry) -> np.ndarray:
    """THOR ``masks["intermediate_dense"]``: the six blocks a LayerNorm output occupies."""
    return (np.arange(g.slot_count) % g.n_slot < FEEDFORWARD_WINDOW).astype(float)


class FeedForwardStages(Stages):
    """Stage 12 over the ``Stages`` primitive surface."""

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
