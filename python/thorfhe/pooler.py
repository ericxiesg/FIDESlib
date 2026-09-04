"""THOR's classification head: the pooler (stage 17) and the classifier (stage 18).

Both work on the CLS token alone, which is what makes them look unlike every other stage. The pooler
masks token 0 out of the last LayerNorm output, broadcasts it back across all 128 tokens, and then
multiplies it by a (768, 768) weight - so there is no rotated-copy dimension over ``pack``, because
every group already holds the same token. ``encode_weight_pooler`` folds the pack offset into the
*input* index instead, giving a (6, 4) weight where an ordinary dense layer would need (8, 6, 64).

The classifier is a single inner product per class, done entirely with rotations: six slots folded by
1, 2 and 4, then 128 tokens folded by 16 through 1024. The ``1, 2, 4`` step is deliberately not a
plain doubling chain - it sums six slots, not eight.
"""
from __future__ import annotations

import numpy as np

from .encoding import block_diagonal_masks, pooler_mask
from .geometry import FEEDFORWARD_STRIDE, FEEDFORWARD_WINDOW, Geometry
from .numeric import NumericMixin
from .stages import Stages


class PoolerStages(NumericMixin, Stages):
    """Stages 17 and 18 over the ``Stages`` primitive surface."""

    def broadcast_cls_token(self, x):
        """Mask token 0's six blocks and copy them across all ``dim`` tokens of every group."""
        g = self.g
        out = self.rescale(self.multiply(pooler_mask(g), x))
        for step in range(int(np.log2(g.dim))):
            out = self.add(out, self.rotate(out, g.n_slot * 2 ** step))
        return out

    def pooler_dense(self, x, weight, bias, masks=None):
        """The (768, 768) pooler weight against the broadcast CLS token.

        THOR calls the mask family ``block_diag_2`` - the same window of six on a stride of eight the
        feed-forward stages use - and closes with an ``interval_sum`` over the groups, because here the
        groups carry parts of one inner product rather than independent output diagonals.
        """
        low, high = (block_diagonal_masks(self.g, stride=FEEDFORWARD_STRIDE,
                                          width=FEEDFORWARD_WINDOW) if masks is None else masks)
        prepared = [self.prepare_for_multiply(self.broadcast_cls_token(ct)) for ct in x]

        rows = weight.shape[0]
        partial = np.empty((rows,), dtype=object)
        for row in range(rows):
            total = self.multiply(weight[row, 0], prepared[0])
            for column in range(1, weight.shape[1]):
                total = self.add(total, self.multiply(weight[row, column], prepared[column]))
            partial[row] = self.rescale(total)

        # the un-rotated term skips a rotate_internal, so it has to be levelled down by hand
        temp = self.level_down(partial[0], by=1)
        for row in range(1, rows):
            temp = self.add(temp, self.rotate_internal(partial[row], rows - row,
                                                       window=FEEDFORWARD_WINDOW,
                                                       masks=low, complements=high))
        temp = self.interval_sum(temp, self.g.group_size)
        temp = self.add(temp, bias)
        return np.array([self.add(temp, self.conjugate(temp))], dtype=object)

    def stage_17_pooler(self, x, weight, bias, masks=None):
        """``tanh(cls @ W.T + b)``, with the ``/40`` that puts the tanh argument inside its fit."""
        half = len(x) // 2
        merged = [self.add(x[index], self.multiply_1j(x[index + half])) for index in range(half)]
        dense = self.pooler_dense(merged, weight, bias, masks=masks)
        return np.array([self.he_tanh_for_pooler(self.rescale(self.multiply(dense[0], 1 / 40)))],
                        dtype=object)

    def stage_18_classifier(self, x, weight, bias):
        """One logit per class in slot 0: six slots folded by 1, 2, 4, then ``dim`` tokens by ``n_slot``."""
        g = self.g
        prepared = self.prepare_for_multiply(x[0])
        out = np.empty((len(weight),), dtype=object)
        for class_index in range(len(weight)):
            temp = self.rescale(self.multiply(weight[class_index], prepared))
            pair = self.add(temp, self.rotate(temp, -1))            # slots 0..1
            quad = self.add(pair, self.rotate(pair, -2))            # slots 0..3
            temp = self.add(quad, self.rotate(pair, -4))            # ... plus 4..5, so six not eight
            for step in range(int(np.log2(g.n_slot)), int(np.log2(g.n_slot * g.dim))):
                temp = self.add(temp, self.rotate(temp, -(2 ** step)))
            out[class_index] = self.add(temp, bias[class_index])
        return out
