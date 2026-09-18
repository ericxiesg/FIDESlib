"""THOR's dense layers after attention: stage 10, and the machinery the feed-forward stages reuse.

Stage 10 is the same plaintext-ciphertext matrix product as the QKV projections, but it *changes
representation*: it consumes twelve blocks of 64 features and produces six of 128. That is why
:class:`~thorfhe.geometry.Geometry` no longer insists ``features // n_out == n_blocks`` - the two are
equal for the projections by coincidence, not by construction.

The fold at the end is what halves the blocking. ``wx + rotate(mask * wx, -out_blocks)`` adds slot
``b + out_blocks`` into slot ``b``, so blocks 0 to 5 carry the answer. **Blocks 6 to 11 are left
holding the unfolded upper half** - THOR does not clear them, and every consumer of a stage 10 output
has to treat them as padding. :func:`used_block_mask` is that mask, spelled out.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .stages import Stages


def dense_fold_mask(g: Geometry) -> np.ndarray:
    """THOR ``masks["attention_dense"]``: the upper half of a token's blocks, to be folded down."""
    return (np.arange(g.slot_count) % g.n_slot >= g.out_blocks).astype(float)


def used_block_mask(g: Geometry) -> np.ndarray:
    """The blocks a stage 10 output actually carries, i.e. the first ``out_blocks`` of each token."""
    return (np.arange(g.slot_count) % g.n_slot < g.out_blocks).astype(float)


class DenseStages(Stages):
    """Stage 10 over the ``Stages`` primitive surface.

    The geometry must be the *working* one - ``n_blocks`` is the window the partial products are
    recombined in (12), while ``out_blocks`` is what survives the fold (6).
    """

    def stage_10_attention_dense(self, x, weight, bias):
        """``context @ W.T + 2 * b``, packed as ``out[ct][group, t, b] = y[t, n_out*b + (l + t) % n_out]``.

        ``l = ct * pack + group`` runs over all ``dim`` diagonals and ``b`` over ``out_blocks``; the
        remaining blocks are the fold's leftovers, see the module docstring.

        The ``2 *`` on the bias is THOR's, the same as in the QKV projections: the weights are halved
        to pay for the ``y + conj(y)`` and the bias is not.
        """
        wx = self.pcmm(weight, x)
        fold = dense_fold_mask(self.g)

        out = np.empty((wx.shape[0],), dtype=object)
        for index in range(wx.shape[0]):
            folded = self.rotate(self.rescale(self.multiply(fold, wx[index])), -self.g.out_blocks)
            # the mask multiply cost a level, so the un-folded half follows it down
            biased = self.add(self.add(self.level_down(wx[index], 1), folded), bias[index])
            out[index] = self.add(biased, self.conjugate(biased))
        self.probed("10.attention_dense", out)
        return out
