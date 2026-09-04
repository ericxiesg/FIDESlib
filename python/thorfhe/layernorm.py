"""THOR's LayerNorm (stages 11 and 16).

The mean and the variance of a token's 768 features have to be computed from a representation that
spreads those features over 8 ciphertexts, 16 groups and 6 slots. THOR reduces them into **slot 0** of
each token - ``interval_sum`` folds the groups, then three rotations fold the eight slots - computes
the inverse square root there, and broadcasts it back out with three more rotations. Keeping the
statistic in one slot is what makes the inverse square root affordable: it runs on one slot in sixteen.

The normalisation is arranged so the variance lands in ``[min_var/max_var, 1]``, which is the range
:meth:`~thorfhe.numeric.InverseSqrtMixin.he_invsqrt` needs. That is what ``max_for_denominator`` is:
the largest value ``n^2 * Var + var_e`` can take, so dividing by it puts the variance under one, and
``min_var/max_var`` is how far under it can go.

Two consequences are easy to miss:

* the ``x + x`` at the end is never cancelled, so **all three variants return twice the LayerNorm**,
  and the stages after them are calibrated for that;
* the ``/2`` folded into the value mask for variants 2 and 3 is *not* what cancels it - it quarters the
  normalised variance, which moves the usable window up by four. Variant 1 accepts a token variance in
  ``[1.05 * min_var, 1.05 * max_var]``; variants 2 and 3 accept ``[4.2 * min_var, 4.2 * max_var]``.
  Feed one a variance outside its window and the inverse square root has nothing to converge to.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .numeric import InverseSqrtMixin, NumericMixin
from .stages import Stages

#: the rotations that fold a token's ``2^k`` slots together, and broadcast them back.
_FOLD = (1, 2, 4)


def statistic_mask(g: Geometry) -> np.ndarray:
    """THOR ``masks["layernorm"]``: slot 0 of every token, where the mean and variance are reduced."""
    return (np.arange(g.slot_count) % g.n_slot == 0).astype(float)


def value_mask(g: Geometry, scale: float) -> np.ndarray:
    """The slots a LayerNorm input actually occupies, carrying the normalisation scale."""
    return (np.arange(g.slot_count) % g.n_slot < g.out_blocks).astype(float) * scale


class LayerNormStages(NumericMixin, InverseSqrtMixin, Stages):
    """LayerNorm over the ``Stages`` primitive surface, on the post-dense 6-block representation."""

    #: THOR's `w_buffer`: how much headroom the variance bound is given.
    variance_headroom = 1.05

    def _fold_into_slot_zero(self, ct):
        """Sum the groups and then a token's slots, leaving the total in slot 0 (and its neighbours)."""
        folded = self.interval_sum(ct, self.g.group_size)
        for delta in _FOLD:
            folded = self.add(folded, self.rotate(folded, -delta))
        return folded

    def _broadcast_from_slot_zero(self, ct):
        """The inverse: copy slot 0 back across the token's slots."""
        for delta in _FOLD:
            ct = self.add(ct, self.rotate(ct, delta))
        return ct

    @staticmethod
    def variance_window(min_var: float, max_var: float, headroom: float = 1.05):
        """The token variances a variant accepts, given its declared bounds."""
        factor = headroom * (4.0 if min_var > 0.16 else 1.0)
        return factor * min_var, factor * max_var

    def he_layernorm(self, x, gamma, beta, ones, *, var_e: float, min_var: float, max_var: float,
                     alpha: float = 0.001):
        """``2 * (gamma * (x - mean) / sqrt(var + eps) + beta)`` - the doubling is THOR's, see above.

        ``ones`` is the encrypted slot-0 indicator the inverse square root starts from (THOR's
        ``masks["invsqrt_b"]``). ``gamma`` and ``beta`` are plaintexts in the input's own layout.
        """
        g = self.g
        n = g.features
        statistic = statistic_mask(g)

        # normalise so the variance lands in [min_var/max_var, 1] - the range he_invsqrt needs
        max_denominator = (max_var * self.variance_headroom + var_e) * n ** 2
        halve = min_var > 0.16
        values = value_mask(g, (0.5 if halve else 1.0) / np.sqrt(max_denominator))
        masked = [self.rescale(self.multiply(ct, values)) for ct in x]

        total = masked[0]
        for ct in masked[1:]:
            total = self.add(total, ct)
        total = self.rescale(self.multiply(self._fold_into_slot_zero(total), statistic))
        squared_total = self.rescale(self.relinearize(self.square(total)))

        mean = self._broadcast_from_slot_zero(total)
        numerator = []
        for ct in masked:
            scaled, centre = self.align(self.multiply(ct, n), mean)
            numerator.append(self.subtract(scaled, centre))

        sum_of_squares = self.rescale(self.relinearize(self.square(masked[0])))
        for ct in masked[1:]:
            sum_of_squares = self.add(sum_of_squares,
                                      self.rescale(self.relinearize(self.square(ct))))
        sum_of_squares = self.rescale(
            self.multiply(self._fold_into_slot_zero(sum_of_squares), statistic))

        variance = self.subtract(self.multiply(sum_of_squares, n), squared_total)
        variance = self.add(variance, (var_e / max_denominator) * statistic)

        inverse = self.he_invsqrt(variance, ones, statistic,
                                  epsilon=min_var / max_var, alpha=alpha)
        inverse = self._broadcast_from_slot_zero(inverse)

        out = np.empty((len(x),), dtype=object)
        for index in range(len(x)):
            scaled = self.rescale(self.multiply(inverse, gamma[index]))
            left, right = self.align(numerator[index], scaled)
            normalised = self.rescale(self.relinearize(self.multiply(left, right)))
            normalised = self.add(normalised, beta[index])
            out[index] = self.add(normalised, normalised)  # THOR's final doubling, never cancelled
        return out

    def _residual(self, x, y):
        """Add a residual to a branch output. The two are never at the same level.

        The skip connection has been sitting at whatever level stage 01 left it while the branch spent
        a dozen on the attention, so FIXEDMANUAL will not add them. desilofhe aligns implicitly; here
        it is a `level_down` on the residual, which is what the hardware does anyway.
        """
        return [self.add(*self.align(a, b)) for a, b in zip(x, y)]

    def stage_11_attention_layernorm(self, x, dense, gamma, beta, ones):
        """The attention residual and its LayerNorm. No bootstrap: stage 10 leaves enough levels."""
        return self.he_layernorm1(self._residual(x, dense), gamma, beta, ones)

    def stage_15_prepare_layernorm(self, x, y, *, keep_levels=3):
        """The feed-forward residual, bootstrapped back up - four bootstraps for eight ciphertexts.

        Unlike stage 13 there is no compensating halving, so this returns **twice** the residual. That
        is deliberate: :meth:`variance_window` accepts four times the variance for the variants that
        halve their input, which is exactly the ones stage 16 uses after this.
        """
        summed = self._residual(x, y)
        half = len(summed) // 2
        out = np.empty((len(summed),), dtype=object)
        for index in range(half):
            merged = self.bootstrap(self.add(summed[index], self.multiply_1j(summed[index + half])))
            merged = self.level_down(merged, by=keep_levels)
            conj = self.conjugate(merged)
            out[index] = self.add(merged, conj)
            out[index + half] = self.multiply_1j(self.subtract(conj, merged))
        return out

    def stage_16_output_layernorm(self, x, gamma, beta, ones, *, layer_index):
        """THOR routes layers 9 and 10 to the widest variance window; everything else to variant 2."""
        variant = self.he_layernorm3 if layer_index in (9, 10) else self.he_layernorm2
        return variant(x, gamma, beta, ones)

    def he_layernorm1(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.15, max_var=10.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var, max_var=max_var)

    def he_layernorm2(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.2, max_var=150.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var, max_var=max_var)

    def he_layernorm3(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.75, max_var=2500.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var, max_var=max_var)
