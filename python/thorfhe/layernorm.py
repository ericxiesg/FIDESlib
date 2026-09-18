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

    #: The factor stage 15's input has been divided by, folded into plaintexts upstream (see
    #: `thorfhe.layer.encode_layer`). Stage 16 normalises, so its output does not move - but the
    #: variance it is told to expect does, by the square. One here means no scaling, which is what
    #: every caller that has not been through `encode_layer(residual_scale=...)` must use: telling
    #: this stage a scale the weights were not encoded with puts `he_invsqrt` outside its range, and
    #: that fails the way everything in this family fails, by returning a plausible wrong number.
    residual_scale = 1.0

    #: An extra division in front of `refresh`'s bootstrap, undone by an integer multiply after, so
    #: the stage stays the identity it is documented to be. One means the halving alone, which is
    #: what every caller that has not asked for more gets. See :meth:`refresh`.
    refresh_scale = 1.0

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
    def variance_window(min_var: float, max_var: float, headroom: float = 1.05,
                        halves: bool | None = None):
        """The token variances a variant accepts, given its declared bounds.

        A variant that halves its input accepts four times the variance, because the halving is
        undone by the doubled representation stage 15 hands it (see
        :meth:`stage_15_prepare_layernorm`).

        ``halves`` used to be inferred as ``min_var > 0.16``, a threshold sitting between variant 1's
        0.15 and variant 2's 0.2 - that is variant identity encoded as a magnitude comparison, and it
        is wrong the moment the bounds are scaled rather than chosen. Pass it; the default keeps the
        old inference so existing callers do not change behaviour.
        """
        if halves is None:
            halves = min_var > 0.16
        factor = headroom * (4.0 if halves else 1.0)
        return factor * min_var, factor * max_var

    def he_layernorm(self, x, gamma, beta, ones, *, var_e: float, min_var: float, max_var: float,
                     alpha: float = 0.001, halves: bool | None = None):
        """``2 * (gamma * (x - mean) / sqrt(var + eps) + beta)`` - the doubling is THOR's, see above.

        ``ones`` is the encrypted slot-0 indicator the inverse square root starts from (THOR's
        ``masks["invsqrt_b"]``). ``gamma`` and ``beta`` are plaintexts in the input's own layout.
        """
        g = self.g
        n = g.features
        statistic = statistic_mask(g)

        # normalise so the variance lands in [min_var/max_var, 1] - the range he_invsqrt needs
        max_denominator = (max_var * self.variance_headroom + var_e) * n ** 2
        # Which variant halves its input is a property of the variant, not of how large its bounds
        # happen to be - see `variance_window`. Inferred only when the caller does not say.
        halve = (min_var > 0.16) if halves is None else halves
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

        # The two halves, before they cancel. `variance` is their difference and it is four to
        # eight orders of magnitude smaller than either - so the ratio these two report *is* the
        # condition number of the subtraction, measured rather than assumed, and it is the number
        # that says how much relative precision a device has to carry into this line.
        scaled_squares = self.multiply(sum_of_squares, n)
        self.probed("11a1.n_sum_of_squares", [scaled_squares])
        self.probed("11a2.squared_total", [squared_total])
        variance = self.subtract(scaled_squares, squared_total)
        variance = self.add(variance, (var_e / max_denominator) * statistic)

        # The same probes stage 07 has, for the same reason: `07a`-`07d` split the softmax into
        # "the scores are wrong" and "the inverse is wrong" in one line, and stage 11 is where a
        # device run diverges with nothing to say which half. The variance is what `he_invsqrt` has
        # to find in `[min_var/max_var, 1]`, and the clear engine puts it there - so if the device
        # does not, this is the line that says so.
        self.probed("11a.variance", [variance])
        inverse = self.he_invsqrt(variance, ones, statistic,
                                  epsilon=min_var / max_var, alpha=alpha)
        self.probed("11b.inverse_sqrt", [inverse])
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
        """The attention residual and its LayerNorm. No bootstrap: stage 10 leaves enough levels.

        Whether it does is exactly what `11.residual` reports: the residual is aligned down to the
        dense output, so this level is the one the whole 14-level LayerNorm chain has to run on.
        """
        residual = self._residual(x, dense)
        self.probed("11.residual", residual)
        return self.he_layernorm1(residual, gamma, beta, ones)

    def refresh(self, x, scale=None):
        """Bootstrap a real 8-ciphertext bundle, folding pairs so it costs four bootstraps not eight.

        Not a THOR stage - ``he.py`` has no refresh here. It is an inserted one, and it is *semantically
        the identity*: the halving in front of the bootstrap and the doubling implicit in
        ``temp + conj(temp)`` cancel exactly, the same way they do in :meth:`stage_13_gelu`. What it
        buys is level headroom.

        A layer's deepest chain runs from the softmax's own bootstrap to the rescale in front of
        GELU's, and costs 37 levels: 14 for the tail of the softmax, 5 for the context and the
        attention dense, 14 for LayerNorm, and 4 for the feed-forward expansion and that rescale.
        Splitting it needs one refresh, and the split that minimises the longer half is right after
        stage 10 - 19 levels before, 18 after - so the layer needs a bootstrap level of 20 instead of
        38, and a depth of about 34 instead of 52. That is the difference between fitting on a 32 GiB
        card and not.

        ``refresh_scale`` divides once more before the bootstrap and multiplies back afterwards by an
        integer, which costs neither a level nor a scale degree, so the identity above still holds
        exactly. It exists because the halving is not enough on the real checkpoint: the attention
        dense reaches 3.8 to 5.2, which halves to 1.9 to 2.6 - comfortable against q0/Delta = 32 and
        over the bound EasyFHE's parameters would give. This was the third such site to be found and
        the least obvious, at 6 to 8% of the current bound.
        """
        half = len(x) // 2
        out = np.empty((len(x),), dtype=object)
        # `scale` overrides the attribute for one call, because the two sites that refresh want
        # different divisors: after stage 10 the attention dense reaches 3.8 to 5.2, while a layer
        # boundary carries the whole hidden state at about 19 to 23.
        restore = int(self.refresh_scale if scale is None else scale)
        for index in range(half):
            merged = self.add(x[index], self.multiply_1j(x[index + half]))
            merged = self.bootstrap(self.rescale(self.multiply(merged, 0.5 / restore)))
            conjugated = self.conjugate(merged)
            out[index] = self.add(merged, conjugated)
            out[index + half] = self.multiply_1j(self.subtract(conjugated, merged))
            if restore != 1:
                out[index] = self.multiply(out[index], restore)
                out[index + half] = self.multiply(out[index + half], restore)
        return out

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

    #: The bounds each stage-16 variant declares, before `residual_scale`.
    VARIANT_BOUNDS = {2: (1e-5, 0.2, 150.0), 3: (1e-5, 0.75, 2500.0)}

    def stage_16_output_layernorm(self, x, gamma, beta, ones, *, layer_index):
        """THOR routes layers 9 and 10 to the widest variance window; everything else to variant 2.

        Dividing the input by `residual_scale` divides the variance by its square, so the bounds
        follow. Their *ratio* is untouched, which is what sets `he_invsqrt`'s iteration count - so
        this changes no levels.
        """
        number = 3 if layer_index in (9, 10) else 2
        variant = self.he_layernorm3 if number == 3 else self.he_layernorm2
        var_e, min_var, max_var = self.VARIANT_BOUNDS[number]
        square = self.residual_scale ** 2
        return variant(x, gamma, beta, ones, var_e=var_e / square,
                       min_var=min_var / square, max_var=max_var / square)

    #: Whether each variant halves its input. Variant 1 runs after stage 11, whose output is not
    #: doubled; variants 2 and 3 run after stage 15, whose output is. Stated rather than inferred
    #: from the bounds, so the bounds can be scaled without changing which variant this is.
    HALVES = {1: False, 2: True, 3: True}

    def he_layernorm1(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.15, max_var=10.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var,
                                 max_var=max_var, halves=self.HALVES[1])

    def he_layernorm2(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.2, max_var=150.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var,
                                 max_var=max_var, halves=self.HALVES[2])

    def he_layernorm3(self, x, gamma, beta, ones, var_e=1e-5, min_var=0.75, max_var=2500.0):
        return self.he_layernorm(x, gamma, beta, ones, var_e=var_e, min_var=min_var,
                                 max_var=max_var, halves=self.HALVES[3])
