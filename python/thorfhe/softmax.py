"""THOR stage 07: the softmax over the attention scores.

The construction is worth stating plainly, because it is not the obvious one. There is no exponential
and no division available; there is a degree-15 polynomial that behaves like ``exp(x/4)`` over the
score range, and a Goldschmidt iteration that inverts a denominator known to lie in ``[epsilon, 1]``.
So THOR computes a *low-temperature* softmax and then sharpens it:

* ``he_exp`` gives ``p ~ exp((x - mid) / 4)``; the ``n = 2`` squaring makes it ``exp((x - mid) / 2)``;
* dividing by the sum gives a softmax at temperature 2;
* each :meth:`SoftmaxMixin.update_inv_D` squares the normalised values and re-inverts the new sum,
  which halves the temperature again and costs one more division.

``l`` is how far to take that: ``l = 2`` for most layers (one squaring) and ``l = 4`` for layer 2,
whose scores are wider and whose polynomial fit is correspondingly looser. Squaring an
already-normalised vector is what keeps the intermediate range inside what the next Goldschmidt call
can invert.

**The exponent bookkeeping is worth writing down**, because three factors have to cancel and nothing
in the code says so:

===============================  ==========================================================
``he_exp1`` (narrow, scale 32)   ``exp(0.5 * u)`` for the ``u`` handed to ``he_softmax``
``he_exp2`` (wide, scale 64)     ``exp(0.25 * u)`` - half the slope, because the range is doubled
each ``update_inv_D``            doubles the exponent
:meth:`Softmax.stage_07_softmax` hands ``he_softmax`` **twice** the score (its bootstrap fold)
===============================  ==========================================================

So the narrow path with ``l = 2`` applied to a raw score gives ``exp(score)``, and the wide path with
``l = 2`` applied to a doubled score gives the same thing. Whether ``he_softmax`` takes the narrow or
the wide polynomial is decided by ``max_x >= 30``, so a re-calibration changes the temperature unless
``l`` moves with it - which is what :func:`calibrate` handles.

The ``DeltaCiphertext`` scaling from :mod:`thorfhe.numeric` runs through all of it: ``inv_D`` is carried
as ``ciphertext / delta`` and the integer ``k`` in ``update_inv_D`` is what pulls the product back to
unit scale without spending a level.
"""
from __future__ import annotations

import numpy as np

from .attention import AttentionContext
from .numeric import EXP1_COEFFICIENTS, EXP2_COEFFICIENTS, DivisionMixin, NumericMixin


class SoftmaxMixin:
    """``he_softmax`` and its Newton refinement, over the ``Stages`` primitive surface."""

    #: the internal precision target of every division but the last.
    internal_alpha = 0.1

    def _sum_over_groups(self, terms):
        """Sum the score ciphertexts and then fold the groups together: the softmax denominator."""
        total = terms[0]
        for term in terms[1:]:
            total = self.add(total, term)
        return self.interval_sum(total, self.g.group_size)

    def update_inv_D(self, exp_u, attention_mask, inv_D, delta, precision, alpha, final=False):
        """Square the normalised numerators and invert their new sum: one halving of the temperature.

        ``k`` is the integer that brings ``2 * exp_u * inv_D`` back to unit scale. It is chosen from
        ``delta`` alone, so it costs no level (see :class:`~thorfhe.numeric.DeltaCiphertext`).
        """
        g = self.g
        masked_inverse = [self.rescale(self.multiply(inv_D, mask)) for mask in attention_mask]
        k = max(int(1 / delta / 2), 1)

        squared = np.empty((len(exp_u),), dtype=object)
        for index in range(len(exp_u)):
            doubled = self.add(exp_u[index], exp_u[index])
            # the numerators and the inverse have taken different routes here; align before the product
            numerator, inverse = self.align(doubled, masked_inverse[index])
            scaled = self.multiply(self._times(numerator, inverse), k)
            squared[index] = self.rescale(self.relinearize(self.square(scaled)))

        total = self._sum_over_groups(list(squared))
        epsilon = precision / 128 / 2
        inv_D, delta, precision = self.he_inv(total, self.ones, epsilon=epsilon, alpha=alpha / 10)

        # the final inverse is only needed in the first group, which is where the broadcast starts
        window = g.group_size if final else g.slot_count
        keep = np.zeros(g.slot_count)
        used = (np.arange(g.slot_count) % g.n_slot) < g.n_blocks
        keep[:window] = used[:window]
        return squared, self.rescale(self.multiply(inv_D, keep)), delta, precision

    def he_softmax(self, scores, attention_mask, min_x, max_x, n, l, inv_epsilon, output_alpha):
        """Softmax over the score diagonals, returned as ``dim`` broadcast diagonals.

        ``scores`` is the ``2 * n_output_ciphertexts`` real ciphertexts stage 06 produced.
        """
        g = self.g
        wide = max_x >= 30
        scale = 64 if wide else 32

        # he.py feeds he_exp a ciphertext already divided by the range scale; the division is a scalar
        # multiply, so under FIXEDMANUAL it needs its own rescale before anything is added to it.
        normalised = [self.rescale(self.multiply(ct, 1.0 / scale)) for ct in scores]
        exp_u = [self.he_exp(ct, min_x, max_x, n, wide=wide) for ct in normalised]
        exp_u = [self.rescale(self.multiply(ct, mask)) for ct, mask in zip(exp_u, attention_mask)]

        self.probed("07b.exp", exp_u)

        total = self._sum_over_groups(exp_u)
        self.probed("07c.denominator", [total])
        inv_D, delta, precision = self.he_inv(total, self.ones, epsilon=inv_epsilon,
                                              alpha=self.internal_alpha / 10)
        self.probed("07d.inverse_denominator", [inv_D])
        for _ in range(int(np.log2(l)) - 1):
            exp_u, inv_D, delta, precision = self.update_inv_D(
                exp_u, attention_mask, inv_D, delta, precision, alpha=self.internal_alpha)
        exp_u, inv_D, delta, precision = self.update_inv_D(
            exp_u, attention_mask, inv_D, delta, precision, alpha=output_alpha, final=True)

        return self._broadcast_softmax(exp_u, inv_D, delta)

    def _broadcast_softmax(self, exp_u, inv_D, delta):
        """Pair every numerator diagonal with the right rotation of the denominator, and spread it.

        Produces the ``dim`` broadcast diagonals :meth:`AttentionContext.stage_08_attention_context`
        consumes: ``out[l][group, tau, b] = softmax(S[b])[tau, (l + tau) mod dim]``.
        """
        g = self.g
        half = len(exp_u) // 2

        rotations = [inv_D]
        for _ in range(g.pack - 1):
            rotations.append(self.rotate(rotations[-1], g.group_size))

        scale = int(1 / (2 * delta)) + 1
        out = np.empty((g.dim,), dtype=object)
        for index in range(half):
            merged = self.add(exp_u[index], self.multiply_1j(exp_u[index + half]))
            # scale is an integer, so this costs neither a level nor a rescale
            numerator = self.multiply(merged, scale)
            for step in range(g.pack):
                left, right = self.align(numerator, rotations[step])
                product = self._times(left, right)
                spread = self.interval_sum(product, g.group_size)
                conjugated = self.conjugate(spread)
                position = index * g.pack + step
                out[position] = self.add(spread, conjugated)
                out[position + half * g.pack] = self.multiply_1j(self.subtract(conjugated, spread))
        return out


class Softmax(SoftmaxMixin, NumericMixin, DivisionMixin, AttentionContext):
    """Stages 06, 07 and 08: attention scores, softmax and context, over one engine.

    ``ones`` is the encrypted indicator of the used slots that the Goldschmidt iteration starts from
    (THOR's ``masks["inv_a"]``); it has to be a ciphertext, since the iteration multiplies it by a
    ciphertext correction.
    """

    #: softmax parameters per layer: THOR's he_softmax1, and he_softmax2 for layer 2.
    NARROW = dict(min_x=-27.2493, max_x=21.72692, n=2, l=2, inv_epsilon=2 ** -11, output_alpha=0.01)
    WIDE = dict(min_x=-70.0, max_x=70.0, n=2, l=4, inv_epsilon=2 ** -18, output_alpha=0.01)

    #: How much the key projection's `softmax_scale` was divided by, so that stage 07 hands its
    #: bootstrap a smaller number. Restored by an integer multiply straight after, which costs no
    #: level and no scale degree, so `he_softmax` sees exactly the scores it would have seen: the
    #: calibration, `inv_epsilon` and the Goldschmidt iteration count are all untouched.
    #:
    #: This exists because stage 07 is one of the two sites that do not halve before bootstrapping,
    #: and on the real checkpoint it hands over 1.99 - fine against q0/Delta = 32, and 99.5% of the
    #: bound EasyFHE's parameters would give. Must match what `encode_layer` was given.
    #:
    #: The alternative - halving the scores for real and taking the temperature back with another
    #: squaring in `he_exp` - was measured and is not affordable: squaring the numerators collapses
    #: the denominator from 1.2e-4 to 5.8e-11, which takes Goldschmidt from 9 iterations to 19.
    score_refresh_scale = 1.0

    def __init__(self, engine, geometry, ones=None, **kwargs):
        super().__init__(engine, geometry, **kwargs)
        self.ones = ones

    def stage_07_softmax(self, scores, attention_mask, layer_index: int, parameters=None):
        """Refresh the scores, then soft-max them. ``he.py`` bootstraps here; so does this.

        Note that the pack-and-unpack around the bootstrap **doubles** the scores: the two real
        ciphertexts go in as the real and imaginary halves of one complex ciphertext and come back out
        as ``x + conj(x)`` and ``i(conj(x) - x)``, which are ``2*Re`` and ``2*Im``. THOR's
        ``min_x``/``max_x`` are therefore the range of *twice* the attention score, and a calibration
        that forgets this overflows the polynomial - see :func:`calibrate`.

        ``parameters`` overrides the per-layer calibration.
        """
        half = len(scores) // 2
        refreshed = np.empty((len(scores),), dtype=object)
        for index in range(half):
            merged = self.add(scores[index], self.multiply_1j(scores[index + half]))
            merged = self.bootstrap(merged)
            if layer_index != 2:
                merged = self.level_down(merged, 3)
            conjugated = self.conjugate(merged)
            restore = int(self.score_refresh_scale)
            refreshed[index] = self.add(merged, conjugated)
            refreshed[index + half] = self.multiply_1j(self.subtract(conjugated, merged))
            if restore != 1:
                # an integer multiply: no level, no scale degree - see `score_refresh_scale`
                refreshed[index] = self.multiply(refreshed[index], restore)
                refreshed[index + half] = self.multiply(refreshed[index + half], restore)

        # The bootstrap is the first thing in the layer that stages 01-06 do not do, so when the
        # softmax is the first wrong stage this is the line that splits the question in two: the
        # refreshed scores are 2x the scores stage 06 produced, and those are measurable.
        self.probed("07a.refreshed_scores", refreshed)

        if parameters is None:
            parameters = self.WIDE if layer_index == 2 else self.NARROW
        return self.he_softmax(list(refreshed), attention_mask, **parameters)


def calibrate(scores, *, n: int = 2, l: int = 2, output_alpha: float = 0.01,
              margin: float = 1.05) -> dict:
    """Softmax parameters for a known score distribution.

    THOR hard-codes ``min_x``/``max_x``/``inv_epsilon`` per layer because it calibrated them on real
    activations, and the window is narrow in *both* directions:

    * above ``max_x`` the degree-15 fit stops being an exponential and diverges quickly - a score 10%
      past the range is enough to overflow the denominator;
    * below ``inv_epsilon`` the Goldschmidt iteration is being asked to invert something outside the
      range it was set up for, and returns nonsense.

    So the useful window is roughly three decades of denominator, and it is what the
    ``softmax_scale`` folded into the key projection (1/512, or 1/1024 for layer 2) exists to hit.
    This computes the same choice from a sample of scores, which is what a re-calibration on real
    activations would do. Pass what actually reaches ``he_softmax``: :meth:`Softmax.stage_07_softmax`
    doubles the scores on its way through the bootstrap.
    """
    low, high = float(np.min(scores)), float(np.max(scores))
    pad = (high - low) * (margin - 1) / 2
    low, high = low - pad, high + pad

    wide = high >= 30
    coefficients = EXP2_COEFFICIENTS if wide else EXP1_COEFFICIENTS
    scale = 64 if wide else 32
    numerators = np.polyval(coefficients[::-1], np.asarray(scores) / scale - (low + high) / 2 / scale)
    for _ in range(int(np.log2(n))):
        numerators = numerators ** 2
    if wide:
        numerators = numerators * 128

    smallest = float(np.min(numerators.sum(axis=-1)))
    largest = float(np.max(numerators.sum(axis=-1)))
    if largest > 1.0:
        raise ValueError(f"denominator reaches {largest:.3g}; the scores overflow the polynomial's "
                         "range, scale them down before the softmax")
    inv_epsilon = 2.0 ** int(np.floor(np.log2(smallest)))
    return dict(min_x=low, max_x=high, n=n, l=l, inv_epsilon=inv_epsilon, output_alpha=output_alpha)
