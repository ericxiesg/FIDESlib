"""Polynomial evaluation and the softmax's exponential, under FIXEDMANUAL.

``he.py``'s ``evaluate_polynomial_stockmeyer`` builds its power basis and combines the baby polynomials
without tracking levels, because desilofhe aligns them. Doing it explicitly is the bulk of the work
here, and it is worth doing carefully: this is the deepest arithmetic in the network, so a level wasted
here is a level the schedule cannot spend on the layer.

The evaluation order below is a standard baby-step giant-step with baby degree 3, which is what
``he.py`` uses. It costs ``log2(degree + 1) + 1`` levels for a degree-15 polynomial - five - and
computes the same polynomial, so the two agree up to CKKS noise even though the operand levels differ.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

#: THOR he_exp1: degree-15 minimax fit of exp(x/32) on the softmax range, coefficients low to high.
EXP1_COEFFICIENTS = np.array([
    0.0006522770224130905, 0.005218196900295354, 0.020873931555133732, 0.05566510463488879,
    0.11128698173597897, 0.1780344447042791, 0.2379982262906916, 0.27220647178765545,
    0.26787311936472025, 0.23721029007596767, 0.20618261949210986, 0.15202099984697098,
    0.0670090353368128, 0.03881607331549499, 0.05948672763856172, 0.032855468333339584,
])

#: THOR he_exp2: the same for the wider range layer 2 needs, and a final factor of 128.
EXP2_COEFFICIENTS = np.array([
    8.615994668877663e-05, 0.0006877176070101981, 0.0027930565779849003, 0.00749909544008294,
    0.014216636671807894, 0.022268203231858744, 0.03517495704921328, 0.04217318306400685,
    0.02336452059478656, 0.016604320800445653, 0.04817928878801878, 0.0397324053296174,
    -0.009262268572236316, -0.008386712802267769, 0.014226972463907047, 0.008201736399899691,
])


class NumericMixin:
    """Polynomial evaluation and exp, over the ``Stages`` primitive surface."""

    def power_basis(self, x, powers):
        """``{k: x^k}`` for the requested exponents, each canonical.

        Levels fall as the exponent grows - ``x^(2^k)`` sits ``k`` levels below ``x`` - so callers have
        to bring the terms they combine to a common level; :meth:`align` does that.
        """
        basis = {1: x}
        for k in sorted(powers):
            if k in basis:
                continue
            half = k // 2
            if k % 2 == 0:
                basis.setdefault(half, self._power(basis, half))
                basis[k] = self.rescale(self.relinearize(self.square(basis[half])))
            else:
                basis.setdefault(k - 1, self._power(basis, k - 1))
                left, right = self.align(basis[k - 1], basis[1])
                basis[k] = self.rescale(self.relinearize(self.multiply(left, right)))
        return basis

    def _power(self, basis, k):
        if k not in basis:
            basis[k] = self.power_basis(basis[1], [k])[k]
        return basis[k]

    def evaluate_polynomial(self, x, coefficients):
        """Baby-step giant-step evaluation of ``sum_i coefficients[i] * x^i`` (low order first).

        ``len(coefficients)`` must be a multiple of four; anything that is not a power of two is
        zero-padded up to one. Costs ``ceil(log2(len(coefficients))) + 1`` levels.
        """
        count = len(coefficients)
        if count < 4 or count % 4:
            raise ValueError(f"coefficient count must be a multiple of four and at least four, got {count}")
        if count & (count - 1):
            # zero-padding to a power of two is free: the extra babies evaluate to zero and the giant
            # steps are the same ones THOR's uneven split would use.
            padded = np.zeros(1 << int(np.ceil(np.log2(count))))
            padded[:count] = coefficients
            coefficients, count = padded, len(padded)

        giants = [4 * 2 ** k for k in range(int(np.log2(count // 4)))]
        basis = self.power_basis(x, [2, 3] + giants)

        babies = []
        for start in range(0, count, 4):
            chunk = coefficients[start:start + 4]
            terms = []
            for power, coefficient in enumerate(chunk[1:], start=1):
                scaled = self.multiply(basis[power], float(coefficient))
                terms.append(self.rescale(scaled))
            aligned = self.align(*terms)
            total = aligned[0]
            for term in aligned[1:]:
                total = self.add(total, term)
            babies.append(self.add(total, float(chunk[0])))

        for giant in giants:
            merged = []
            for index in range(0, len(babies), 2):
                if index + 1 >= len(babies):
                    merged.append(babies[index])
                    continue
                high, step = self.align(babies[index + 1], basis[giant])
                product = self.rescale(self.relinearize(self.multiply(high, step)))
                low, product = self.align(babies[index], product)
                merged.append(self.add(low, product))
            babies = merged
        return babies[0]

    def he_exp(self, x, min_x: float, max_x: float, n: int, wide: bool, shift: float | None = None):
        """THOR's ``he_exp1`` / ``he_exp2``: a degree-15 fit, then ``log2(n)`` squarings.

        The fit approximates ``exp(x / s)`` for ``s = 32`` (narrow) or ``64`` (wide); squaring ``log2(n)``
        times raises it to ``exp(n * x / s)``. ``wide`` also carries THOR's final factor of 128.

        ``shift`` is the point the fit is centred on, and defaults to the window's midpoint - which is
        what THOR uses, and what ``calibrate`` returns. It is separable from the window because the two
        do different jobs: ``max_x`` chooses the polynomial, while the centre sets where the softmax
        denominator lands, and the denominator's height is what the Goldschmidt iteration is priced on.
        Every term carries a common factor of ``exp(-centre / 2)``, so moving the centre moves the whole
        denominator without touching the ratio between its largest and smallest value.
        """
        centre = (min_x + max_x) / 2 if shift is None else shift
        shifted = self.add(x, -centre / (64 if wide else 32))
        result = self.evaluate_polynomial(shifted, EXP2_COEFFICIENTS if wide else EXP1_COEFFICIENTS)
        for _ in range(int(np.log2(n))):
            result = self.rescale(self.relinearize(self.square(result)))
        return self.multiply(result, 128) if wide else result

    def he_exp1(self, x, min_x: float, max_x: float, n: int, shift: float | None = None):
        return self.he_exp(x, min_x, max_x, n, wide=False, shift=shift)

    def he_exp2(self, x, min_x: float, max_x: float, n: int, shift: float | None = None):
        return self.he_exp(x, min_x, max_x, n, wide=True, shift=shift)

    def he_tanh_for_pooler(self, x):
        """``tanh`` for the pooler, as two degree-15 polynomials with a bootstrap on either side.

        Unlike the GELU composite this one is cheap - the ``/40`` in :meth:`stage_17_pooler` puts the
        argument comfortably inside the fitted range, so degree 15 twice is enough.
        """
        inner = self.evaluate_polynomial(self.bootstrap(x), POOLER_INNER)
        return self.bootstrap(self.evaluate_polynomial(inner, POOLER_OUTER))


class DeltaCiphertext:
    """A ciphertext together with the factor its plaintext has been scaled by.

    The represented value is ``ciphertext / delta``. THOR carries this pair through the division so it
    can rescale by *integers* - which cost no level - whenever the ciphertext's magnitude drifts too
    far below the CKKS scale to keep its precision. Doubling via ``x + conj(x)`` is free in the same
    way, and used for the same purpose.
    """

    __slots__ = ("ciphertext", "delta")

    def __init__(self, ciphertext, delta: float):
        self.ciphertext = ciphertext
        self.delta = delta

    def __repr__(self):
        return f"DeltaCiphertext(delta={self.delta:g})"


class DivisionMixin:
    #: Check that a denominator lies in the range its iteration was set up for, where the values can
    #: be read. Off for a dry run over dummy data - :func:`thorfhe.he.plan_rotations` measures the
    #: *schedule*, and the zeros it feeds the layer make every value-based check meaningless.
    check_ranges = True

    """Goldschmidt division, as ``he.py``'s ``he_inv``."""

    #: keep the ciphertext's magnitude within this many bits of the scale before rescaling by an integer.
    delta_headroom_bits = 8

    @staticmethod
    def goldschmidt_iterations(epsilon: float, alpha: float) -> int:
        """How many iterations the loop runs. Data-independent, so the level cost is known up front."""
        error, count = epsilon, 0
        while error < 1 - alpha:
            k = 2 / (error + 1)
            error = k * error * (2 - k * error)
            count += 1
        return count

    def he_inv(self, denominator, ones, epsilon: float, alpha: float, delta: float = 1.0):
        """``1 / denominator`` for a denominator known to lie in ``[epsilon, 1]``.

        ``ones`` is an encrypted indicator of the slots that carry data - THOR's ``masks["inv_a"]`` -
        which is what the iteration starts from and what confines the result to those slots.

        Returns ``(ciphertext, delta, precision)``: the value is ``ciphertext / delta``, and
        ``precision`` is the achieved lower bound on the normalised denominator (it ends above
        ``1 - alpha``). One level per iteration.
        """
        self._check_inversion_range(denominator, ones, epsilon)
        # `ones` is a fresh encryption and the denominator has been through a stage, so they are
        # almost never at the same level; FIXEDMANUAL will not multiply across levels.
        start, refreshed = self.align(ones, self.bootstrap(denominator))
        a = DeltaCiphertext(start, delta)
        b = DeltaCiphertext(refreshed, delta)
        error = epsilon
        iterations = 0

        while error < 1 - alpha:
            iterations += 1
            k = 2 / (error + 1)

            # (2/k) * delta_b - b, i.e. Goldschmidt's `2 - k*value` carried in the scaled representation
            correction = self.prepare_for_multiply(self.subtract(2 / k * b.delta, b.ciphertext))
            a = DeltaCiphertext(self._times(a.ciphertext, correction), a.delta * b.delta / k ** 2)
            b = DeltaCiphertext(self._times(b.ciphertext, correction), b.delta * b.delta / k ** 2)
            error = k * error * (2 - k * error)

            a, b = self._restore_magnitude(a, b)

        return a.ciphertext, a.delta, error

    def _check_inversion_range(self, denominator, ones, epsilon: float):
        """Refuse a denominator outside ``[epsilon, 1]`` where the values can be read.

        The iteration's schedule - how many steps, and the ``k`` at each - is derived from ``epsilon``
        alone, so a denominator below it is not merely less accurate. It **saturates**: measured on
        the clear engine at ``epsilon = 2^-11``, a denominator of ``epsilon/2`` comes back 3.6% low,
        ``epsilon/4`` 22% low, ``epsilon/10`` 56% low, and past that the answer stops growing at all,
        pinned near 11500 however small the input gets. Nothing raises, and the result is an ordinary
        finite number - which is why a softmax built on it produces attention weights that look like
        weights and classify at chance.

        Only the slots ``ones`` marks are examined: the rest are empty by construction, and it is the
        ``ones`` operand that confines the iteration to the ones that are not.

        The device cannot do this - it has no way to read a ciphertext - so the check lives on the
        engines that can, and the corresponding device-side signal is the magnitude probe's ``p50``
        against the same ``epsilon``.
        """
        if not (self.check_ranges and getattr(self.engine, "inspectable", False)):
            return
        values = np.real(np.asarray(self.engine.decrypt(denominator)))
        carried = np.abs(np.real(np.asarray(self.engine.decrypt(ones)))) > 1e-9
        if not carried.any():
            return
        low, high = float(values[carried].min()), float(values[carried].max())
        if low >= epsilon and high <= 1.0:
            return
        median = float(np.median(values[carried]))
        raise ValueError(
            f"he_inv: the denominator runs over [{low:.4g}, {high:.4g}] (median {median:.4g}) on the "
            f"{int(carried.sum())} slots that carry data, but the iteration is set up for "
            f"[{epsilon:.4g}, 1]. Below the lower bound Goldschmidt saturates rather than failing - "
            f"it returns a finite, plausible, wrong number - so this cannot be left to surface "
            f"downstream. Re-calibrate: the scale folded into the key projection is what places the "
            f"denominator in this window (see thorfhe.softmax.calibrate).")

    def _times(self, x, y):
        """Ciphertext product, relinearised and brought back to canonical scale."""
        return self.rescale(self.relinearize(self.multiply(x, y)))

    def _restore_magnitude(self, a: DeltaCiphertext, b: DeltaCiphertext):
        """Scale both operands back up without spending a level.

        ``delta`` shrinks quadratically each iteration, so the ciphertexts would sink into the CKKS
        noise floor. Doubling through the conjugate and multiplying by an integer are both level-free,
        so the value (``ciphertext / delta``) is untouched while the magnitude is restored.
        """
        headroom = 2 ** self.delta_headroom_bits
        if int(1 / b.delta / headroom) > 1:
            for scaled in (a, b):
                scaled.ciphertext = self.add(scaled.ciphertext, self.conjugate(scaled.ciphertext))
                scaled.delta *= 2
        factor = max(int(1 / b.delta / headroom), 1)
        if factor > 1:
            a = DeltaCiphertext(self.multiply(a.ciphertext, factor), a.delta * factor)
            b = DeltaCiphertext(self.multiply(b.ciphertext, factor), b.delta * factor)
        return a, b


class InverseSqrtMixin:
    """Cubically-convergent inverse square root, as ``he.py``'s ``he_invsqrt``.

    The iteration keeps two ciphertexts: ``a``, which is driven to 1, and ``b``, which accumulates
    ``1 / sqrt(a_0)``. Each step picks ``k`` as the middle root of

        ``(1 - e^3) k^2 + (6 e^2 - 6) k + (9 - 9 e) = 0``

    which is the choice that makes ``e -> k e (3 - k e)^2 / 4`` converge fastest from the current lower
    bound ``e``. Cubic convergence means five or six iterations cover a range of several decades, which
    is why LayerNorm can afford it where a Newton iteration could not.

    Everything is confined to ``mask``, the slots that carry the statistic; the scalars are folded into
    that mask rather than applied separately, exactly as ``he.py`` does.
    """

    @staticmethod
    def invsqrt_step(e: float) -> float:
        """The ``k`` for the current lower bound, i.e. the middle root of the cubic's derivative."""
        roots = np.roots([1 - e ** 3, 6 * e ** 2 - 6, 9 - 9 * e])
        return float(np.real(roots[1]))

    @classmethod
    def invsqrt_iterations(cls, epsilon: float, alpha: float) -> int:
        """How many iterations the loop runs; data-independent, so the level cost is known up front."""
        error, count = epsilon, 0
        while error < 1 - alpha:
            k = cls.invsqrt_step(error)
            error = k * error * (3 - k * error) ** 2 / 4
            count += 1
        return count

    def he_invsqrt(self, denominator, ones, mask, epsilon: float, alpha: float):
        """``1 / sqrt(denominator)`` for a denominator known to lie in ``[epsilon, 1]``.

        ``ones`` is the encrypted indicator the accumulator starts from (THOR's ``masks["invsqrt_b"]``)
        and ``mask`` the plaintext indicator of the same slots. Two levels per iteration.
        """
        a = denominator
        b = ones
        error = epsilon

        while error < 1 - alpha:
            k = self.invsqrt_step(error)
            correction = self.subtract((3 / k) * mask, a)

            scaled_b = self.rescale(self.multiply(b, (k ** 1.5 / 2) * mask))
            left, right = self.align(scaled_b, correction)
            next_b = self.rescale(self.relinearize(self.multiply(left, right)))

            scaled_a = self.rescale(self.multiply(a, (k ** 3 / 4) * mask))
            squared = self.rescale(self.relinearize(self.square(correction)))
            left, right = self.align(scaled_a, squared)
            a = self.rescale(self.relinearize(self.multiply(left, right)))

            b = next_b
            error = k * error * (3 - k * error) ** 2 / 4

        return b


#: THOR's GELU: the inner polynomial of the two-stage tanh composite (degree 31, low order first).
#: THOR ``he_tanh_single_for_pooler``: two degree-15 polynomials, with a bootstrap on each side.
#: A far easier fit than GELU's - the pooler's own ``/40`` puts the argument well inside the range.
POOLER_INNER = np.array([
    2.06201784e-03, 1.95056729e01, -4.29024545e-01, -4.13341496e02, 8.17596753e00, 3.58757327e03,
    -5.64098990e01, -1.51158283e04, 1.82989351e02, 3.42189880e04, -3.01953016e02, -4.25793697e04,
    2.45150249e02, 2.74279201e04, -7.76519925e01, -7.14529052e03,
])

POOLER_OUTER = np.array([
    -4.08442578e-03, 2.02874846e00, 1.30194294e-02, -2.38934873e00, -1.77631073e-02, 2.74974898e00,
    1.17381072e-02, -2.43259032e00, -2.22416152e-03, 1.46476749e00, -1.42873183e-03, -5.41327356e-01,
    7.96793166e-04, 1.08762008e-01, -1.12320034e-04, -9.02573450e-03,
])


GELU_INNER = np.array([
    -1.06240033e-05, 1.64454894e-04, -5.83533517e-04, -3.80912692e-04, 2.24431193e-03,
    8.92295204e-03, -1.05277477e-02, -1.91827040e-02, -2.04634786e-01, 4.54014410e-01,
    -5.40759203e-01, 5.67745523e00, -1.36433727e01, 1.82574621e01, -8.48849601e01,
    1.28686741e02, 3.66720281e02, -1.01400159e03, -1.26278856e02, 2.21728878e03,
    -9.95421415e02, -2.31059465e03, 1.73583957e03, 1.27394360e03, -1.27836230e03,
    -3.66781716e02, 4.79663919e02, 4.94610178e01, -9.06754761e01, -2.36515790e00,
    8.74311855e00, 1.62838703e-02,
])[::-1].copy()

#: The outer polynomial (degree 27), already carrying THOR's factor of a half.
GELU_OUTER = np.array([
    -1.70270667e02, 6.81076279e01, 1.79197364e03, -6.81621043e02, -8.49256169e03,
    3.05629446e03, 2.39579397e04, -8.10435126e03, -4.48145152e04, 1.41297616e04,
    5.86197512e04, -1.70371505e04, -5.51326382e04, 1.45532495e04, 3.77866438e04,
    -8.87673890e03, -1.89514802e04, 3.84972853e03, 6.94169727e03, -1.16901058e03,
    -1.84658407e03, 2.41693754e02, 3.54452276e02, -3.24499570e01, -4.91918227e01,
    2.58122977e00, 5.78392852e00, -9.45171527e-02,
])[::-1].copy() * 0.5

#: GELU's argument is carried divided by this, so the composite's own range is about [-1, 1].
#: THOR's invariant: every activation ciphertext carries *twice* the value it represents. The weight
#: encoders halve, the bias is added before the ``y + conj(y)`` that doubles, and LayerNorm's
#: uncancelled doubling hands the next layer the same footing - so a layer is entered on it too.
#:
#: It lives here, with :data:`GELU_SCALE`, because three places need it and they must not drift: the
#: amplitude a layer is entered at (``bench --output-scale``), the one the feed-forward divides out of
#: GELU's argument, and the one the pooler divides out of ``tanh``'s. Every *linear* stage carries the
#: factor through untouched, so a disagreement cancels everywhere except at a non-linearity - where it
#: silently computes a different function. That has already cost this port two bugs in ``SOFTMAX_SCALES``.
ACTIVATION_SCALE = 2.0

GELU_SCALE = 64


class GeluMixin:
    """THOR's GELU: two polynomials in series, then one multiply.

    A single minimax fit of ``tanh`` over the pre-activation range would need a degree far past what
    is affordable, so THOR composes a degree-31 with a degree-27, which reaches the same accuracy for
    twelve levels instead of a hopeless number. The composite approximates ``tanh(...) / 2``, and
    ``64 * x * (t + 1/2)`` is then ``gelu(64 * x)`` to about 2e-4 relative.

    The factor of 64 is a convention, not a nicety: the ciphertext carries the pre-activation divided
    by 64 so that the composite's own argument lands in ``[-1, 1]``, which is where the fit is valid.
    """

    @staticmethod
    @lru_cache(maxsize=None)
    def _gelu_inner_for(carrier: float):
        """``GELU_INNER`` pre-divided by ``carrier``, so the composite can take the doubled argument.

        Evaluating ``p(x / c)`` is the same as evaluating ``[p_k / c^k]`` at ``x``, and the second form
        costs nothing: the coefficients are plaintext and this runs once. The first form costs a
        rescale, which is a whole level, every time GELU runs.

        The outer polynomial already carries a constant of this kind - it is exactly half of THOR's,
        which is the other side of the same doubled-ciphertext bookkeeping - so this only moves the
        input scaling to where the output scaling already lives.
        """
        inner = np.asarray(GELU_INNER, dtype=float)
        return inner if carrier == 1.0 else inner * (1.0 / carrier) ** np.arange(len(inner))

    def he_tanh_for_gelu(self, x, carrier: float = 1.0):
        """``tanh(64 x * sqrt(2/pi) * (1 + ...)) / 2``, as the two-polynomial composite. Twelve levels."""
        inner = self.evaluate_polynomial(x, self._gelu_inner_for(carrier))
        return self.evaluate_polynomial(inner, GELU_OUTER)

    def gelu(self, x, carrier: float = 1.0):
        """``carrier * gelu(64 * x / carrier)``, for a ciphertext carrying ``carrier * pre / 64``.

        ``carrier`` exists because GELU is the one place THOR's doubled-ciphertext convention cannot
        simply pass through. Everywhere else a ciphertext holds twice the value it represents and the
        factor commutes with the arithmetic; here the composite's argument has to be the *actual*
        pre-activation over 64, since the degree-31 fit is only valid on ``[-1, 1]``. So the tanh is
        evaluated at ``x / carrier`` while the linear factor keeps the full ``x``, and the product
        comes out on the same doubled footing as everything around it.

        That division is folded into the inner coefficients rather than performed on the ciphertext,
        which is where the twelve levels of the composite become twelve rather than thirteen.
        """
        # The division by `carrier` lives in the inner polynomial's coefficients, not in a rescale
        # here: it is the same arithmetic and it is a level cheaper. See `_gelu_inner_for`.
        shifted = self.add(self.he_tanh_for_gelu(x, carrier=carrier), 0.5)
        # The linear factor stays 64 whatever the carrier: `x` already carries it, so `64 * x` is
        # `carrier * pre_activation` - which is the footing the result is wanted on. Only the tanh's
        # argument has to be un-carried. It also has to stay an integer, because an integer multiply
        # is level- and scale-free under FIXEDMANUAL and a float one costs a rescale.
        scaled, shifted = self.align(self.multiply(x, GELU_SCALE), shifted)
        return self.rescale(self.relinearize(self.multiply(scaled, shifted)))
