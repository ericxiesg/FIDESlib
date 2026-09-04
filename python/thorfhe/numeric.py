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
                basis[k] = self.rescale(self.square(self.relinearize(basis[half])))
            else:
                basis.setdefault(k - 1, self._power(basis, k - 1))
                left, right = self.align(basis[k - 1], basis[1])
                basis[k] = self.rescale(self.relinearize(self.multiply(left, right)))
        return basis

    def _power(self, basis, k):
        if k not in basis:
            basis[k] = self.power_basis(basis[1], [k])[k]
        return basis[k]

    def align(self, *cts):
        """Drop every operand to the lowest level present, so they can be combined."""
        target = min(self.engine.level(ct) for ct in cts)
        return tuple(ct if self.engine.level(ct) == target
                     else self.level_down(ct, self.engine.level(ct) - target) for ct in cts)

    def evaluate_polynomial(self, x, coefficients):
        """Baby-step giant-step evaluation of ``sum_i coefficients[i] * x^i`` (low order first).

        ``len(coefficients)`` must be a power of two of at least four. Costs
        ``log2(len(coefficients)) + 1`` levels.
        """
        count = len(coefficients)
        if count < 4 or count & (count - 1):
            raise ValueError(f"coefficient count must be a power of two >= 4, got {count}")

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

    def he_exp(self, x, min_x: float, max_x: float, n: int, wide: bool):
        """THOR's ``he_exp1`` / ``he_exp2``: a degree-15 fit, then ``log2(n)`` squarings.

        The fit approximates ``exp(x / s)`` for ``s = 32`` (narrow) or ``64`` (wide); squaring ``log2(n)``
        times raises it to ``exp(n * x / s)``. ``wide`` also carries THOR's final factor of 128.
        """
        shift = (min_x + max_x) / 2 / (64 if wide else 32)
        shifted = self.add(x, -shift)
        result = self.evaluate_polynomial(shifted, EXP2_COEFFICIENTS if wide else EXP1_COEFFICIENTS)
        for _ in range(int(np.log2(n))):
            result = self.rescale(self.relinearize(self.square(result)))
        return self.multiply(result, 128) if wide else result

    def he_exp1(self, x, min_x: float, max_x: float, n: int):
        return self.he_exp(x, min_x, max_x, n, wide=False)

    def he_exp2(self, x, min_x: float, max_x: float, n: int):
        return self.he_exp(x, min_x, max_x, n, wide=True)
