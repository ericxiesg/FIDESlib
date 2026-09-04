"""THOR stages 01-05, written once against an engine-shaped object.

Transcribed from ``THOR/src/thor/he.py`` (``stage_01_complexify_x`` .. ``stage_05_value`` and the
``parallel_diagonal_pc_mult`` / ``rotate_internal`` / ``pcmm`` / ``apply_qkv_weight_bias`` helpers),
with THOR's hard-coded ``16``/``12``/``2**11`` replaced by :class:`~thorfhe.geometry.Geometry` fields.

Two conventions differ from THOR and are handled here, once, rather than being spread through the code:

* **Rotation direction.** desilofhe's ``rotate(ct, delta)`` moves slot ``i`` to slot ``i + delta``;
  OpenFHE's and FIDESlib's ``EvalRotate(ct, delta)`` moves it to ``i - delta``. :meth:`Stages.rotate`
  negates, so every ``delta`` in this file reads exactly as it does in ``he.py``.
* **Plaintext operand order.** ``he.py`` calls ``multiply(mask, x)`` and ``add(bias, wx)`` with the
  plaintext first; ``Engine.multiply`` dispatches on the second operand, so the helpers below swap.

A third difference is not a convention but a consequence of the scaling technique. desilofhe manages
CKKS scales automatically, so ``he.py`` can call ``rescale`` where a rescale is merely *pending* and can
subtract a freshly masked ciphertext from an unmasked one. fideslib runs FIXEDMANUAL, where a
ciphertext is canonical at scale ``Delta``, a product sits at ``Delta^2`` until rescaled, and adding the
two is silently wrong. Two places therefore read differently from ``he.py``:

* ``prepare_for_multiply`` is the identity. In ``he.py`` it is ``ntt(rescale(x))``: the NTT is a no-op
  because fideslib keeps ciphertexts in the evaluation domain, and the rescale is desilofhe settling a
  deferred product. Here the operand is already canonical and the rescale happens once, after the
  accumulation, where ``parallel_diagonal_pc_mult`` already does it.
* ``rotate_internal`` masks with ``mask`` and with ``1 - mask`` instead of masking once and subtracting,
  so both halves sit at ``Delta^2`` and one rescale brings the result back. ``x - mask * x`` and
  ``(1 - mask) * x`` are the same value; only the scale bookkeeping differs.

:class:`~thorfhe.clear.ClearEngine` enforces the contract, so a mistake here fails a millisecond test
rather than showing up as noise after a GPU run.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry


class Stages:
    """Stage 01-05 of one BERT layer, over ``engine`` (``pyfideslib.Engine`` or ``ClearEngine``)."""

    def __init__(self, engine, geometry: Geometry, masks=None, complement_masks=None):
        self.engine = engine
        self.g = geometry
        #: ``rotate_internal`` masks, keyed by delta: 1 on the slots with ``slot % n_slot < delta``.
        self.masks = masks
        #: their complements, so both halves of ``rotate_internal`` are products (see module docstring).
        self.complement_masks = complement_masks

    # ---------------------------------------------------------------- primitives
    def rotate(self, x, delta: int):
        """THOR's rotation direction: slot ``i`` moves to slot ``i + delta``."""
        return self.engine.rotate(x, -int(delta) % self.g.slot_count)

    def add(self, x, y):
        # he.py passes plaintexts on either side; the engine wants the ciphertext first.
        if not self._is_ciphertext(x):
            x, y = y, x
        return self.engine.add(x, y)

    def add_inplace(self, x, y):
        return self.engine.add_inplace(x, y)

    def subtract(self, x, y):
        return self.engine.subtract(x, y)

    def multiply(self, x, y):
        if not self._is_ciphertext(x):
            x, y = y, x
        return self.engine.multiply(x, y)

    def conjugate(self, x):
        return self.engine.conjugate(x)

    def multiply_1j(self, x):
        return self.engine.multiply_1j(x)

    def rescale(self, x):
        return self.engine.rescale(x)

    def relinearize(self, x):
        return self.engine.relinearize(x)

    def square(self, x):
        return self.engine.square(x)

    def bootstrap(self, x, keep_levels=None):
        return self.engine.bootstrap(x, keep_levels)

    def level_down(self, x, by: int):
        return self.engine.level_down(x, by)

    def prepare_for_multiply(self, x):
        """Identity under FIXEDMANUAL - see the module docstring for why ``he.py`` rescales here."""
        return x

    #: everything the engines accept as the plaintext side of a mixed operation.
    _PLAINTEXT_TYPES = ("LightPlaintext", "Plaintext")

    @classmethod
    def _is_ciphertext(cls, value) -> bool:
        if isinstance(value, (int, float, complex, np.ndarray, np.generic)):
            return False
        return type(value).__name__ not in cls._PLAINTEXT_TYPES

    # ---------------------------------------------------------------- stages
    def stage_01_complexify_x(self, x, layer_index: int):
        """Split each ciphertext into its real and imaginary halves (and, from layer 1 on, fold the copy).

        Returns ``(x_real_imag, x_complex)``: the first is what layernorm consumes later, the second is
        what the QKV product consumes. For layer 0 the input is already complex-packed, so the second
        is the input itself (level-dropped by 6, matching he.py's schedule).
        """
        n_in_ct = self.g.n_input_ciphertexts
        x_cplx = np.full((n_in_ct,), None, dtype=object)

        if x.shape == (2 * n_in_ct,):
            for i in range(n_in_ct):
                x_cplx[i] = self.add(x[i], self.multiply_1j(x[i + n_in_ct]))
            if layer_index != 0:
                for i in range(n_in_ct):
                    self.add_inplace(x_cplx[i], self.rotate(x_cplx[i], self.g.n_blocks // 2))
            return x, x_cplx

        if x.shape != (n_in_ct,):
            raise ValueError(f"stage_01 expects {n_in_ct} or {2 * n_in_ct} ciphertexts, got {x.shape}")

        temp = np.full((2 * n_in_ct,), None, dtype=object)
        for i in range(n_in_ct):
            orig = x[i]
            conj = self.conjugate(orig)
            # The halving is a scalar multiply, so it costs a level under FIXEDMANUAL; he.py leaves the
            # rescale to desilofhe. Only stage 11 consumes `temp`, so the extra level is charged there.
            temp[i] = self.rescale(self.multiply(self.add(orig, conj), 1 / 2))
            temp[i + n_in_ct] = self.rescale(
                self.multiply(self.multiply_1j(self.subtract(conj, orig)), 1 / 2))
            x_cplx[i] = self.level_down(orig, 6)
        return temp, x_cplx

    def stage_02_make_rotated_copies(self, x):
        """One copy per group: copy ``r`` has group ``r`` of the original aligned to group 0."""
        rotated = np.full((self.g.pack * x.shape[0],), None, dtype=object)
        for source in range(x.shape[0]):
            base = self.g.pack * source
            rotated[base] = x[source]
            for r in range(1, self.g.pack):
                rotated[base + r] = self.rotate(rotated[base + r - 1], -self.g.group_size)
        return rotated

    def parallel_diagonal_pc_mult(self, ws, xs):
        """The plaintext-ciphertext inner product, one term per input copy, for every block diagonal."""
        out_dim, diag_dim, in_dim = ws.shape
        prepared = np.full(xs.shape, None, dtype=object)
        for index, x in np.ndenumerate(xs):
            prepared[index] = self.prepare_for_multiply(x)

        output = np.full((out_dim, diag_dim), None, dtype=object)
        for out_index in range(out_dim):
            for diag_index in range(diag_dim):
                temp = self.multiply(ws[out_index, diag_index, 0], prepared[(self.g.pack * out_index) % in_dim])
                for in_index in range(1, in_dim):
                    wx = self.multiply(ws[out_index, diag_index, in_index],
                                       prepared[(self.g.pack * out_index + in_index) % in_dim])
                    self.add_inplace(temp, wx)
                output[out_index, diag_index] = self.rescale(temp)
        return output

    def rotate_internal(self, x, delta: int):
        """Rotate the used blocks of every token by ``delta``, wrapping within the ``n_blocks`` window.

        Costs exactly one level, which is what ``pcmm`` assumes when it drops the un-rotated term by one.
        """
        if delta == 0:
            return self.level_down(x, by=1)
        low, high = self.masks[delta], self.complement_masks[delta]
        rotated = self.add(self.rotate(self.multiply(high, x), -delta),
                           self.rotate(self.multiply(low, x), self.g.n_blocks - delta))
        return self.rescale(rotated)

    def pcmm(self, ws, xs):
        """Block-diagonal plaintext-ciphertext matrix product: the sum of the rotated partial products."""
        out_dim, diag_dim, _ = ws.shape
        submatrices = self.parallel_diagonal_pc_mult(ws, xs)
        output = np.full((out_dim,), None, dtype=object)
        for out_index in range(out_dim):
            temp = self.level_down(submatrices[out_index, 0], by=1)
            for diag_index in range(1, diag_dim):
                self.add_inplace(temp, self.rotate_internal(submatrices[out_index, diag_index],
                                                            self.g.n_blocks - diag_index))
            output[out_index] = temp
        return output

    def apply_qkv_weight_bias(self, x, weight, bias):
        """``pcmm`` then bias then ``y + conj(y)``, which makes the result real (THOR stages 03-05).

        The conjugate doubles the real part; ``encode_weight`` halves the weights to compensate but
        ``encode_bias`` does not, so the result is ``x @ w.T + 2 * b``. See docs/thor_port.md.
        """
        wx = self.pcmm(weight, x)
        output = np.full((wx.shape[0],), None, dtype=object)
        for i in range(wx.shape[0]):
            biased = self.add(bias[i], wx[i])
            output[i] = self.add(biased, self.conjugate(biased))
        return output

    def stage_03_query(self, x, weight, bias):
        return self.apply_qkv_weight_bias(x, weight, bias)

    def stage_04_key(self, x, weight, bias):
        return self.apply_qkv_weight_bias(x, weight, bias)

    def stage_05_value(self, x, weight, bias):
        return self.apply_qkv_weight_bias(x, weight, bias)

    # ---------------------------------------------------------------- key schedule
    def rotation_indexes(self) -> set[int]:
        """Every rotation index stages 01-05 ask the engine for, in the engine's own sign convention."""
        n = self.g.slot_count
        deltas = {self.g.n_blocks // 2, -self.g.group_size}
        for delta in range(1, self.g.n_blocks):
            deltas.add(-delta)
            deltas.add(self.g.n_blocks - delta)
        return {(-d) % n for d in deltas}
