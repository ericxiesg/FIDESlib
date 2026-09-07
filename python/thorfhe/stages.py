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
        #: content -> engine plaintext, so a mask is encoded once per engine rather than per multiply
        self._plaintexts: dict = {}

    # ---------------------------------------------------------------- primitives
    def rotate(self, x, delta: int):
        """THOR's rotation direction: slot ``i`` moves to slot ``i + delta``.

        A delta that is zero modulo the slot count is skipped entirely rather than handed to the
        engine. It is not just a wasted call: ``ClearEngine`` records every rotation it is asked for,
        so a zero would put index 0 into ``plan_rotation_keys``' output, and the GPU would then be
        asked to generate a rotation key that OpenFHE has no index for - a 123 MiB key for a no-op.
        """
        index = -int(delta) % self.g.slot_count
        if index == 0:
            return x
        return self.engine.rotate(x, index)

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
        return self.engine.multiply(x, self.plaintext(y))

    def plaintext(self, value):
        """Turn a mask into whatever the engine wants to multiply by, and remember it.

        Every mask in the port is a fixed numpy array - the ``rotate_internal`` families, LayerNorm's
        value and statistic masks, the feed-forward window, the pooler's CLS mask - and each one is
        multiplied into a ciphertext many times per layer. Handing the raw array to
        ``pyfideslib.Engine`` encodes a fresh full-tower plaintext on *every* call: a GV100 run of one
        layer reported 378 plaintexts holding 6.4 GB, which is most of the way to the card on its own.

        A light plaintext is the level-agnostic form (``docs/light_plaintext.md``) and the engine
        expands it through its own cache, so one encode serves every level. Keyed by content rather
        than identity because several masks are rebuilt per call site.

        ``ClearEngine`` has no such notion and wants the array itself, so this is a no-op there.
        """
        if not isinstance(value, np.ndarray) or not hasattr(self.engine, "encode_to_light_plaintext"):
            return value
        key = (value.shape, value.dtype.str, value.tobytes())
        cached = self._plaintexts.get(key)
        if cached is None:
            cached = self.engine.encode_to_light_plaintext(value)
            self._plaintexts[key] = cached
        return cached

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

    def align(self, *cts):
        """Drop every operand to the lowest level present, so they can be combined.

        FIXEDMANUAL will not add or multiply across levels, and operands that took different routes
        through a stage rarely arrive together. desilofhe aligns implicitly; here it is one call, at
        the point where the mismatch is meaningful.
        """
        target = min(self.engine.level(ct) for ct in cts)
        return tuple(ct if self.engine.level(ct) == target
                     else self.level_down(ct, self.engine.level(ct) - target) for ct in cts)

    def interval_sum(self, x, interval: int):
        """Fold the slot vector onto itself in steps of ``interval``, so every window holds the total.

        The workhorse reduction: it is how a value spread over ``pack`` groups is summed, and how a
        statistic is broadcast back. ``log2(slot_count / interval)`` rotations, no levels.
        """
        out = x
        for step in range(int(np.log2(self.g.slot_count / interval))):
            out = self.add(out, self.rotate(out, -interval * 2 ** step))
        return out

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

    def diagonal_product(self, ws, prepared, out_index: int, diag_index: int):
        """One block diagonal's plaintext-ciphertext inner product, over all the input copies."""
        in_dim = ws.shape[2]
        base = self.g.pack * out_index
        temp = self.multiply(ws[out_index, diag_index, 0], prepared[base % in_dim])
        for in_index in range(1, in_dim):
            self.add_inplace(temp, self.multiply(ws[out_index, diag_index, in_index],
                                                 prepared[(base + in_index) % in_dim]))
        return self.rescale(temp)

    def parallel_diagonal_pc_mult(self, ws, xs):
        """The whole ``(out_dim, diag_dim)`` grid of partial products.

        Kept for tests and for reading against ``he.py``; :meth:`pcmm` does not use it, because
        holding the grid is what makes a layer run out of GPU memory - see :meth:`pcmm`.
        """
        out_dim, diag_dim, _ = ws.shape
        prepared = [self.prepare_for_multiply(x) for x in np.asarray(xs).ravel()]
        output = np.full((out_dim, diag_dim), None, dtype=object)
        for out_index in range(out_dim):
            for diag_index in range(diag_dim):
                output[out_index, diag_index] = self.diagonal_product(ws, prepared, out_index,
                                                                      diag_index)
        return output

    def rotate_internal(self, x, delta: int, *, window=None, masks=None, complements=None):
        """Rotate the used blocks of every token by ``delta``, wrapping within ``window`` blocks.

        ``window`` defaults to the geometry's ``n_blocks`` (THOR's ``block_diag_1``); the feed-forward
        stages pass 6, with masks built on a stride of 8 (``block_diag_2``). Costs exactly one level,
        which is what ``pcmm`` assumes when it drops the un-rotated term by one.
        """
        if delta == 0:
            return self.level_down(x, by=1)
        window = self.g.n_blocks if window is None else window
        masks = self.masks if masks is None else masks
        complements = self.complement_masks if complements is None else complements
        rotated = self.add(self.rotate(self.multiply(complements[delta], x), -delta),
                           self.rotate(self.multiply(masks[delta], x), window - delta))
        return self.rescale(rotated)

    def pcmm(self, ws, xs, *, window=None, masks=None, complements=None):
        """Block-diagonal plaintext-ciphertext matrix product: the sum of the rotated partial products.

        Each output ciphertext needs only its own row of the ``(out_dim, diag_dim)`` grid, and needs it
        only long enough to rotate and accumulate, so the row is formed and folded one diagonal at a
        time. Building the whole grid first is the obvious transcription of ``he.py`` and it is what
        exhausts a 32 GB card: at N=2^16 and depth 48 a ciphertext is about 50 MB, and the grid is
        ``out_dim * diag_dim`` of them (48 for the QKV and feed-forward stages) where this holds two.
        """
        out_dim, diag_dim, _ = ws.shape
        window = self.g.n_blocks if window is None else window
        prepared = [self.prepare_for_multiply(x) for x in np.asarray(xs).ravel()]

        output = np.full((out_dim,), None, dtype=object)
        for out_index in range(out_dim):
            # the un-rotated term skips a rotate_internal, so it is levelled down by hand instead
            temp = self.level_down(self.diagonal_product(ws, prepared, out_index, 0), by=1)
            for diag_index in range(1, diag_dim):
                partial = self.diagonal_product(ws, prepared, out_index, diag_index)
                self.add_inplace(temp, self.rotate_internal(partial, window - diag_index,
                                                            window=window, masks=masks,
                                                            complements=complements))
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
