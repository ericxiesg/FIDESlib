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

    #: How rotations reach an index that has no key of its own. ``False`` gives every index its own
    #: key (210 for a layer, 28 GiB - it does not fit); ``True`` keeps only the powers of two, so an
    #: index costs about eight rotations; a :class:`~thorfhe.rotation.RotationBasis` keeps the powers
    #: of two *plus* a few measured indices, which halves the rotation count for six more keys. Off by
    #: default: it is a memory-for-time trade. See :meth:`rotation_steps` and :mod:`thorfhe.rotation`.
    binary_rotations = False

    #: Optional ``(name, ciphertexts) -> None`` callback for looking inside a stage. A stage's output
    #: is comparable against the plaintext model, but its intermediates are not, and when a stage is
    #: wrong the question is which half of it. A probe that only reports magnitudes needs no
    #: reference: a value that should be a probability and comes back at 1e12, or as a constant, says
    #: where to look. ``None`` everywhere unless a caller sets it, and never called in a normal run.
    probe = None

    def probed(self, name, value):
        """Report an intermediate if anyone is listening, and return it unchanged."""
        if self.probe is not None:
            self.probe(name, value)
        return value

    def __init__(self, engine, geometry: Geometry, masks=None, complement_masks=None,
                 binary_rotations: bool | None = None):
        self.engine = engine
        self.g = geometry
        if binary_rotations is not None:
            self.binary_rotations = binary_rotations
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

        With ``binary_rotations`` set, an index that has no key of its own is performed as the
        sequence of rotations that sums to it. See :attr:`binary_rotations`.
        """
        index = -int(delta) % self.g.slot_count
        if index == 0:
            return x
        if not self.binary_rotations:
            return self.engine.rotate(x, index)
        for step in self.rotation_steps(index):
            x = self.engine.rotate(x, step)
        return x

    def rotation_steps(self, index: int) -> list[int]:
        """``index`` as a sum of powers of two, which is what the binary rotation basis performs.

        Rotations compose additively modulo the slot count and cost no levels, so any index can be
        reached from the powers of two alone. That trades key *memory* for rotation *count*: one
        layer uses 210 distinct indices, which is 51 GiB of rotation keys at N=2^16 and depth 50 -
        more than a 32 GB card holds even fully truncated - while the powers of two are 29 keys and
        about 7 GiB. The cost is a rotation per set bit (about eight on average here) instead of one,
        and a key-switch's worth of noise with it.

        A :class:`~thorfhe.rotation.RotationBasis` is the same trade made less bluntly: it adds the
        handful of non-power-of-two indices the layer asks for most, which brings those eight steps
        down to two at the heaviest sites. The powers of two stay in it as the fallback.
        """
        basis = self.binary_rotations
        if not isinstance(basis, bool):
            # An explicit basis owns the decomposition, so the key plan derived from it and the run
            # that spends the keys cannot disagree about which keys exist - a key that was planned
            # but not built does not raise here, it surfaces much later as a wrong plaintext.
            return list(basis.steps(index))
        return [1 << bit for bit in range(index.bit_length()) if index >> bit & 1]

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

    def release_pooled_memory(self):
        """Hand back what the engine is holding for reuse. A no-op on engines without a device pool.

        Called at stage boundaries, where the working set has genuinely shrunk: mid-stage the pooled
        polynomials are about to be needed again, so draining then would only make the next allocation
        re-take them.
        """
        trim = getattr(self.engine, "trim_auxiliary_polys", None)
        if trim is not None:
            trim()

    def device_memory(self) -> dict:
        """What the device pool holds, or ``{}`` on an engine that has no device."""
        probe = getattr(self.engine, "device_memory", None)
        return probe() if probe is not None else {}

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

        The key is a *digest* of the content, not the content. Holding the bytes made every entry
        cost its array twice over - once for the light plaintext and once for the key - and a layer
        has 19,137 distinct arrays at half a MiB each, so the cache alone was 9.3 GiB of host memory
        per layer and nothing ever evicts it. A twelve-layer run would need 112 GiB for the keys.
        The digest is Python's own hash of the same bytes, so it costs what the old key cost (183 us
        against 221 us, measured on a 512 KiB array) and retains eight bytes instead of half a MiB.

        That trades an exact comparison for a 64-bit one. Over a layer's 19,137 entries the odds of
        two different arrays sharing a digest are about 1e-11, which is far below the other ways this
        run can go wrong - but it is not zero, and the way to stop relying on it is to hand encoded
        weights down from `encode_layer` (see `he.LightWeights`) so there is no content cache at all.

        ``ClearEngine`` has no such notion and wants the array itself, so this is a no-op there.
        """
        if not isinstance(value, np.ndarray) or not hasattr(self.engine, "encode_to_light_plaintext"):
            return value
        key = (value.shape, value.dtype.str, hash(value.tobytes()))
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

    #: Skip a bootstrap whose input already sits at or above the level a bootstrap would restore.
    #: Four of a layer's eighteen bootstraps are like that - they arrive 28 levels *above* it - so
    #: each costs a full bootstrap and hands back a ciphertext with fewer levels than it had.
    #: Measured: this does not lower the minimum depth, it only removes the waste, which is why it is
    #: opt-in. The tradeoff is noise - a bootstrap refreshes that too - and this trusts that a
    #: ciphertext which has consumed few levels has little noise to refresh.
    skip_pointless_bootstraps = False

    def bootstrap(self, x, keep_levels=None):
        if (self.skip_pointless_bootstraps and keep_levels is None
                and getattr(self.engine, "bootstrap_level", None) is not None
                and self.engine.level(x) >= self.engine.bootstrap_level):
            return x
        return self.engine.bootstrap(x, keep_levels)

    #: How far `bootstrap_twice` lifts the residual before refreshing it. The gain in accuracy is
    #: this many bits, and the ceiling is the message bound: `2^bits * |error|` has to stay inside it,
    #: so at a measured error of 2.6e-05 against `q0/Delta = 2` anything up to 2^15 is in range.
    meta_bts_bits = 10

    def bootstrap_twice(self, x, keep_levels=None):
        """Meta-BTS: refresh, refresh the residual error, and add it back scaled down.

        `b0 = BTS(x)` leaves `x + e`. Reducing `b0` to the input's modulus and subtracting gives
        `-e`, which is small enough that lifting it by an integer `2^k` - level-free - puts it back
        in a range the bootstrap handles well. Refreshing that gives `-2^k e + e'`, and scaling by
        `2^-k` and adding to `b0` returns `x + 2^-k e'`: the same message with the error `k` bits
        smaller.

        Two bootstraps and one level for `k` bits. This is what thor-openfhe gets from OpenFHE's
        `EvalBootstrap(ct, numIterations=2, precision=10)`; FIDESlib's GPU path takes those two
        arguments and forwards them only on its CPU fallback (`CryptoContext.cpp:1763-1778`), and
        `FIDESlib::CKKS::Bootstrap` has no such parameter - so on the device it is a no-op, and the
        iteration has to be built out of primitives instead. All of them exist.
        """
        first = self.bootstrap(x, keep_levels)
        drop = self.engine.level(first) - self.engine.level(x)
        at_input = self.engine.level_down(first, by=drop) if drop > 0 else first
        residual = self.subtract(x, at_input)
        lifted = self.multiply(residual, 2 ** self.meta_bts_bits)      # integer: no level, no degree
        second = self.bootstrap(lifted, keep_levels)
        corrected = self.rescale(self.multiply(second, 2.0 ** -self.meta_bts_bits))
        left, right = self.align(first, corrected)
        return self.add(left, right)

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

    def iter_rotated_copies(self, x):
        """The same copies as :meth:`stage_02_make_rotated_copies`, yielded as ``(index, ciphertext)``.

        The array form is ``pack * len(x)`` ciphertexts - 64 at THOR's geometry, near full level, and
        the largest single item in a layer's working set. They are built all at once and then read one
        at a time, so it is holding them that costs, not making them. Each copy is one rotation of the
        one before it, so a generator holds one per source instead of sixteen.
        """
        for source in range(x.shape[0]):
            base = self.g.pack * source
            current = x[source]
            yield base, current
            for r in range(1, self.g.pack):
                current = self.rotate(current, -self.g.group_size)
                yield base + r, current

    def pcmm_streamed(self, ws, copies, *, window=None, masks=None, complements=None):
        """:meth:`pcmm` over copies taken as they are produced rather than out of a filled array.

        Same arithmetic, same result, different thing held. `pcmm` holds all ``in_dim`` copies and
        forms one ``(out, diag)`` sum at a time; this holds one copy and all ``out_dim * diag_dim``
        sums - 64 against 24 at THOR's QKV geometry, both near full level.

        The price is that the copies cannot be shared between the three projections, since each
        consumes the stream. Stages 03/04/05 therefore make them three times: 180 rotations instead
        of 60, against a layer's 8258.
        """
        out_dim, diag_dim, in_dim = ws.shape
        window = self.g.n_blocks if window is None else window

        partials = np.full((out_dim, diag_dim), None, dtype=object)
        for index, copy in copies:
            prepared = self.prepare_for_multiply(copy)
            for out_index in range(out_dim):
                # `pcmm` reads `prepared[(pack * out_index + j) % in_dim]` for its j-th term, so the
                # copy that has just arrived is the j-th term of this row for exactly one j
                j = (index - self.g.pack * out_index) % in_dim
                for diag_index in range(diag_dim):
                    term = self.multiply(ws[out_index, diag_index, j], prepared)
                    if partials[out_index, diag_index] is None:
                        partials[out_index, diag_index] = term
                    else:
                        self.add_inplace(partials[out_index, diag_index], term)

        output = np.full((out_dim,), None, dtype=object)
        for out_index in range(out_dim):
            temp = self.level_down(self.rescale(partials[out_index, 0]), by=1)
            for diag_index in range(1, diag_dim):
                partial = self.rescale(partials[out_index, diag_index])
                self.add_inplace(temp, self.rotate_internal(partial, window - diag_index,
                                                            window=window, masks=masks,
                                                            complements=complements))
            output[out_index] = temp
        return output

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

    #: Stages 03-05 make their own rotated copies and stream them, instead of reading an array the
    #: caller built. Measured at THOR's geometry, depth 37, one projection: peak 84 ciphertexts and
    #: 2743 MiB held against **50 and 1652** - 1.07 GiB, which is the margin `--extra-rotation-keys`
    #: needs (headroom 2.7 GiB against the 3 GiB key generation wants for its scratch).
    #:
    #: Costs the copies three times over rather than once, since a stream cannot be shared: 180
    #: rotations instead of 60, against a layer's 8258.
    stream_qkv = False

    def apply_qkv_weight_bias(self, x, weight, bias):
        """``pcmm`` then bias then ``y + conj(y)``, which makes the result real (THOR stages 03-05).

        The conjugate doubles the real part; ``encode_weight`` halves the weights to compensate but
        ``encode_bias`` does not, so the result is ``x @ w.T + 2 * b``. See docs/thor_port.md.

        With ``stream_qkv`` set, ``x`` is the *complexified* input and the copies are made here; the
        result is identical either way (`test_streaming_the_copies_computes_the_same_projection`).
        """
        wx = (self.pcmm_streamed(weight, self.iter_rotated_copies(x)) if self.stream_qkv
              else self.pcmm(weight, x))
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
