"""A numpy engine with the same primitive surface as the FHE one.

Stages are written once against this surface (see :mod:`thorfhe.stages`) and run on either backend, so
"does the FHE port compute what THOR computes" reduces to comparing two runs of the same code.

A "ciphertext" here carries its level and its **scale exponent** as well as its slots, and the engine
refuses to add or subtract operands whose level or scale disagree. That is the FIXEDMANUAL contract
the fideslib engine runs under: a ciphertext is canonical at scale ``Delta``, a product sits at
``Delta^2`` until it is rescaled, and mixing the two silently produces a wrong plaintext rather than an
error. Checking it here means the level schedule is validated in milliseconds on a laptop instead of
being debugged through decryption noise on a GPU.
"""
from __future__ import annotations

import collections

import numpy as np

from .geometry import Geometry


class ScaleMismatch(AssertionError):
    """Raised when two operands of an add/subtract are not at the same level and scale."""


class ClearCiphertext:
    """Slots, level (remaining multiplications) and scale exponent (1 = canonical, 2 = needs rescale)."""

    # `__weakref__` is here so a ciphertext can be tracked without being kept alive. That is what
    # `thorfhe.workingset` needs to measure a stage's peak, and measuring it is the only way to know
    # which stage actually dominates GPU memory rather than which one looks like it should.
    __slots__ = ("slots", "level", "scale_exp", "degree", "__weakref__")

    def __init__(self, slots: np.ndarray, level: int, scale_exp: int = 1, degree: int = 1):
        self.slots = np.asarray(slots, dtype=complex)
        self.level = level
        self.scale_exp = scale_exp
        #: 2 between a lazy multiplication and its relinearisation. Exact arithmetic does not care,
        #: but almost nothing on the device accepts a degree-2 operand, so the contract is checked.
        self.degree = degree

    def __repr__(self):
        return (f"ClearCiphertext(level={self.level}, scale=D^{self.scale_exp}, "
                f"degree={self.degree}, slots={self.slots.shape})")


class ClearEngine:
    """Exact, plaintext mirror of the primitives :mod:`thorfhe.stages` uses."""

    #: Values can be read back without a key here, so a stage may check a precondition it can only
    #: state - not verify - on the device. :meth:`thorfhe.numeric.DivisionMixin.he_inv` uses this to
    #: enforce the range its iteration is set up for, which is otherwise silent: outside that range
    #: Goldschmidt does not fail, it saturates and returns a plausible wrong number.
    inspectable = True

    def __init__(self, geometry: Geometry, depth: int, strict: bool = True, bootstrap_level: int = 14,
                 noise_model: bool = False, scaling_bits: int = 50, first_mod_bits: int = 55,
                 bootstrap_precision_bits: int = 22, seed: int = 0):
        self.geometry = geometry
        self.slots = geometry.slot_count
        self.depth = depth
        #: level a bootstrap refreshes to; THOR assumes 14 (`use_bootstrap_to_14_levels`).
        self.bootstrap_level = bootstrap_level
        #: when set, add/subtract enforce matching level and scale (the FIXEDMANUAL contract).
        self.strict = strict
        #: rotation indexes actually requested, so a run can report the key set it needs.
        self.rotations_used: set[int] = set()
        #: rotation index -> highest ciphertext level it was applied at. This is THOR's
        #: ``rotation_contexts`` table, derived from a dry run instead of transcribed by hand;
        #: it is exactly what ``SetRotationKeyLevels`` wants (see ``thorfhe.he.plan_rotation_keys``).
        self.rotation_levels: dict[int, int] = {}
        #: rotation index -> how many times it was applied. The levels alone say which keys are
        #: needed; the counts say which ones are worth spending a key on, which is what
        #: ``thorfhe.rotation.factored_basis`` chooses from.
        self.rotation_counts: collections.Counter = collections.Counter()

        #: Perturb results by an amount of the right *order of magnitude*, so that a chain whose
        #: correctness depends on precision - rather than on the level and scale schedule - can fail
        #: here instead of on the device. Off by default: with it on this engine is no longer an
        #: exact oracle, so it cannot be compared against the plaintext model to high accuracy.
        #: See :meth:`_perturb`.
        self.noise_model = noise_model
        self.scaling_bits = scaling_bits
        self.first_mod_bits = first_mod_bits
        #: `q0 / Delta`, the ratio the bootstrap's modular reduction is periodic in.
        self.message_bound = 2.0 ** (first_mod_bits - scaling_bits)
        #: What a message actually has to stay inside, which is **half** of that. ModRaise leaves
        #: `m + q0 * I`; the sine recovers `m` from it only while `|m| < q0/2`, so in message units
        #: the limit is `q0 / (2 * Delta)`. Past it the value is not refreshed inaccurately, it is
        #: replaced by its residue - a different number entirely.
        #:
        #: This was `message_bound` until the GPU's sb=59 run: at `q0/Delta = 32` every site sits at
        #: a few percent of the limit and the distinction never showed, but at `q0/Delta = 2` the
        #: limit is 1.0 and the 22-site table's worst entry is 0.971 - which reads as "under the
        #: bound" and is really 97% of what the sine can recover.
        self.recoverable_bound = self.message_bound / 2
        #: Bits of the *bound* a bootstrap reproduces. A bootstrap's error is set by `message_bound`,
        #: not by the value it is given, so bootstrapping a small number is much less accurate in
        #: relative terms than bootstrapping a large one - which is the whole reason THOR keeps
        #: magnitudes up (`numeric._restore_magnitude`).
        self.bootstrap_precision_bits = bootstrap_precision_bits
        #: Fraction of ``message_bound`` a bootstrap input may reach. One by default - the hard bound,
        #: so only a real violation fails. It is a knob because the sine approximation degrades well
        #: before the bound rather than at it, and how much margin THOR's schedule actually leaves is
        #: a measurement, not a constant: stage 07 bootstraps the doubled scores, and on the
        #: `peaked_input` test those reach 20.4 of 32 - 64% of the bound, with nothing saying so.
        self.bootstrap_message_margin = 1.0
        self._rng = np.random.default_rng(seed)

    # ---- noise ----
    #: Absolute error each source adds to a slot, in the message domain, as a power of two relative
    #: to the scaling factor. These are orders of magnitude, not a CKKS noise analysis: what they are
    #: for is to tell apart a chain that survives finite precision from one that does not, and the
    #: two differ by far more than a factor of two. A canonical-embedding error of `e` coefficients
    #: spreads over `sqrt(N)` slots, which is where the 8 bits at N=2^16 come from.
    #:
    #: The one that matters is the bootstrap, and it is larger than the rest by seven orders: its
    #: error scales with `message_bound`, so it is the same absolute size whatever it is handed.
    FRESH_BITS_BELOW_SCALE = 42        # encode rounding + encryption noise
    RESCALE_BITS_BELOW_SCALE = 42      # the rounding a rescale leaves behind
    KEYSWITCH_BITS_BELOW_SCALE = 40    # one rotation, conjugation or relinearisation

    def _perturb(self, slots: np.ndarray, sigma: float) -> np.ndarray:
        """Add complex Gaussian noise of scale ``sigma`` to every slot."""
        if not self.noise_model or sigma <= 0:
            return slots
        shape = slots.shape
        return slots + sigma * (self._rng.normal(0, 1, shape) + 1j * self._rng.normal(0, 1, shape))

    def _sigma(self, bits_below_scale: int) -> float:
        return 2.0 ** -bits_below_scale

    @property
    def _bootstrap_sigma(self) -> float:
        """A bootstrap reproduces ``message_bound`` to ``bootstrap_precision_bits``."""
        return self.message_bound * 2.0 ** -self.bootstrap_precision_bits

    # ---- encoding ----
    def encrypt(self, message, level: int = 0) -> ClearCiphertext:
        msg = np.zeros((self.slots,), dtype=complex)
        msg[: len(message)] = message
        return ClearCiphertext(self._perturb(msg, self._sigma(self.FRESH_BITS_BELOW_SCALE)),
                               self.depth - level)

    def decrypt(self, ct: ClearCiphertext) -> np.ndarray:
        return ct.slots.copy()

    def decrypt_real(self, ct: ClearCiphertext) -> np.ndarray:
        return np.real(ct.slots)

    def noise_level(self, ct: ClearCiphertext) -> int:
        """Scale degree, mirroring ``pyfideslib.Engine.noise_level``."""
        return ct.scale_exp

    def level(self, ct: ClearCiphertext) -> int:
        return ct.level

    # ---- operand helpers ----
    @staticmethod
    def _is_ct(v) -> bool:
        return isinstance(v, ClearCiphertext)

    @staticmethod
    def _slots_of(v):
        return v.slots if isinstance(v, ClearCiphertext) else np.asarray(v)

    def _require_degree_one(self, ct, what: str):
        """Refuse an operand the device would silently mishandle.

        A lazy product leaves a third component behind, and only some operations carry it: add, sub,
        plaintext and scalar multiplication, rescale, level reduction and copy do; a
        ciphertext-ciphertext product, a monomial multiply, an integer multiply and every key switch
        do not. They do not fail either - they drop it - so a missing relinearise reads as a plausible
        wrong answer many stages downstream. That is exactly how `power_basis` squared without
        relinearising and made `he_exp` return 1e124 on the device while exact arithmetic, which has
        no third component to lose, stayed happy.
        """
        if self.strict and self._is_ct(ct) and ct.degree != 1:
            raise ScaleMismatch(
                f"{what}: the ciphertext is degree {ct.degree}; relinearize it first. "
                "A lazy product has to be relinearised before anything that key-switches it, "
                "multiplies it by another ciphertext, or multiplies it by a monomial or an integer.")

    def _require_canonical_scale(self, ct, what: str):
        """Refuse a multiplication whose operand is not at scale Delta.

        The device asserts this - `Ciphertext::mult` has `assert(NoiseLevel == 1)` for both operands
        (Ciphertext.cpp:551,563), `multPt` has `assert(NoiseLevel < 2)` (:478) and `multScalar` has
        `assert(this->NoiseLevel == 1)` (:913), all of them outside the FIXEDAUTO/FLEXIBLE branches
        and so reached under FIXEDMANUAL too. But they are `assert`, so a release build drops them,
        and the metadata could not represent the result anyway: `ScalingFactorReal` and
        `ScalingFactorRealBig` hold the scale factors for NoiseLevel 1 and 2, and nothing holds
        Delta^3. A product of two unrescaled operands carries a NoiseFactor matching no entry, and
        the error surfaces at a later addPt, or at decryption, as a value off by a factor of Delta -
        never as a failure where it was made.

        The integer multiply is deliberately not covered: `multIntScalar` touches no metadata
        (ApproxModEval.cu:131-136) and is legal on an unrescaled ciphertext, which is what
        `numeric._restore_magnitude` relies on.

        The float scalar multiply *is* covered, and the reason is worth recording because it looks
        like it should not be: `multScalarNoPrecheck` does scale c2 along with c0 and c1, so a float
        multiply is legal on a **degree**-2 ciphertext. That is a different question from the
        **scale**. `EvalMult(ct, double)` reaches `Ciphertext::multScalar` (CryptoContext.cpp:1298),
        which carries `assert(this->NoiseLevel == 1)` at Ciphertext.cpp:913 - outside the
        FLEXIBLE/FIXEDAUTO branch, so under FIXEDMANUAL too - and then adds one to NoiseLevel and
        multiplies NoiseFactor by `ScalingFactorReal`, the factor for degree 1. On an operand at
        Delta^2 the metadata comes out describing a scale the ciphertext does not have.
        """
        if self.strict and self._is_ct(ct) and ct.scale_exp != 1:
            raise ScaleMismatch(
                f"{what}: the operand is at scale D^{ct.scale_exp}, not canonical D^1. Rescale it "
                f"first. The device asserts this, but only in a debug build, and its metadata has no "
                f"scaling factor for D^{ct.scale_exp + 1} - the product would decrypt to a value off "
                f"by a factor of Delta rather than fail.")

    def _binary(self, op, x, y, what: str):
        """Add/subtract: a plaintext operand is canonical by definition, ciphertexts must agree."""
        if self._is_ct(x) and self._is_ct(y):
            if self.strict and (x.level != y.level or x.scale_exp != y.scale_exp):
                raise ScaleMismatch(
                    f"{what}: operands differ - level {x.level} scale D^{x.scale_exp} vs "
                    f"level {y.level} scale D^{y.scale_exp}")
            level, scale = min(x.level, y.level), x.scale_exp
        else:
            ct = x if self._is_ct(x) else y
            if self.strict and ct.scale_exp != 1:
                raise ScaleMismatch(
                    f"{what}: ciphertext is at scale D^{ct.scale_exp}, a fresh plaintext is at D^1; "
                    "rescale the ciphertext first")
            level, scale = ct.level, ct.scale_exp
        degree = max(x.degree if self._is_ct(x) else 1, y.degree if self._is_ct(y) else 1)
        return ClearCiphertext(op(self._slots_of(x), self._slots_of(y)), level, scale, degree)

    # ---- arithmetic ----
    def add(self, x, y):
        return self._binary(lambda a, b: a + b, x, y, "add")

    def add_inplace(self, x: ClearCiphertext, y):
        out = self.add(x, y)
        # `degree` too, and it is the one that used to be dropped. `_binary` computes
        # max(x.degree, y.degree) because the device's add builds a c2 when either side has one;
        # writing back everything except the degree left `x` claiming to be degree 1 while carrying a
        # third component, and rotate/multiply_1j/bootstrap - precisely the operations that discard
        # c2 silently - were then waved through. That is the 1e124 failure with its guard removed.
        x.slots, x.level, x.scale_exp, x.degree = out.slots, out.level, out.scale_exp, out.degree
        return x

    def subtract(self, x, y):
        return self._binary(lambda a, b: a - b, x, y, "subtract")

    def multiply(self, x, y):
        """Ciphertext times ciphertext / plaintext / scalar. Integer scalars are level- and scale-free."""
        if isinstance(y, (int, np.integer)) and not isinstance(y, bool):
            self._require_degree_one(x, "multiply by an integer")
            return ClearCiphertext(x.slots * y, x.level, x.scale_exp, x.degree)
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            self._require_degree_one(y, "multiply by an integer")
            return ClearCiphertext(y.slots * x, y.level, y.scale_exp, y.degree)

        if self._is_ct(x) and self._is_ct(y):
            if self.strict and x.level != y.level:
                raise ScaleMismatch(f"multiply: levels differ, {x.level} vs {y.level}")
            self._require_canonical_scale(x, "multiply a ciphertext by a ciphertext")
            self._require_canonical_scale(y, "multiply a ciphertext by a ciphertext")
            # The device's ciphertext-ciphertext product forms (c0*d0, c0*d1 + c1*d0, c1*d1) and has
            # nowhere to put a third input component, so it ignores one silently.
            self._require_degree_one(x, "multiply a ciphertext by a ciphertext")
            self._require_degree_one(y, "multiply a ciphertext by a ciphertext")
            level = min(x.level, y.level)
            scale = x.scale_exp + y.scale_exp
            degree = 2
        else:
            ct = x if self._is_ct(x) else y
            self._require_canonical_scale(ct, "multiply a ciphertext by a plaintext or float scalar")
            level, scale, degree = ct.level, ct.scale_exp + 1, ct.degree
        return ClearCiphertext(self._slots_of(x) * self._slots_of(y), level, scale, degree)

    def conjugate(self, ct: ClearCiphertext) -> ClearCiphertext:
        self._require_degree_one(ct, "conjugate")   # an automorphism, so a key switch
        return ClearCiphertext(
            self._perturb(np.conj(ct.slots), self._sigma(self.KEYSWITCH_BITS_BELOW_SCALE)),
            ct.level, ct.scale_exp)

    def multiply_1j(self, ct: ClearCiphertext) -> ClearCiphertext:
        # A monomial multiply, and the device's does not touch the third component.
        self._require_degree_one(ct, "multiply by i")
        return ClearCiphertext(1j * ct.slots, ct.level, ct.scale_exp)

    def rotate(self, ct: ClearCiphertext, delta: int) -> ClearCiphertext:
        self._require_degree_one(ct, "rotate")      # an automorphism, so a key switch
        """``out[i] = ct[(i + delta) mod slots]`` - the fideslib direction, not THOR's.

        THOR's ``he.rotate`` shifts the other way; :class:`thorfhe.stages.Stages` negates once so that
        every delta in the stage code reads exactly as it does in ``he.py``.
        """
        delta %= self.slots
        self.rotations_used.add(delta)
        self.rotation_levels[delta] = max(self.rotation_levels.get(delta, -1), ct.level)
        self.rotation_counts[delta] += 1
        return ClearCiphertext(
            self._perturb(np.roll(ct.slots, -delta), self._sigma(self.KEYSWITCH_BITS_BELOW_SCALE)),
            ct.level, ct.scale_exp)

    # ---- level management ----
    def _spend(self, ct: ClearCiphertext, level: int, scale_exp: int, what: str) -> ClearCiphertext:
        """Build the result of an operation that consumes levels, refusing to go below zero.

        A ciphertext at a negative level has no moduli left and means the level budget does not fit -
        on hardware it is a crash or garbage, so it is an error here. This is the only sound place to
        catch it: ``plan_rotation_keys`` used to infer starvation from negative *rotation* levels,
        which silently stops working as soon as two rotations share an index (as they all do under
        ``binary_rotations``), because the plan keeps the maximum level per index.
        """
        if self.strict and level < 0:
            raise ScaleMismatch(
                f"{what}: would leave the ciphertext at level {level}. The level budget does not fit "
                f"- raise the depth, or the bootstrap level if this is after a bootstrap.")
        return ClearCiphertext(ct.slots.copy(), level, scale_exp, ct.degree)

    def rescale(self, ct: ClearCiphertext) -> ClearCiphertext:
        if self.strict and ct.scale_exp < 2:
            raise ScaleMismatch("rescale: ciphertext is already canonical (scale D^1); "
                                "rescaling it would leave the scale below Delta")
        out = self._spend(ct, ct.level - 1, ct.scale_exp - 1, "rescale")
        out.slots = self._perturb(out.slots, self._sigma(self.RESCALE_BITS_BELOW_SCALE))
        return out

    def level_down(self, ct: ClearCiphertext, by: int) -> ClearCiphertext:
        return self._spend(ct, ct.level - by, ct.scale_exp, f"level_down(by={by})")

    def square(self, ct: ClearCiphertext) -> ClearCiphertext:
        return self.multiply(ct, ct)

    def bootstrap(self, ct: ClearCiphertext, keep_levels: int | None = None) -> ClearCiphertext:
        """Refresh to ``bootstrap_level``: exact here, so a stage's *schedule* is tested, not its noise."""
        if self.strict and ct.scale_exp != 1:
            raise ScaleMismatch("bootstrap: ciphertext must be canonical (scale D^1)")
        self._require_degree_one(ct, "bootstrap")
        peak = float(np.max(np.abs(ct.slots)))
        if self.strict and peak > self.recoverable_bound * self.bootstrap_message_margin:
            raise ScaleMismatch(
                f"bootstrap: the message reaches {peak:.4g}, which is "
                f"{peak / self.recoverable_bound:.0%} of q0/(2*Delta) = {self.recoverable_bound:.4g} "
                f"(q0/Delta is {self.message_bound:.0f}). The "
                f"bootstrap raises the modulus to q0 and recovers the message with a sine that only "
                f"approximates the modular reduction near zero, so a message at the bound is not "
                f"refreshed, it is replaced. Halve it first.")
        if keep_levels is not None and self.strict and keep_levels > self.bootstrap_level:
            raise ScaleMismatch(
                f"bootstrap: keep_levels={keep_levels} is above the {self.bootstrap_level} this "
                f"bootstrap restores to. Engine.bootstrap raises on the shortfall rather than "
                f"handing back fewer levels than asked for, so a schedule that does this does not "
                f"run - it only looks like it does here.")
        level = self.bootstrap_level if keep_levels is None else keep_levels
        # Above the recoverable bound the sine does not refresh the message, it replaces it by its
        # residue - the modular reduction it approximates is periodic in `message_bound`. Modelling
        # that is what lets this engine produce the *catastrophic* failure rather than only the
        # gradual one: a Gaussian perturbation can never turn a denominator negative, so a model
        # without this cannot be used to rule a wrap out, which is exactly how sb=59 was called
        # sufficient here and then exploded on the device.
        #
        # Per slot rather than per coefficient, which is where the real reduction happens: a slot is
        # a linear combination of coefficients, so this is a model of the effect and not of the
        # mechanism. It is right about when a value stops being recoverable, not about the precise
        # garbage that replaces it.
        slots = ct.slots
        if self.noise_model:
            period = self.message_bound
            fold = lambda part: part - period * np.round(part / period)      # noqa: E731
            slots = fold(np.real(slots)) + 1j * fold(np.imag(slots))

        # The error a bootstrap leaves is set by `message_bound`, not by the value: refreshing a
        # number much smaller than the bound costs most of its significant digits. That is invisible
        # in exact arithmetic and it is what `numeric._restore_magnitude` exists to avoid.
        return ClearCiphertext(self._perturb(np.asarray(slots).copy(), self._bootstrap_sigma),
                               level, 1)

    def relinearize(self, ct: ClearCiphertext) -> ClearCiphertext:
        """Fold the third component away. Exact in value; what it changes here is the degree."""
        # Copy, like every other operation here. Sharing the array made the result alias its input,
        # which is harmless while `add_inplace` rebinds rather than writes through - but that is a
        # property of another method, not of this one.
        return ClearCiphertext(
            self._perturb(ct.slots.copy(), self._sigma(self.KEYSWITCH_BITS_BELOW_SCALE)),
            ct.level, ct.scale_exp, 1)

    def ntt(self, ct):
        return ct

    def intt(self, ct):
        return ct
