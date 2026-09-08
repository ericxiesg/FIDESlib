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

import numpy as np

from .geometry import Geometry


class ScaleMismatch(AssertionError):
    """Raised when two operands of an add/subtract are not at the same level and scale."""


class ClearCiphertext:
    """Slots, level (remaining multiplications) and scale exponent (1 = canonical, 2 = needs rescale)."""

    __slots__ = ("slots", "level", "scale_exp")

    def __init__(self, slots: np.ndarray, level: int, scale_exp: int = 1):
        self.slots = np.asarray(slots, dtype=complex)
        self.level = level
        self.scale_exp = scale_exp

    def __repr__(self):
        return f"ClearCiphertext(level={self.level}, scale=D^{self.scale_exp}, slots={self.slots.shape})"


class ClearEngine:
    """Exact, plaintext mirror of the primitives :mod:`thorfhe.stages` uses."""

    def __init__(self, geometry: Geometry, depth: int, strict: bool = True, bootstrap_level: int = 14):
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

    # ---- encoding ----
    def encrypt(self, message, level: int = 0) -> ClearCiphertext:
        msg = np.zeros((self.slots,), dtype=complex)
        msg[: len(message)] = message
        return ClearCiphertext(msg, self.depth - level)

    def decrypt(self, ct: ClearCiphertext) -> np.ndarray:
        return ct.slots.copy()

    def decrypt_real(self, ct: ClearCiphertext) -> np.ndarray:
        return np.real(ct.slots)

    def level(self, ct: ClearCiphertext) -> int:
        return ct.level

    # ---- operand helpers ----
    @staticmethod
    def _is_ct(v) -> bool:
        return isinstance(v, ClearCiphertext)

    @staticmethod
    def _slots_of(v):
        return v.slots if isinstance(v, ClearCiphertext) else np.asarray(v)

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
        return ClearCiphertext(op(self._slots_of(x), self._slots_of(y)), level, scale)

    # ---- arithmetic ----
    def add(self, x, y):
        return self._binary(lambda a, b: a + b, x, y, "add")

    def add_inplace(self, x: ClearCiphertext, y):
        out = self.add(x, y)
        x.slots, x.level, x.scale_exp = out.slots, out.level, out.scale_exp
        return x

    def subtract(self, x, y):
        return self._binary(lambda a, b: a - b, x, y, "subtract")

    def multiply(self, x, y):
        """Ciphertext times ciphertext / plaintext / scalar. Integer scalars are level- and scale-free."""
        if isinstance(y, (int, np.integer)) and not isinstance(y, bool):
            return ClearCiphertext(x.slots * y, x.level, x.scale_exp)
        if isinstance(x, (int, np.integer)) and not isinstance(x, bool):
            return ClearCiphertext(y.slots * x, y.level, y.scale_exp)

        if self._is_ct(x) and self._is_ct(y):
            if self.strict and x.level != y.level:
                raise ScaleMismatch(f"multiply: levels differ, {x.level} vs {y.level}")
            level = min(x.level, y.level)
            scale = x.scale_exp + y.scale_exp
        else:
            ct = x if self._is_ct(x) else y
            level, scale = ct.level, ct.scale_exp + 1
        return ClearCiphertext(self._slots_of(x) * self._slots_of(y), level, scale)

    def conjugate(self, ct: ClearCiphertext) -> ClearCiphertext:
        return ClearCiphertext(np.conj(ct.slots), ct.level, ct.scale_exp)

    def multiply_1j(self, ct: ClearCiphertext) -> ClearCiphertext:
        return ClearCiphertext(1j * ct.slots, ct.level, ct.scale_exp)

    def rotate(self, ct: ClearCiphertext, delta: int) -> ClearCiphertext:
        """``out[i] = ct[(i + delta) mod slots]`` - the fideslib direction, not THOR's.

        THOR's ``he.rotate`` shifts the other way; :class:`thorfhe.stages.Stages` negates once so that
        every delta in the stage code reads exactly as it does in ``he.py``.
        """
        delta %= self.slots
        self.rotations_used.add(delta)
        self.rotation_levels[delta] = max(self.rotation_levels.get(delta, -1), ct.level)
        return ClearCiphertext(np.roll(ct.slots, -delta), ct.level, ct.scale_exp)

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
        return ClearCiphertext(ct.slots.copy(), level, scale_exp)

    def rescale(self, ct: ClearCiphertext) -> ClearCiphertext:
        if self.strict and ct.scale_exp < 2:
            raise ScaleMismatch("rescale: ciphertext is already canonical (scale D^1); "
                                "rescaling it would leave the scale below Delta")
        return self._spend(ct, ct.level - 1, ct.scale_exp - 1, "rescale")

    def level_down(self, ct: ClearCiphertext, by: int) -> ClearCiphertext:
        return self._spend(ct, ct.level - by, ct.scale_exp, f"level_down(by={by})")

    def square(self, ct: ClearCiphertext) -> ClearCiphertext:
        return self.multiply(ct, ct)

    def bootstrap(self, ct: ClearCiphertext, keep_levels: int | None = None) -> ClearCiphertext:
        """Refresh to ``bootstrap_level``: exact here, so a stage's *schedule* is tested, not its noise."""
        if self.strict and ct.scale_exp != 1:
            raise ScaleMismatch("bootstrap: ciphertext must be canonical (scale D^1)")
        level = self.bootstrap_level if keep_levels is None else keep_levels
        return ClearCiphertext(ct.slots.copy(), level, 1)

    def relinearize(self, ct: ClearCiphertext) -> ClearCiphertext:
        """No-op: ciphertext degree is not modelled here, only the values, levels and scales."""
        return ct

    def ntt(self, ct):
        return ct

    def intt(self, ct):
        return ct
