"""How many ciphertexts a stage keeps alive at once, which is what decides whether it fits on a card.

The GPU runs have been a sequence of out-of-memory failures, and every reduction so far was found by
*reading* the code - `pcmm` holding its whole grid, `_accumulate_product` holding all its diagonals.
That finds real peaks but not necessarily the largest one. This measures it instead.

A ciphertext's cost is its RNS limbs: ``2 * (level + 1)`` polynomials of ``N`` 8-byte coefficients, or
three of them while it is still degree 2. At N=2^16 and level 37 that is about 38 MiB, so a stage
holding 128 of them is 4.9 GiB - against the 3.7 GiB a 32 GiB card has left after key generation.

Nothing here touches the engine's arithmetic; it only watches object lifetimes, so a measured run and
a normal one compute the same thing.
"""
from __future__ import annotations

import weakref
from dataclasses import dataclass, field

from .clear import ClearCiphertext, ClearEngine


def ciphertext_bytes(level: int, slot_count: int, *, degree: int = 2) -> int:
    """Device bytes one ciphertext occupies at ``level``. ``N = 2 * slot_count`` for a complex packing."""
    return degree * (level + 1) * (2 * slot_count) * 8


@dataclass
class Peak:
    """The high-water mark of live ciphertexts, and where it happened.

    ``nbytes`` is what the mark is ranked on, and it is not interchangeable with ``count``: a
    ciphertext at level 36 costs six times one at level 5, so the moment a stage holds the most
    ciphertexts is generally not the moment it holds the most memory. Ranking by count and then
    pricing the result was how both this file and the remote's estimate first got the order wrong.
    """

    count: int = 0
    where: str = ""
    levels: tuple = ()
    nbytes: int = 0


@dataclass
class WorkingSet:
    """Tracks live ciphertexts by weak reference, so watching costs nothing but a pointer each."""

    engine: ClearEngine
    slot_count: int = 0
    live: weakref.WeakSet = field(default_factory=weakref.WeakSet)
    label: str = "start"
    peak: Peak = field(default_factory=Peak)
    per_label: dict = field(default_factory=dict)

    def note(self, ct):
        """Register a ciphertext and update the high-water marks, overall and for this stage."""
        self.live.add(ct)
        levels = tuple(c.level for c in self.live)
        nbytes = sum(ciphertext_bytes(level, self.slot_count) for level in levels)
        if nbytes > self.peak.nbytes:
            self.peak = Peak(len(levels), self.label, levels, nbytes)
        best = self.per_label.get(self.label)
        if best is None or nbytes > best.nbytes:
            self.per_label[self.label] = Peak(len(levels), self.label, levels, nbytes)
        return ct

    def at(self, label: str):
        self.label = label

    def report(self, slot_count: int, level: int) -> str:
        """One line per stage, largest first.

        Two figures, and the difference between them is the point: ``actual`` prices every live
        ciphertext at the level it is really at, ``if full`` prices them all at ``level``. Deep in a
        layer most ciphertexts have spent most of their levels, so the first is what the card sees and
        the second is only an upper bound.
        """
        rows = sorted(self.per_label.values(), key=lambda p: -p.nbytes)
        width = max((len(p.where) for p in rows), default=0)
        lines = [f"  {'stage':<{width}}  {'live':>5}  {'actual':>10}  {'if full':>10}"]
        for row in rows:
            full = row.count * ciphertext_bytes(level, slot_count)
            lines.append(f"  {row.where:<{width}}  {row.count:5d}  "
                         f"{row.nbytes / (1 << 30):6.2f} GiB  {full / (1 << 30):6.2f} GiB")
        return "\n".join(lines)


def tracking_engine(geometry, **kwargs):
    """A ``ClearEngine`` that records every ciphertext it makes. Returns ``(engine, workingset)``."""
    tracker_box = {}

    class Tracking(ClearEngine):
        def _track(self, ct):
            tracker = tracker_box.get("t")
            return tracker.note(ct) if tracker is not None and isinstance(ct, ClearCiphertext) else ct

    for name in ("encrypt", "add", "subtract", "multiply", "conjugate", "multiply_1j", "rotate",
                 "rescale", "level_down", "square", "bootstrap", "relinearize"):
        def make(method_name):
            parent = getattr(ClearEngine, method_name)

            def wrapper(self, *args, **kw):
                return self._track(parent(self, *args, **kw))
            return wrapper
        setattr(Tracking, name, make(name))

    engine = Tracking(geometry, **kwargs)
    tracker = WorkingSet(engine, geometry.slot_count)
    tracker_box["t"] = tracker
    return engine, tracker
