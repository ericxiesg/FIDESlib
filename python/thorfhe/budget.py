"""What a parameter set will cost on the GPU, before spending twenty minutes finding out.

Every GPU round so far has ended in an out-of-memory, and each one cost a full context build and key
generation to discover. The key part of that is exactly predictable, so it is predicted here; the
bootstrap part is not, so it is *measured* - the numbers below are from the runs in ``bugs/``, and the
model interpolates rather than pretending to derive OpenFHE's precomputation.

Validated against three reported runs (see ``MEASURED_BOOTSTRAP`` and the tests): key memory is
reproduced to better than 1%.
"""
from __future__ import annotations

from dataclasses import dataclass

MIB = 1 << 20
GIB = 1 << 30


def key_bytes(*, log_n: int, level: int, special_primes: int, dnum: int) -> int:
    """One key-switching key held to ``level``.

    A key is ``2 * dnum`` polynomials over ``level + 1 + K`` RNS towers - this is the same expression
    ``ContextData::printKeyMemoryReport`` uses for its "would be ... untruncated" figure, so it can be
    checked directly against the log line.
    """
    return 2 * dnum * (level + 1 + special_primes) * (1 << log_n) * 8


def special_prime_count(*, log_n: int, depth: int, dnum: int, measured_key_mib: float,
                        keys: int) -> int:
    """Recover ``K`` from a reported key-memory line, since it depends on the digit decomposition.

    ``computeK`` counts special primes until their bits cover the largest digit, which depends on the
    actual prime sizes; rather than re-derive it, read it back out of a run.
    """
    per_key = measured_key_mib * MIB / keys
    towers = per_key / (2 * dnum * (1 << log_n) * 8)
    return max(0, round(towers) - (depth + 1))


#: Bootstrap precomputation footprints actually observed, keyed by (log_n, slots, level budget).
#: ``plaintexts`` is the count the context reports; ``towers`` is the per-plaintext RNS tower count
#: recovered from the reported total, which is what scales with depth.
MEASURED_BOOTSTRAP = {
    (16, 32768, (3, 3)): {"plaintexts": 378, "keys": 48,
                          "samples": [(30, 6426), (50, 10395)]},   # (depth, plaintext MiB)
    # Measured 2026-09-08 at depth 51 / dnum 4. Only one depth sample, so the tower-per-depth slope
    # is borrowed from (3,3) - it is a property of the RNS chain, not of the level budget.
    (16, 32768, (4, 4)): {"plaintexts": 248, "keys": 73,
                          "samples": [(51, 6882)],
                          "resident_key_mib": 17452,   # 73 bootstrap + 15 rotation, 36 truncated
                          "bootstrap_depth": 18},
}


def bootstrap_plaintext_bytes(*, log_n: int, depth: int, slots: int,
                             level_budget: tuple[int, int]) -> int | None:
    """The StC/CtS plaintexts, interpolated from measurement. ``None`` when nothing was measured.

    The count is fixed by the level budget and the slot count; only the tower count grows with depth,
    linearly, which the two samples pin down exactly.
    """
    entry = MEASURED_BOOTSTRAP.get((log_n, slots, tuple(level_budget)))
    if entry is None:
        return None
    samples = entry["samples"]
    per_tower = entry["plaintexts"] * (1 << log_n) * 8
    if len(samples) >= 2:
        (d0, m0), (d1, m1) = samples[:2]
        slope = (m1 * MIB / per_tower - m0 * MIB / per_tower) / (d1 - d0)
    else:
        # One sample only: the towers-per-depth slope belongs to the RNS chain, not to the level
        # budget, so borrow the one the two-sample entry pins down.
        (a0, n0), (a1, n1) = MEASURED_BOOTSTRAP[(16, 32768, (3, 3))]["samples"]
        base = MEASURED_BOOTSTRAP[(16, 32768, (3, 3))]["plaintexts"] * (1 << 16) * 8
        slope = (n1 * MIB / base - n0 * MIB / base) / (a1 - a0)
        (d0, m0) = samples[0]
    towers = m0 * MIB / per_tower + slope * (depth - d0)
    return int(entry["plaintexts"] * towers * (1 << log_n) * 8)


def bootstrap_depth(*, log_n: int = 16, slots: int = 32768,
                    level_budget: tuple[int, int] = (3, 3)) -> int | None:
    """Levels ``EvalBootstrap`` itself consumes, where it has been measured.

    This is what ties the level wall to the memory wall: the level a bootstrap restores to is
    ``depth - bootstrap_depth``, so a level budget that shrinks the precomputation also raises the
    depth the layer needs.
    """
    entry = MEASURED_BOOTSTRAP.get((log_n, slots, tuple(level_budget)))
    return None if entry is None else entry.get("bootstrap_depth")


#: Measured totals sit about 9 GiB above the sum of keys + bootstrap plaintexts, and that has to be
#: carried or the prediction comes out optimistic and the next run fails the same way.
#:
#: What it is, established 2026-09-08 by reading `GPUmalloc`: the device allocator is a **slab pool
#: keyed by exact allocation size**, and slabs are never returned to the driver. A size class whose
#: free list is empty takes a fresh 1 GiB slab. So this term is the ciphertext working set - a layer
#: holds 64 rotated copies in stage 02 alone - *rounded up to whole slabs per size class*, plus the
#: auxiliary polynomial pool, which `ContextData::trimAuxilarPoly` could drain but nothing ever calls.
#: It is a calibration, not a derivation, so it is named as one.
CALIBRATED_OVERHEAD = 9 * GIB

#: Key generation needs scratch on top of the steady state: `KeySwitchingKey::Initialize` calls
#: `generateAllDecompAndDigit`, which allocates the decomposition and digit buffers before the key
#: settles. Round 4 (depth 44, predicted +0.6 GiB of headroom) still died there, in `AddRotationKeys`
#: rather than at use - so the peak is at least the headroom that run had. Carried as a floor on
#: required headroom rather than as a term in the total, because it is transient.
KEYGEN_SCRATCH = 3 * GIB


@dataclass
class Budget:
    """A predicted GPU footprint, in bytes."""

    rotation_keys: int
    bootstrap_keys: int
    bootstrap_plaintexts: int | None
    special_primes: int
    rotation_key_count: int
    overhead: int = CALIBRATED_OVERHEAD

    @property
    def predictable(self) -> bool:
        """False when the bootstrap precomputation for this level budget has never been measured."""
        return self.bootstrap_plaintexts is not None

    @property
    def total(self) -> int | None:
        if not self.predictable:
            return None
        return (self.rotation_keys + self.bootstrap_keys + self.bootstrap_plaintexts
                + self.overhead)

    def format(self, card_bytes: int | None = None) -> str:
        lines = [f"  {'rotation keys':<24}{self.rotation_keys / GIB:7.1f} GiB  "
                 f"{self.rotation_key_count} keys",
                 f"  {'bootstrap keys':<24}{self.bootstrap_keys / GIB:7.1f} GiB"]
        if not self.predictable:
            # Reporting zero for the unmeasured part and adding it up would give a confident, rosy,
            # wrong total - and the cost of believing it is another run that ends in an OOM.
            lines += ["  bootstrap plaintexts     UNKNOWN - this level budget has never been measured.",
                      "                           Run it once and add the reported plaintext count "
                      "and MiB to",
                      "                           budget.MEASURED_BOOTSTRAP. No total without it.",
                      f"  (K = {self.special_primes} special primes)"]
            return "\n".join(lines)

        lines.append(f"  {'bootstrap plaintexts':<24}{self.bootstrap_plaintexts / GIB:7.1f} GiB")
        lines.append(f"  {'everything else':<24}{self.overhead / GIB:7.1f} GiB  "
                     "calibrated from round 3, not derived")
        lines.append(f"  {'total':<24}{self.total / GIB:7.1f} GiB")
        if card_bytes:
            free = card_bytes - self.total
            lines.append(f"  {'headroom on card':<24}{free / GIB:7.1f} GiB"
                         f"{'  -- WILL NOT FIT' if free < 0 else ''}")
            if 0 <= free < KEYGEN_SCRATCH:
                lines.append(f"  -- headroom is under the {KEYGEN_SCRATCH / GIB:.0f} GiB key "
                             "generation needs for its decomposition scratch:")
                lines.append("     expect the OOM in AddRotationKeys, before anything runs.")
        lines.append(f"  (K = {self.special_primes} special primes)")
        return "\n".join(lines)


def estimate(*, log_n: int = 16, depth: int = 50, dnum: int = 4, slots: int | None = None,
             rotation_levels: dict[int, int] | None = None, rotation_keys: int | None = None,
             level_budget: tuple[int, int] | None = (3, 3), special_primes: int = 11,
             truncate: bool = True, overhead: int = CALIBRATED_OVERHEAD) -> Budget:
    """Predict the resident footprint of a context.

    ``rotation_levels`` is a :func:`~thorfhe.he.plan_rotation_keys` plan, which lets the truncated
    size be computed key by key; ``rotation_keys`` is the fallback when only a count is known.
    """
    slots = (1 << (log_n - 1)) if slots is None else slots
    full = key_bytes(log_n=log_n, level=depth, special_primes=special_primes, dnum=dnum)

    if rotation_levels:
        count = len(rotation_levels)
        rotation = sum(
            key_bytes(log_n=log_n, level=min(level + 1, depth), special_primes=special_primes,
                      dnum=dnum) if truncate else full
            for level in rotation_levels.values())
    else:
        count = rotation_keys or 0
        rotation = count * full

    entry = MEASURED_BOOTSTRAP.get((log_n, slots, tuple(level_budget))) if level_budget else None
    if entry and "resident_key_mib" in entry:
        # Anchor on what a run actually reported rather than re-deriving how many bootstrap keys were
        # truncated and to what level - that accounting is ambiguous in the log and the derived figure
        # came out 4% high. Scale by tower count for other depths, which is exact.
        measured_depth = entry["samples"][0][0]
        anchor_towers = measured_depth + 1 + special_primes
        scaled = entry["resident_key_mib"] * MIB * (depth + 1 + special_primes) / anchor_towers
        boot_keys = max(0, int(scaled) - rotation) if rotation_levels or rotation_keys else int(scaled)
    else:
        boot_keys = (entry["keys"] if entry else 0) * full
    boot_pt = (bootstrap_plaintext_bytes(log_n=log_n, depth=depth, slots=slots,
                                         level_budget=level_budget)
               if level_budget else 0)

    return Budget(rotation_keys=rotation, bootstrap_keys=boot_keys, bootstrap_plaintexts=boot_pt,
                  special_primes=special_primes, rotation_key_count=count,
                  overhead=overhead if level_budget else 0)
