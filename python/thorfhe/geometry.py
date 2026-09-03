"""The slot geometry THOR packs a BERT layer into.

THOR hard-codes ``SLOT_COUNT = 2**15``, ``GROUP_SIZE = 2**11``, ``PACK = 16``, ``DIM = 128`` and a
handful of bare ``12``/``16``/``64`` literals across ``model_encoder.py``, ``data_encoder.py`` and
``he.py``. Those numbers are not independent, and separating them is what makes the port testable:
the whole stage 01-05 pipeline can then run in 512 slots on a laptop instead of 32768 slots on a V100.

The packing is a block-diagonal (BSGS-free) matrix product. One ciphertext holds ``pack`` groups of
``dim`` tokens, each token occupying ``n_slot`` consecutive slots of which ``n_blocks`` are used::

    slot index = group * group_size + token * n_slot + block
    group_size = dim * n_slot          slot_count = pack * group_size

A hidden vector of ``features`` values per token is carried as ``horizontal = features // n_in``
blocks of ``n_in`` features, complex-packed two features to a slot, and the linear layer's output as
``vertical = features // n_out`` blocks of ``n_out``.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Geometry:
    """Slot layout of one THOR-packed BERT layer. See the module docstring for the index formula."""

    #: tokens per group (BERT sequence length).
    dim: int
    #: groups per ciphertext; also the number of rotated copies stage 02 makes.
    pack: int
    #: slots reserved per token; ``n_blocks`` of them carry data, the rest are padding.
    n_slot: int
    #: block-diagonal blocks actually used per token.
    n_blocks: int
    #: hidden size of the layer (768 for BERT-base).
    features: int
    #: rows of an input block; a complex slot carries features ``f`` and ``f + n_in // 2``.
    n_in: int
    #: rows of an output block.
    n_out: int

    def __post_init__(self):
        for name, value in (("dim", self.dim), ("pack", self.pack), ("n_slot", self.n_slot),
                            ("n_in", self.n_in), ("n_out", self.n_out)):
            if value <= 0 or value & (value - 1):
                raise ValueError(f"Geometry.{name} must be a power of two, got {value}")
        if self.n_blocks > self.n_slot:
            raise ValueError(f"n_blocks {self.n_blocks} does not fit in n_slot {self.n_slot}")
        if self.features % self.n_in or self.features % self.n_out:
            raise ValueError("features must be divisible by both n_in and n_out")
        if self.features // self.n_out != self.n_blocks:
            raise ValueError(f"features // n_out = {self.features // self.n_out} must equal n_blocks {self.n_blocks}")
        if self.n_in_complex % self.pack:
            raise ValueError(f"n_in // 2 = {self.n_in_complex} must be divisible by pack {self.pack}")
        if self.n_out % self.pack:
            raise ValueError("n_out must be divisible by pack")

    # ---- derived ----
    @property
    def group_size(self) -> int:
        return self.dim * self.n_slot

    @property
    def slot_count(self) -> int:
        return self.pack * self.group_size

    @property
    def n_in_complex(self) -> int:
        """Input ciphertext copies consumed by one parallel-diagonal product."""
        return self.n_in // 2

    @property
    def n_input_ciphertexts(self) -> int:
        return self.n_in_complex // self.pack

    @property
    def n_output_ciphertexts(self) -> int:
        return self.n_out // self.pack

    @property
    def diag_count(self) -> int:
        """Block diagonals of the weight matrix, i.e. terms combined by ``pcmm``."""
        return min(self.features // self.n_out, self.features // self.n_in)

    @property
    def block_shape(self) -> tuple[int, int]:
        return (self.n_out, self.n_in)

    def slot(self, group: int, token: int, block: int) -> int:
        return group * self.group_size + token * self.n_slot + block


#: BERT-base as THOR packs it: 128 tokens, 768 features, 32768 slots (log N = 16).
THOR_BERT = Geometry(dim=128, pack=16, n_slot=16, n_blocks=12, features=768, n_in=128, n_out=64)

#: A 4096-slot (log N = 13) instance with the same structure, small enough to run under OpenFHE on a
#: laptop while keeping BERT's 128 tokens. ``pack != n_slot`` on purpose: THOR uses 16 for both, so a
#: port that conflated them would still pass at production size.
SMALL = Geometry(dim=128, pack=4, n_slot=8, n_blocks=6, features=48, n_in=16, n_out=8)
