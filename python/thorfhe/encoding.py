"""Slot encoders for the THOR packing, parameterised by :class:`~thorfhe.geometry.Geometry`.

Ports THOR's ``model_encoder.py`` (weights, biases) and ``data_encoder.py`` (activations), replacing
the module-level ``SLOT_COUNT`` / ``GROUP_SIZE`` / ``PACK`` / ``DIM`` constants and the bare ``12``,
``16`` and ``64`` literals with the corresponding geometry fields. Everything here is numpy; nothing
imports an FHE backend.

The one thing worth reading twice is which ``16`` is which: in ``encode_weight`` the ``16`` inside
``rotations`` and ``input_indices`` is ``pack`` (the group index a value belongs to), while the ``16``
in the slot base is ``n_slot`` (slots reserved per token). THOR uses 16 for both, so this only shows
up at a geometry where they differ - which is exactly why :data:`~thorfhe.geometry.SMALL` sets
``pack=4, n_slot=8``.
"""
from __future__ import annotations

import numpy as np

from .geometry import Geometry
from .numeric import ACTIVATION_SCALE


def lower_diagonal_entry(matrix: np.ndarray, diagonal: int, index: int):
    """THOR ``utils.ld_entry``: entry ``index`` of lower diagonal ``diagonal``."""
    rows, cols = matrix.shape
    return matrix[(diagonal + index) % rows, index % cols]


def to_diagonal_blocks(matrix: np.ndarray, block_shape: tuple[int, int]) -> np.ndarray:
    """THOR ``utils.to_blocks(..., diag=True)``: (diag_rows, diag_cols, block_rows, block_cols)."""
    rows, cols = matrix.shape
    block_rows, block_cols = block_shape
    if rows % block_rows or cols % block_cols:
        raise ValueError("matrix shape must be divisible by the block shape")
    vertical, horizontal = rows // block_rows, cols // block_cols
    blocks = matrix.reshape(vertical, block_rows, horizontal, block_cols).transpose(0, 2, 1, 3)

    diag_rows, diag_cols = min(vertical, horizontal), max(vertical, horizontal)
    out = np.empty((diag_rows, diag_cols, block_rows, block_cols), dtype=matrix.dtype)
    rows_index = np.arange(diag_rows)
    for diag_col in range(diag_cols):
        out[:, diag_col] = blocks[(rows_index + diag_col) % vertical, diag_col % horizontal]
    return out


def _slot_positions(g: Geometry) -> np.ndarray:
    """(pack, dim, n_blocks) array of slot indices, i.e. ``group * group_size + token * n_slot + block``."""
    return (g.group_size * np.arange(g.pack)[:, None, None]
            + g.n_slot * np.arange(g.dim)[None, :, None]
            + np.arange(g.n_blocks)[None, None, :])


# ---------------------------------------------------------------- activations
def encode_activations(g: Geometry, x: np.ndarray) -> np.ndarray:
    """Pack a (dim, features) activation matrix into ``n_input_ciphertexts`` slot vectors.

    THOR ``DataEncoder.encrypt_embedding``. Feature ``f`` of token ``t`` lands in the real part when
    ``f % n_in < n_in_complex`` and in the imaginary part otherwise, on the lower diagonal
    ``(f - t) mod n_in_complex`` of block ``(f // n_in) % horizontal``.
    """
    if x.shape != (g.dim, g.features):
        raise ValueError(f"activations must be ({g.dim}, {g.features}), got {x.shape}")
    horizontal = g.features // g.n_in
    blocks = np.stack(np.split(x.T, horizontal), axis=0)  # (horizontal, n_in, dim)

    out = np.empty((g.n_input_ciphertexts,), dtype=object)
    for ct in range(g.n_input_ciphertexts):
        msg = np.zeros((g.slot_count,), dtype=complex)
        for group in range(g.pack):
            diagonal = ct * g.pack + group
            for token in range(g.dim):
                for block in range(g.n_blocks):
                    b = blocks[block % horizontal]
                    msg[g.slot(group, token, block)] = complex(
                        lower_diagonal_entry(b, diagonal, token),
                        lower_diagonal_entry(b, diagonal + g.n_in_complex, token),
                    )
        out[ct] = msg
    return out


def decode_linear_output(g: Geometry, messages) -> np.ndarray:
    """Inverse of the layout stages 03-05 produce: read a (dim, features) matrix out of the slots.

    ``msg[ct][group, token, block] = y[token, n_out * block + ((ct * pack + group) + token) % n_out]``.
    """
    y = np.zeros((g.dim, g.features))
    for ct in range(g.n_output_ciphertexts):
        msg = np.real(np.asarray(messages[ct]))
        for group in range(g.pack):
            diagonal = ct * g.pack + group
            for token in range(g.dim):
                for block in range(g.n_blocks):
                    feature = g.n_out * block + (diagonal + token) % g.n_out
                    y[token, feature] = msg[g.slot(group, token, block)]
    return y


# ---------------------------------------------------------------- weights
def _gather_upper_diagonal(blocks, input_indices, rotations, dim, n_in):
    """THOR ``model_encoder.gather_upper_diagonal_batch``."""
    n_in_complex = n_in // 2
    offsets = rotations[:, None] + np.arange(dim)[None, :]
    row_indices = offsets % blocks.shape[1]
    real_cols = (input_indices[:, None] + offsets) % blocks.shape[2]
    imag_cols = (((input_indices + n_in_complex) % n_in)[:, None] + offsets) % blocks.shape[2]
    block_indices = np.arange(blocks.shape[0])[None, :, None]
    real = blocks[block_indices, row_indices[:, None, :], real_cols[:, None, :]]
    imag = blocks[block_indices, row_indices[:, None, :], imag_cols[:, None, :]]
    # The 1/2 pairs with the `y + conj(y)` at the end of the layer, which doubles the real part.
    return (real - 1j * imag) / 2


def encode_weight_raw(w: np.ndarray, *, dim: int, pack: int, n_slot: int, group_size: int,
                      slot_count: int, n_in: int, n_out: int, slot_indices, scale: float = 1.0):
    """THOR ``model_encoder.encode_w_att``, with every shape taken as an argument.

    :class:`~thorfhe.geometry.Geometry` fixes ``n_in``/``n_out`` for one representation, but a BERT
    layer changes representation as it goes: the QKV projections are 12 blocks of 64 while the
    attention dense output is 6 blocks of 128, and the feed-forward stages change again. So the
    encoder takes the block shape directly, and ``slot_indices`` says which of a token's slots the
    result occupies (THOR's ``ATT_SLOT_INDICES`` and ``FF_SLOT_INDICES``).

    Returns an ``(out_ct, diag_count, n_in // 2)`` object array of slot vectors.
    """
    return _pack_diagonals(to_diagonal_blocks(w, (n_out, n_in)), dim=dim, pack=pack, n_slot=n_slot,
                           group_size=group_size, slot_count=slot_count, n_in=n_in, n_out=n_out,
                           slot_indices=slot_indices, scale=scale)


def _pack_diagonals(diag_blocks, *, dim, pack, n_slot, group_size, slot_count, n_in, n_out,
                    slot_indices, scale):
    """The shared body: gather each block diagonal and lay it out over the slots."""
    n_in_complex = n_in // 2
    diag_count = diag_blocks.shape[0]
    n_out_packed = n_out // pack
    pack_range = np.arange(pack)
    slot_indices = np.asarray(slot_indices)

    positions = (group_size * np.arange(pack)[:, None, None]
                 + n_slot * np.arange(dim)[None, :, None]
                 + slot_indices[None, None, :])

    messages = np.empty((n_out_packed, diag_count, n_in_complex), dtype=object)
    for diag_index in range(diag_count):
        diagonal = diag_blocks[diag_index]
        for n in range(n_in_complex):
            for out_ct in range(n_out_packed):
                rotations = out_ct * pack + pack_range
                input_indices = ((n // pack) * pack + out_ct * pack + (n + pack_range) % pack) % n_in_complex
                input_indices = (input_indices - rotations) % n_in
                values = scale * _gather_upper_diagonal(diagonal, input_indices, rotations, dim, n_in)
                msg = np.zeros((slot_count,), dtype=complex)
                msg[positions] = values.transpose(0, 2, 1)
                messages[out_ct, diag_index, n] = msg
    return messages


#: THOR ``FF_SLOT_INDICES``: the feed-forward stages carry twelve blocks as two windows of six.
FF_SLOT_INDICES = np.array([0, 1, 2, 3, 4, 5, 8, 9, 10, 11, 12, 13])


def encode_weight_ff(w: np.ndarray, *, dim: int, pack: int, n_slot: int, group_size: int,
                     slot_count: int, n_in: int, n_out: int, split: int = 4, axis: int = 0,
                     slot_indices=FF_SLOT_INDICES, scale: float = 1.0):
    """THOR ``model_encoder.encode_w_ff``: an oversized weight split into pieces, packed two at a time.

    Neither feed-forward weight fits one block diagonal. The expansion (stage 12) is
    (4 * features, features) and is split *vertically* (``axis=0``); the contraction (stage 14) is
    (features, 4 * features) and is split *horizontally* (``axis=1``). Either way the four pieces are
    square, and each ``rep`` stacks two of them - so a rep carries twelve block rows where the attention
    stages carry six, laid out as two windows of six per token. That is what ``FF_SLOT_INDICES`` and the
    ``block_diag_2`` masks (modulo 8, not 16) are about.

    Returns a ``(2, out_ct, diag_count, n_in // 2)`` object array.
    """
    splitter = np.vsplit if axis == 0 else np.hsplit
    pieces = [to_diagonal_blocks(piece, (n_out, n_in)) for piece in splitter(w, split)]
    reps = []
    for rep in range(2):
        combined = np.concatenate((pieces[2 * rep], pieces[2 * rep + 1]), axis=1)
        reps.append(_pack_diagonals(combined, dim=dim, pack=pack, n_slot=n_slot,
                                    group_size=group_size, slot_count=slot_count, n_in=n_in,
                                    n_out=n_out, slot_indices=slot_indices, scale=scale))
    return np.stack(reps)


def encode_weight(g: Geometry, w: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pack a (features, features) weight for the geometry's own representation.

    THOR ``model_encoder.encode_w_qkv``. ``w`` is in BERT's ``(out, in)`` order, so the layer computes
    ``x @ w.T``. For a stage whose input and output blockings differ, use :func:`encode_weight_raw`.
    """
    if w.shape != (g.features, g.features):
        raise ValueError(f"weight must be ({g.features}, {g.features}), got {w.shape}")
    return encode_weight_raw(w, dim=g.dim, pack=g.pack, n_slot=g.n_slot, group_size=g.group_size,
                             slot_count=g.slot_count, n_in=g.n_in, n_out=g.n_out,
                             slot_indices=np.arange(g.n_blocks), scale=scale)


def encode_bias_raw(b: np.ndarray, *, dim: int, pack: int, n_slot: int, group_size: int,
                    slot_count: int, n_out: int, n_blocks: int, slot_indices, scale: float = 1.0):
    """THOR ``model_encoder.encode_b``, with every shape taken as an argument.

    ``n_blocks`` here is how many pieces the bias splits into (``features // n_out``), which is not
    the same as the geometry's ``n_blocks`` once a stage changes representation - see
    :func:`encode_weight_raw`.
    """
    blocks = np.stack(np.split(b, n_blocks), axis=0)  # (n_blocks, n_out)
    slot_indices = np.asarray(slot_indices)
    positions = (group_size * np.arange(pack)[:, None, None]
                 + n_slot * np.arange(dim)[None, :, None]
                 + slot_indices[None, None, :])

    messages = np.empty((n_out // pack,), dtype=object)
    for out_ct in range(n_out // pack):
        rotations = out_ct * pack + np.arange(pack)
        gather = (rotations[:, None] + np.arange(dim)[None, :]) % n_out  # (pack, dim)
        rotated = np.take_along_axis(blocks[:, None, :], gather[None, :, :], axis=2).transpose(1, 2, 0)
        msg = np.zeros((slot_count,), dtype=float)
        msg[positions] = scale * rotated
        messages[out_ct] = msg
    return messages


def encode_bias(g: Geometry, b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pack a (features,) bias into ``n_output_ciphertexts`` slot vectors (THOR ``encode_b``).

    Note that THOR does **not** halve the bias the way ``encode_weight`` halves the weights, so the
    ``y + conj(y)`` at the end of a QKV stage yields ``x @ w.T + 2 * b``. That is THOR's convention and
    the port reproduces it; see ``docs/thor_port.md``.
    """
    if b.shape != (g.features,):
        raise ValueError(f"bias must be ({g.features},), got {b.shape}")
    # `out_blocks`, not `n_blocks`: THOR's `encode_b(b, n_blocks=features // n_out)` fills the first
    # `features // n_out` slots of a token. The two coincide for the QKV geometry and do not for the
    # attention dense one, which is twelve blocks of 64 in and six of 128 out.
    return encode_bias_raw(b, dim=g.dim, pack=g.pack, n_slot=g.n_slot, group_size=g.group_size,
                           slot_count=g.slot_count, n_out=g.n_out, n_blocks=g.out_blocks,
                           slot_indices=np.arange(g.out_blocks), scale=scale)


# ---------------------------------------------------------------- pooler and classifier
def encode_weight_pooler(g: Geometry, w: np.ndarray,
                         carrier: float = ACTIVATION_SCALE) -> np.ndarray:
    """THOR ``model_encoder.encode_w_pooler``: a (features, features) weight for the CLS token only.

    The pooler multiplies a single token, broadcast over all of them, so there is no rotated-copy
    dimension over ``pack`` - the pack offset is folded into the *input* index instead. Returns a
    ``(diag_count, n_in // (2 * pack))`` object array, i.e. ``(6, 4)``.

    ``carrier`` is the amplitude the input arrives at, and it is divided out here rather than after the
    product because it cannot be divided out afterwards: ``encode_bias_pooler`` halves the bias so the
    closing fold gives ``+b``, which leaves ``stage_17_pooler`` holding ``carrier * (y @ w) + b`` - the
    linear term scaled and the bias not, which is no uniform factor at all. And ``tanh`` is the next
    thing that happens to it.

    The default is the amplitude a layer leaves its output at, which is what the pooler is fed in the
    network. Measured on the clear engine, a pooler encoded for amplitude 1 and handed the 2 a
    LayerNorm actually produces is off by 0.154 on a function whose range is [-1, 1].
    """
    if w.shape != (g.features, g.features):
        raise ValueError(f"pooler weight must be ({g.features}, {g.features}), got {w.shape}")
    diag_blocks = to_diagonal_blocks(w / carrier, (g.n_out, g.n_in))
    diag_count = diag_blocks.shape[0]
    columns = g.n_in_complex // g.pack

    messages = np.full((diag_count, columns), None, dtype=object)
    tokens = g.n_slot * np.arange(g.dim)[:, None]
    for diag_index in range(diag_count):
        blocks = diag_blocks[diag_index]
        for column in range(columns):
            msg = np.zeros((g.slot_count,), dtype=complex)
            for offset in range(g.pack):
                index = column * g.pack + offset
                values = (blocks[:, :, index] - 1j * blocks[:, :, index + g.n_in_complex]) / 2
                msg[g.group_size * offset + tokens + np.arange(values.shape[0])[None, :]] = values.T
            messages[diag_index, column] = msg
    return messages


def encode_weight_classifier(g: Geometry, w: np.ndarray) -> np.ndarray:
    """THOR ``model_encoder.encode_w_cls``: one slot vector per class, six blocks in slots 0..5."""
    if w.ndim != 2 or w.shape[1] != g.features:
        raise ValueError(f"classifier weight must be (classes, {g.features}), got {w.shape}")
    positions = g.n_slot * np.arange(g.dim)[:, None] + np.arange(g.out_blocks)[None, :]
    messages = np.full((w.shape[0],), None, dtype=object)
    for class_index in range(w.shape[0]):
        msg = np.zeros((g.slot_count,), dtype=float)
        msg[positions] = w[class_index].reshape(g.out_blocks, g.dim).T
        messages[class_index] = msg
    return messages


def encode_bias_pooler(g: Geometry, b: np.ndarray) -> np.ndarray:
    """THOR ``model_encoder.encode_b_pooler``: slot ``n_slot * t + block`` holds ``b[n_out * block + t]``.

    Halved, because ``pooler_dense`` adds the bias before its ``y + conj(y)``. That is the opposite of
    the QKV convention, where the bias is *not* halved and the layer therefore computes ``x @ w.T + 2b``.
    """
    if b.shape != (g.features,):
        raise ValueError(f"bias must be ({g.features},), got {b.shape}")
    group = np.zeros((g.group_size,), dtype=float)
    positions = g.n_slot * np.arange(g.dim)[:, None] + np.arange(g.out_blocks)[None, :]
    group[positions] = np.stack(np.split(b, g.out_blocks), axis=1) / 2
    return np.tile(group, g.pack)


def encode_bias_classifier(g: Geometry, b: np.ndarray) -> np.ndarray:
    """THOR ``model_encoder.encode_b_cls``: one slot vector per class, the bias in slot 0 alone."""
    messages = np.full((b.shape[0],), None, dtype=object)
    for class_index in range(b.shape[0]):
        msg = np.zeros((g.slot_count,), dtype=float)
        msg[0] = b[class_index]
        messages[class_index] = msg
    return messages


def pooler_mask(g: Geometry) -> np.ndarray:
    """THOR ``masks["pooler_dense"]``: the CLS token's six blocks, in every group."""
    return (np.arange(g.slot_count) % g.group_size < g.out_blocks).astype(float)


# ---------------------------------------------------------------- masks
def block_diagonal_masks(g: Geometry, stride: int | None = None,
                        width: int | None = None) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """``rotate_internal`` masks and their complements, keyed by delta.

    Mask ``d`` selects the slots with ``slot % stride < d``, restricted to the ``width`` that carry
    data. THOR has two families: ``block_diag_1`` wraps twelve blocks in a token's sixteen slots, and
    ``block_diag_2`` wraps six in a *half* token's eight - its masks are modulo 8, not 16. Passing
    ``stride`` and ``width`` gives either; the defaults give the first.

    The complements exist because ``Stages.rotate_internal`` multiplies by both instead of masking once
    and subtracting: that keeps the two halves at the same scale under FIXEDMANUAL. Only the used slots
    may be masked back in - the padding would otherwise rotate garbage into the window.
    """
    stride = g.n_slot if stride is None else stride
    width = g.n_blocks if width is None else width
    within = np.arange(g.slot_count) % stride
    used = within < width
    low = {d: ((within < d) & used).astype(float) for d in range(1, stride + 1)}
    high = {d: ((within >= d) & used).astype(float) for d in range(1, stride + 1)}
    return low, high
