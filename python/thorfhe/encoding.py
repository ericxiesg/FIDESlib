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
    n_in_complex = n_in // 2
    diag_blocks = to_diagonal_blocks(w, (n_out, n_in))
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
    return encode_bias_raw(b, dim=g.dim, pack=g.pack, n_slot=g.n_slot, group_size=g.group_size,
                           slot_count=g.slot_count, n_out=g.n_out, n_blocks=g.out_blocks,
                           slot_indices=np.arange(g.n_blocks), scale=scale)


# ---------------------------------------------------------------- masks
def block_diagonal_masks(g: Geometry) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray]]:
    """``rotate_internal`` masks and their complements, keyed by delta.

    Mask ``d`` selects the slots with ``slot % n_slot < d``. THOR stores these per mode
    (``block_diag_1`` splits at 12 blocks, ``block_diag_2`` at 6); here the split point is ``n_blocks``,
    so one family covers both.

    The complements exist because ``Stages.rotate_internal`` multiplies by both instead of masking once
    and subtracting: that keeps the two halves at the same scale under FIXEDMANUAL. Only the first
    ``n_blocks`` slots of a token carry data, so the complement is restricted to those - masking the
    padding slots back in would rotate garbage into the used window.
    """
    within = np.arange(g.slot_count) % g.n_slot
    used = within < g.n_blocks
    low = {d: ((within < d) & used).astype(float) for d in range(1, g.n_slot + 1)}
    high = {d: ((within >= d) & used).astype(float) for d in range(1, g.n_slot + 1)}
    return low, high
