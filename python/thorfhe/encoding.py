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
def _gather_upper_diagonal(g: Geometry, blocks, input_indices, rotations):
    """THOR ``model_encoder.gather_upper_diagonal_batch``."""
    offsets = rotations[:, None] + np.arange(g.dim)[None, :]
    row_indices = offsets % blocks.shape[1]
    real_cols = (input_indices[:, None] + offsets) % blocks.shape[2]
    imag_cols = (((input_indices + g.n_in_complex) % g.n_in)[:, None] + offsets) % blocks.shape[2]
    block_indices = np.arange(blocks.shape[0])[None, :, None]
    real = blocks[block_indices, row_indices[:, None, :], real_cols[:, None, :]]
    imag = blocks[block_indices, row_indices[:, None, :], imag_cols[:, None, :]]
    # The 1/2 pairs with the `y + conj(y)` at the end of the layer, which doubles the real part.
    return (real - 1j * imag) / 2


def encode_weight(g: Geometry, w: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pack a (features, features) weight into an ``(out_ct, diag_count, n_in_complex)`` object array.

    THOR ``model_encoder.encode_w_att`` / ``encode_w_qkv``. ``w`` is in BERT's ``(out, in)`` order, so
    the layer computes ``x @ w.T``.
    """
    if w.shape != (g.features, g.features):
        raise ValueError(f"weight must be ({g.features}, {g.features}), got {w.shape}")
    diag_blocks = to_diagonal_blocks(w, g.block_shape)
    pack_range = np.arange(g.pack)

    messages = np.empty((g.n_output_ciphertexts, g.diag_count, g.n_in_complex), dtype=object)
    for diag_index in range(g.diag_count):
        diagonal = diag_blocks[diag_index]
        for n in range(g.n_in_complex):
            for out_ct in range(g.n_output_ciphertexts):
                rotations = out_ct * g.pack + pack_range
                input_indices = ((n // g.pack) * g.pack + out_ct * g.pack
                                 + (n + pack_range) % g.pack) % g.n_in_complex
                input_indices = (input_indices - rotations) % g.n_in
                values = scale * _gather_upper_diagonal(g, diagonal, input_indices, rotations)
                msg = np.zeros((g.slot_count,), dtype=complex)
                msg[_slot_positions(g)] = values.transpose(0, 2, 1)
                messages[out_ct, diag_index, n] = msg
    return messages


def encode_bias(g: Geometry, b: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Pack a (features,) bias into ``n_output_ciphertexts`` slot vectors (THOR ``encode_b``).

    Note that THOR does **not** halve the bias the way ``encode_weight`` halves the weights, so the
    ``y + conj(y)`` at the end of a QKV stage yields ``x @ w.T + 2 * b``. That is THOR's convention and
    the port reproduces it; see ``docs/thor_port.md``.
    """
    if b.shape != (g.features,):
        raise ValueError(f"bias must be ({g.features},), got {b.shape}")
    blocks = np.stack(np.split(b, g.n_blocks), axis=0)  # (n_blocks, n_out)
    positions = _slot_positions(g)

    messages = np.empty((g.n_output_ciphertexts,), dtype=object)
    for out_ct in range(g.n_output_ciphertexts):
        rotations = out_ct * g.pack + np.arange(g.pack)
        gather = (rotations[:, None] + np.arange(g.dim)[None, :]) % g.n_out  # (pack, dim)
        rotated = np.take_along_axis(blocks[:, None, :], gather[None, :, :], axis=2).transpose(1, 2, 0)
        msg = np.zeros((g.slot_count,), dtype=float)
        msg[positions] = scale * rotated
        messages[out_ct] = msg
    return messages


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
