"""Encoded weights on disk, in THOR's directory layout but in the fideslib light-plaintext format.

THOR's ``encode_weights.py`` pre-encodes every BERT weight once and writes one desilofhe light
plaintext per file, then ``he.py`` reads them back per stage (``stage_03/layer_0/w_0_2_37``). The
layout is worth keeping - it is what makes a 110 GiB model streamable - so this module writes the same
tree with :class:`fideslib.LightPlaintext` files instead.

Loading the BERT checkpoint is deliberately *not* here: pass the weight and bias as numpy arrays and
keep safetensors/transformers out of the runtime that has to run on the GPU box.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .encoding import encode_bias, encode_weight
from .geometry import Geometry

#: file name a multi-index maps to, e.g. ``w_0_2_37``. Matches THOR's ``"_".join(map(str, index))``.
WEIGHT_PREFIX = "w_"
BIAS_PREFIX = "b_"


def _write_all(engine, messages: np.ndarray, path: Path, prefix: str) -> int:
    path.mkdir(parents=True, exist_ok=True)
    for index, message in np.ndenumerate(messages):
        name = prefix + "_".join(map(str, index))
        engine.write_light_plaintext(engine.encode_to_light_plaintext(message), path / name)
    return messages.size


def _read_all(engine, shape: tuple[int, ...], path: Path, prefix: str) -> np.ndarray:
    out = np.empty(shape, dtype=object)
    for index in np.ndindex(shape):
        name = prefix + "_".join(map(str, index))
        out[index] = engine.read_light_plaintext(path / name)
    return out


def write_linear(engine, geometry: Geometry, path, w: np.ndarray, b: np.ndarray, scale: float = 1.0) -> int:
    """Encode one linear layer into ``path`` and return the number of files written.

    ``scale`` folds a constant into both operands, the way THOR folds the softmax scale into the key
    projection (``1/512``, or ``1/1024`` for layer 2).
    """
    path = Path(path)
    written = _write_all(engine, encode_weight(geometry, w, scale), path, WEIGHT_PREFIX)
    written += _write_all(engine, encode_bias(geometry, b, scale), path, BIAS_PREFIX)
    return written


def read_linear(engine, geometry: Geometry, path) -> tuple[np.ndarray, np.ndarray]:
    """Read back what :func:`write_linear` wrote: ``(weight, bias)`` arrays of light plaintexts."""
    path = Path(path)
    weight_shape = (geometry.n_output_ciphertexts, geometry.diag_count, geometry.n_in_complex)
    weight = _read_all(engine, weight_shape, path, WEIGHT_PREFIX)
    bias = _read_all(engine, (geometry.n_output_ciphertexts,), path, BIAS_PREFIX)
    return weight, bias


def qkv_path(root, stage: str, layer_index: int) -> Path:
    """THOR's ``<root>/stage_03/layer_0`` convention (``stage_03``/``stage_04``/``stage_05``)."""
    return Path(root) / stage / f"layer_{layer_index}"


def write_qkv_layer(engine, geometry: Geometry, root, layer_index: int,
                    query: tuple[np.ndarray, np.ndarray],
                    key: tuple[np.ndarray, np.ndarray],
                    value: tuple[np.ndarray, np.ndarray],
                    key_scale: float = 1.0) -> int:
    """Write the query/key/value projections of one encoder layer as stages 03/04/05.

    Each argument is the ``(weight, bias)`` pair straight out of the checkpoint, in BERT's ``(out, in)``
    orientation. ``key_scale`` is THOR's softmax scale, folded into the key projection so the attention
    score comes out in range.
    """
    written = write_linear(engine, geometry, qkv_path(root, "stage_03", layer_index), *query)
    written += write_linear(engine, geometry, qkv_path(root, "stage_04", layer_index), *key, scale=key_scale)
    written += write_linear(engine, geometry, qkv_path(root, "stage_05", layer_index), *value)
    return written


def read_qkv_layer(engine, geometry: Geometry, root, layer_index: int) -> dict[str, tuple]:
    """Read back the three projections of one encoder layer, keyed by stage name."""
    return {stage: read_linear(engine, geometry, qkv_path(root, stage, layer_index))
            for stage in ("stage_03", "stage_04", "stage_05")}


def bytes_on_disk(geometry: Geometry) -> int:
    """Size of one encoded linear layer on disk, for the memory model.

    A light plaintext is ``slot_count * 2`` coefficients of 8 bytes plus a 32-byte header, and a layer
    is ``n_output_ciphertexts * (diag_count * n_in_complex + 1)`` of them.
    """
    per_file = geometry.slot_count * 2 * 8 + 32
    files = geometry.n_output_ciphertexts * (geometry.diag_count * geometry.n_in_complex + 1)
    return per_file * files
