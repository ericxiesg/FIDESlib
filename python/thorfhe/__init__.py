"""THOR's BERT layer on top of ``pyfideslib``.

Port of ``THOR/src/thor/he.py`` (which targets the closed-source desilofhe engine) onto the fideslib
CKKS API, so the same code runs on OpenFHE (CPU) and FIDESlib (CUDA). Stages 01-05 - complexify,
rotated copies, and the query/key/value block-diagonal products - are here; the rest follow.

Read :mod:`thorfhe.geometry` first: THOR's slot layout is spelled out there as data rather than as
literals scattered through the code, which is what lets the whole pipeline run in 512 slots in a test.
"""
from .clear import ClearCiphertext, ClearEngine, ScaleMismatch
from .encoding import (
    block_diagonal_masks,
    pooler_mask,
    decode_linear_output,
    encode_activations,
    encode_bias,
    encode_bias_raw,
    encode_weight,
    FF_SLOT_INDICES,
    encode_bias_classifier,
    encode_bias_pooler,
    encode_weight_classifier,
    encode_weight_ff,
    encode_weight_pooler,
    encode_weight_raw,
    lower_diagonal_entry,
    to_diagonal_blocks,
)
from .geometry import (FEEDFORWARD_STRIDE, FEEDFORWARD_WINDOW, SMALL, THOR_ATTENTION_DENSE,
                       THOR_BERT, THOR_FEEDFORWARD, Geometry)
from .dense import DenseStages
from .feedforward import FeedForwardStages
from .layernorm import LayerNormStages
from .he import LightWeights, decrypt_ciphertexts, encrypt_activations, plan_rotation_keys
from .stages import Stages
from .weights_io import (
    bytes_on_disk,
    qkv_path,
    read_linear,
    read_qkv_layer,
    write_linear,
    write_qkv_layer,
)

__all__ = [
    "ClearCiphertext",
    "ClearEngine",
    "DenseStages",
    "FeedForwardStages",
    "Geometry",
    "LayerNormStages",
    "LightWeights",
    "ScaleMismatch",
    "SMALL",
    "THOR_ATTENTION_DENSE",
    "THOR_BERT",
    "THOR_FEEDFORWARD",
    "FEEDFORWARD_WINDOW",
    "FEEDFORWARD_STRIDE",
    "Stages",
    "block_diagonal_masks",
    "pooler_mask",
    "decode_linear_output",
    "encode_activations",
    "encode_bias",
    "encode_bias_raw",
    "encode_weight",
    "FF_SLOT_INDICES",
    "encode_bias_classifier",
    "encode_bias_pooler",
    "encode_weight_classifier",
    "encode_weight_ff",
    "encode_weight_pooler",
    "encode_weight_raw",
    "encrypt_activations",
    "decrypt_ciphertexts",
    "lower_diagonal_entry",
    "plan_rotation_keys",
    "to_diagonal_blocks",
    "bytes_on_disk",
    "qkv_path",
    "read_linear",
    "read_qkv_layer",
    "write_linear",
    "write_qkv_layer",
]
