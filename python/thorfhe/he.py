"""Wiring THOR's stages onto a ``pyfideslib.Engine``.

Two things live here that the engine-agnostic modules cannot do on their own: turning the numpy
encodings into light plaintexts, and working out which rotation keys the layer needs and at which
level. Both come from the same place - a dry run of the stages on :class:`~thorfhe.clear.ClearEngine`,
which is exact and costs milliseconds - so the key schedule cannot drift away from the code that uses
the keys, the way THOR's hand-maintained ``rotation_contexts`` table can.
"""
from __future__ import annotations

import numpy as np

from .clear import ClearEngine
from .encoding import block_diagonal_masks, encode_activations, encode_bias, encode_weight
from .geometry import Geometry
from .stages import Stages


def plan_rotation_keys(geometry: Geometry, depth: int, layer_index: int = 0) -> dict[int, int]:
    """``{rotation index: highest level it is used at}`` for the full encoder layer.

    Derived by running the entire layer on the clear engine with dummy data: every ``rotate`` records
    the level of its operand, which is exactly what a level-truncated rotation key has to cover.
    """
    from .layer import EncoderLayer, encode_layer
    from .geometry import THOR_ATTENTION_DENSE, THOR_FEEDFORWARD

    g = geometry
    engine = ClearEngine(g, depth=depth)

    dummy_params = {
        "query.weight": np.zeros((g.features, g.features)),
        "query.bias": np.zeros((g.features,)),
        "key.weight": np.zeros((g.features, g.features)),
        "key.bias": np.zeros((g.features,)),
        "value.weight": np.zeros((g.features, g.features)),
        "value.bias": np.zeros((g.features,)),
        "attention.output.dense.weight": np.zeros((g.features, g.features)),
        "attention.output.dense.bias": np.zeros((g.features,)),
        "attention.output.LayerNorm.weight": np.zeros((g.features,)),
        "attention.output.LayerNorm.bias": np.zeros((g.features,)),
        "intermediate.dense.weight": np.zeros((g.features * 4, g.features)),
        "intermediate.dense.bias": np.zeros((g.features * 4,)),
        "output.dense.weight": np.zeros((g.features, g.features * 4)),
        "output.dense.bias": np.zeros((g.features,)),
        "output.LayerNorm.weight": np.zeros((g.features,)),
        "output.LayerNorm.bias": np.zeros((g.features,)),
    }

    weights = encode_layer(dummy_params, layer_index)
    layer = EncoderLayer(engine)
    x = np.zeros((g.dim, g.features))
    state = encrypt_activations(engine, g, x)
    padding = layer.padding_mask(g.dim)
    try:
        layer.forward(state, weights, padding, layer_index, softmax_parameters=None)
    except Exception:
        pass
    return {k: v for k, v in dict(engine.rotation_levels).items() if v >= 0}


class LightWeights:
    """A QKV weight/bias pair and the ``rotate_internal`` masks, encoded once as light plaintexts.

    The masks and the weights are reused across every layer and every one of stages 03-05, and a light
    plaintext is 1/(L+1) of an expanded one, so holding them all is what makes the layer fit. See
    ``docs/light_plaintext.md``.
    """

    def __init__(self, engine, geometry: Geometry):
        self.engine = engine
        self.g = geometry
        low, high = block_diagonal_masks(geometry)
        self.masks = {d: engine.encode_to_light_plaintext(m) for d, m in low.items()}
        self.complement_masks = {d: engine.encode_to_light_plaintext(m) for d, m in high.items()}

    def _encode_all(self, messages):
        out = np.empty(messages.shape, dtype=object)
        for index, message in np.ndenumerate(messages):
            out[index] = self.engine.encode_to_light_plaintext(message)
        return out

    def weight(self, w: np.ndarray, scale: float = 1.0):
        return self._encode_all(encode_weight(self.g, w, scale))

    def bias(self, b: np.ndarray, scale: float = 1.0):
        return self._encode_all(encode_bias(self.g, b, scale))

    def stages(self) -> Stages:
        return Stages(self.engine, self.g, masks=self.masks, complement_masks=self.complement_masks)


def encrypt_activations(engine, geometry: Geometry, x: np.ndarray, level: int = 0) -> np.ndarray:
    """Encode a (dim, features) activation matrix and encrypt it into ``n_input_ciphertexts``."""
    return np.array([engine.encrypt(m, level=level) for m in encode_activations(geometry, x)], dtype=object)


def decrypt_ciphertexts(engine, cts) -> list[np.ndarray]:
    return [engine.decrypt(ct) for ct in cts]
