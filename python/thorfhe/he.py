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
    """``{rotation index: highest level it is used at}`` for stages 01-05, ready for ``SetRotationKeyLevels``.

    Derived by running the stages on the clear engine with dummy data: every ``rotate`` records the
    level of its operand, which is exactly what a level-truncated rotation key has to cover. This is
    THOR's ``rotation_contexts`` table, computed rather than transcribed.
    """
    g = geometry
    engine = ClearEngine(g, depth=depth)
    low, high = block_diagonal_masks(g)
    stages = Stages(engine, g, masks=low, complement_masks=high)

    zeros_x = np.zeros((g.dim, g.features))
    zeros_w = np.zeros((g.features, g.features))
    zeros_b = np.zeros((g.features,))

    x = np.array([engine.encrypt(m) for m in encode_activations(g, zeros_x)], dtype=object)
    x8, x_cplx = stages.stage_01_complexify_x(x, layer_index=layer_index)
    rots = stages.stage_02_make_rotated_copies(x_cplx)
    stages.stage_03_query(rots, encode_weight(g, zeros_w), encode_bias(g, zeros_b))
    return dict(engine.rotation_levels)


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
