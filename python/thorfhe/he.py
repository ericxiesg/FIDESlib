"""Wiring THOR's stages onto a ``pyfideslib.Engine``.

Two things live here that the engine-agnostic modules cannot do on their own: turning the numpy
encodings into light plaintexts, and working out which rotation keys the layer needs and at which
level. Both come from the same place - a dry run of the stages on :class:`~thorfhe.clear.ClearEngine`,
which is exact and costs milliseconds - so the key schedule cannot drift away from the code that uses
the keys, the way THOR's hand-maintained ``rotation_contexts`` table can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clear import ClearEngine
from .encoding import block_diagonal_masks, encode_activations, encode_bias, encode_weight
from .geometry import Geometry
from .rotation import RotationBasis, factored_basis, key_levels, rotation_cost
from .stages import Stages


@dataclass(frozen=True)
class RotationPlan:
    """What the key set will be, and how the run must reach the indices it does not contain.

    ``levels`` goes to ``SetRotationKeyLevels``; ``basis`` goes to the layer as its
    ``binary_rotations``. They are returned together because they are two views of one decision: a
    plan built from one basis and spent by another is missing keys, and that failure does not raise -
    it decrypts to an unrelated value.
    """

    levels: dict[int, int]
    basis: "RotationBasis | bool"
    #: how many engine rotations a layer costs under this basis, when it is known
    rotations: int | None = None
    #: what the dry run asked for, before any basis decomposed it: ``{index: times used}`` and
    #: ``{index: highest level}``. Kept because choosing a basis is a *budget* question - the size of
    #: a truncated key depends on the level it is used at - and that cannot be re-derived from
    #: ``levels``, which is already the decomposition's own demand.
    demand: dict[int, int] | None = None
    demand_levels: dict[int, int] | None = None


def plan_rotations(geometry: Geometry, depth: int, layer_index: int = 0, *,
                   bootstrap_level: int | None = None, scope: str = "layer",
                   dense: Geometry | None = None,
                   feedforward: Geometry | None = None,
                   binary_rotations: bool = False,
                   extra_rotation_keys: int = 0,
                   refresh_after_dense: bool = False,
                   compact: bool = False,
                   key_cost=None,
                   rotation_key_budget: int | None = None,
                   rotation_max_steps: int = 4) -> "RotationPlan":
    """The rotation keys this scope needs, and how the run must reach the indices they do not cover.

    Derived by running the stages on the clear engine with dummy data: every ``rotate`` records the
    level of its operand, which is exactly what a level-truncated rotation key has to cover. This is
    THOR's ``rotation_contexts`` table, computed rather than transcribed, so it cannot drift away from
    the code that uses the keys.

    ``extra_rotation_keys`` asks for at most that many keys beyond the powers of two, chosen from the
    indices the dry run rotated by most (see :mod:`thorfhe.rotation`); the returned ``basis`` then has
    to go to the layer as its ``binary_rotations``, or the run will spend keys the plan did not build.
    With ``key_cost`` (``level -> bytes``, normally :func:`~thorfhe.budget.key_bytes` bound to the
    parameters) the extra keys are chosen by rotations removed per byte rather than per key, and
    ``rotation_key_budget`` caps the bytes the key set may reach - which is the form the question
    actually takes on a 32 GB card. ``rotation_max_steps`` is how far a single index may be reached
    through, and it is where most of the saving is: at four, the same six extra keys cost 3465
    rotations a layer instead of 4237, for 0.01 GiB. Six buys four more rotations for four times the
    planning, so it is not the default.

    ``scope="layer"`` covers a whole encoder layer (about 210 indices) and needs the production
    geometries, since stages 10-16 change representation. ``scope="qkv"`` covers stages 01-05 only
    (11 indices) and works for any geometry, which is what the small-geometry tests use.

    Zeros are safe as dummy data: the only value-dependent control flow in the layer is ``he_inv``'s
    iteration count, and that comes from scalars, not from the ciphertext.
    """
    g = geometry
    # With an explicit basis the dry run must use one key per index: its counts and levels have to
    # describe the rotations the layer *wants*, not the steps some other basis would have split them
    # into. Choosing a basis from a basis' own output would compound the decomposition.
    dry_binary = False if extra_rotation_keys > 0 else binary_rotations
    # The plan has to be made on the level schedule the run will use. `ClearEngine` defaults to a
    # bootstrap level of 14 and stages 12-14 alone need 17 (GELU is 13 of them), so leaving it at the
    # default sends the tail of the layer below level 0 and the plan comes out short.
    engine = ClearEngine(g, depth=depth,
                         bootstrap_level=depth - 10 if bootstrap_level is None else bootstrap_level)

    if scope == "qkv":
        low, high = block_diagonal_masks(g)
        stages = Stages(engine, g, masks=low, complement_masks=high,
                        binary_rotations=dry_binary)
        zeros_x = np.zeros((g.dim, g.features))
        x = np.array([engine.encrypt(m) for m in encode_activations(g, zeros_x)], dtype=object)
        _, x_cplx = stages.stage_01_complexify_x(x, layer_index=layer_index)
        rots = stages.stage_02_make_rotated_copies(x_cplx)
        stages.stage_03_query(rots, encode_weight(g, np.zeros((g.features, g.features))),
                              encode_bias(g, np.zeros((g.features,))))
    elif scope == "layer":
        from .geometry import THOR_ATTENTION_DENSE, THOR_FEEDFORWARD
        from .layer import EncoderLayer, encode_layer

        dense = THOR_ATTENTION_DENSE if dense is None else dense
        feedforward = THOR_FEEDFORWARD if feedforward is None else feedforward
        square = np.zeros((g.features, g.features))
        dummy = {"query.weight": square, "query.bias": np.zeros(g.features),
                 "key.weight": square, "key.bias": np.zeros(g.features),
                 "value.weight": square, "value.bias": np.zeros(g.features),
                 "attention.output.dense.weight": square,
                 "attention.output.dense.bias": np.zeros(g.features),
                 "attention.output.LayerNorm.weight": np.zeros(g.features),
                 "attention.output.LayerNorm.bias": np.zeros(g.features),
                 "intermediate.dense.weight": np.zeros((4 * g.features, g.features)),
                 "intermediate.dense.bias": np.zeros(4 * g.features),
                 "output.dense.weight": np.zeros((g.features, 4 * g.features)),
                 "output.dense.bias": np.zeros(g.features),
                 "output.LayerNorm.weight": np.zeros(g.features),
                 "output.LayerNorm.bias": np.zeros(g.features)}

        weights = encode_layer(dummy, layer_index, qkv=g, dense=dense, feedforward=feedforward,
                               lazy=compact)
        # `compact` has to match the run: `factored_basis` picks its extra keys from the *measured*
        # frequency of each index, and streaming the QKV copies makes them three times instead of
        # once. A basis chosen on the wrong profile spends keys where the run does not rotate.
        layer = EncoderLayer(engine, qkv=g, dense=dense, feedforward=feedforward,
                             binary_rotations=dry_binary,
                             refresh_after_dense=refresh_after_dense,
                             compact=compact)
        # The dummy weights are zeros, so every intermediate value is an artefact of that rather than
        # of the circuit: the scores are uniform and he_inv's denominator lands far below the range
        # real activations put it in. What is being measured here is which rotations happen at which
        # level, which the values do not affect - so turn the range check off rather than calibrate
        # dummy data to satisfy it.
        for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
            owner.check_ranges = False
        state = np.array([engine.encrypt(m)
                          for m in encode_activations(g, np.zeros((g.dim, g.features)))],
                         dtype=object)
        # No `except` here on purpose. Anything this raises is a real bug, and swallowing it would
        # hand back a *partial* plan - which fails on the GPU as a missing rotation key, the very
        # thing this function exists to prevent.
        layer.forward(state, weights, layer.padding_mask(g.dim), layer_index,
                      softmax_parameters=None)
    else:
        raise ValueError(f"scope must be 'layer' or 'qkv', got {scope!r}")

    plan = dict(engine.rotation_levels)
    starved = {index: level for index, level in plan.items() if level < 0}
    if starved:
        raise ValueError(
            f"depth={depth} / bootstrap_level={engine.bootstrap_level} cannot run this scope: "
            f"{len(starved)} rotations happen below level 0 (e.g. "
            f"{dict(list(starved.items())[:3])}). Stages 12-14 need 17 levels after the last "
            f"bootstrap, GELU being 13 of them. Dropping these silently would produce a plan that is "
            f"missing keys the layer actually uses.")

    if extra_rotation_keys > 0:
        # The dry run above used one key per index, so its counts and levels describe the rotations
        # the layer *wants*, independent of how they will be reached. Choose the basis from that, then
        # re-derive the plan through the basis' own decomposition so the keys built are exactly the
        # keys spent - see `thorfhe.rotation`.
        basis = factored_basis(engine.rotation_counts, geometry.slot_count,
                               extra_keys=extra_rotation_keys, max_steps=rotation_max_steps,
                               levels=plan, key_cost=key_cost, budget=rotation_key_budget)
        return RotationPlan(levels=key_levels(basis, plan), basis=basis,
                            rotations=rotation_cost(basis, engine.rotation_counts),
                            demand=dict(engine.rotation_counts), demand_levels=plan)
    return RotationPlan(levels=plan, basis=binary_rotations,
                        rotations=sum(engine.rotation_counts.values())
                        if not binary_rotations else None,
                        demand=dict(engine.rotation_counts), demand_levels=plan)


def plan_rotation_keys(*args, **kwargs) -> dict[int, int]:
    """``{rotation index: highest level it is used at}`` alone - see :func:`plan_rotations`."""
    return plan_rotations(*args, **kwargs).levels


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
