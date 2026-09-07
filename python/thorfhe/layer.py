"""One BERT encoder layer, stages 01 through 16, chained the way ``THOR/src/thor/forward.py`` does.

The stages do not all speak the same representation - the QKV projections are twelve blocks of 64, the
attention dense output six blocks of 128, and the feed-forward stages six of 128 in two windows - so
this composes one stage object per representation over a single engine. Ciphertexts are engine-level
objects, so they pass between them freely; only the encoders care which geometry they were built for.

``encode_layer`` is the port of ``encode_weights.pre_encode_*``: it takes a layer's raw BERT arrays and
returns every plaintext the chain needs, with THOR's scaling conventions applied where THOR applies
them (the key projection's ``softmax_scale``, the feed-forward's ``1/64``, the halving inside
``encode_weight``).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .attention import attention_rotate_masks, ccmm_masks, make_copies_masks, transpose_masks
from .dense import DenseStages
from .encoding import (FF_SLOT_INDICES, block_diagonal_masks, encode_bias, encode_bias_raw,
                       encode_weight, encode_weight_ff, encode_weight_raw)
from .feedforward import FeedForwardStages
from .geometry import THOR_ATTENTION_DENSE, THOR_BERT, THOR_FEEDFORWARD, Geometry
from .layernorm import LayerNormStages, statistic_mask
from .numeric import GELU_SCALE
from .softmax import Softmax

#: The key scaling folded into the key projection. It is fixed by one requirement: what reaches
#: ``he_softmax`` must be the BERT attention score itself, because a softmax is not scale-invariant
#: and ``he_softmax(x)`` approximates ``softmax(x)`` (which is the contract ``test_stage9`` pins, and
#: the units THOR's window is in - its [-27.2, 21.7] is the range of a BERT-base score).
#:
#: Counting the factors under THOR's doubled-ciphertext convention: q and k are each carried at 2x,
#: stage 06's own masks contribute a half, and ``stage_07_softmax``'s bootstrap fold doubles again -
#: measured end to end, ``he_softmax`` sees ``4 * (q.k) * scale``. Setting ``scale = 1/64`` makes that
#: ``(q.k) / 16 * 4 == (q.k) / 8``, the score after BERT's ``1/sqrt(head_dim)``. Layer 2 keeps THOR's
#: factor of two. THOR writes 1/512 and 1/1024 here, which its own softmax must compensate elsewhere;
#: this port's value is the one its own stages measure, and ``--per-stage`` is how to re-check it.
SOFTMAX_SCALES = {2: 1 / 128}
DEFAULT_SOFTMAX_SCALE = 1 / 64


@dataclass
class LayerWeights:
    """Every plaintext one encoder layer consumes, in the order the stages ask for them."""

    query: tuple
    key: tuple
    value: tuple
    attention_dense: tuple
    attention_norm: tuple
    intermediate: tuple
    output_dense: tuple
    output_norm: tuple


def encode_layer(parameters: dict, layer_index: int, *, qkv: Geometry = THOR_BERT,
                 dense: Geometry = THOR_ATTENTION_DENSE,
                 feedforward: Geometry = THOR_FEEDFORWARD) -> LayerWeights:
    """Encode one layer's BERT arrays. ``parameters`` uses HuggingFace's names, without the prefix.

    Expected keys: ``{query,key,value}.{weight,bias}``, ``attention.output.dense.{weight,bias}``,
    ``attention.output.LayerNorm.{weight,bias}``, ``intermediate.dense.{weight,bias}``,
    ``output.dense.{weight,bias}``, ``output.LayerNorm.{weight,bias}``.
    """
    def get(name):
        return np.asarray(parameters[name])

    softmax_scale = SOFTMAX_SCALES.get(layer_index, DEFAULT_SOFTMAX_SCALE)

    ff_kwargs = dict(dim=feedforward.dim, pack=feedforward.pack, n_slot=feedforward.n_slot,
                     group_size=feedforward.group_size, slot_count=feedforward.slot_count,
                     n_in=feedforward.n_in, n_out=feedforward.n_out, split=4)

    intermediate_bias = np.empty((2, feedforward.n_output_ciphertexts), dtype=object)
    for rep, chunk in enumerate(np.split(get("intermediate.dense.bias"), 2)):
        intermediate_bias[rep] = encode_bias_raw(
            chunk, dim=feedforward.dim, pack=feedforward.pack, n_slot=feedforward.n_slot,
            group_size=feedforward.group_size, slot_count=feedforward.slot_count,
            n_out=feedforward.n_out, n_blocks=2 * feedforward.out_blocks,
            slot_indices=FF_SLOT_INDICES, scale=1.0 / GELU_SCALE)

    return LayerWeights(
        query=(encode_weight(qkv, get("query.weight")), encode_bias(qkv, get("query.bias"))),
        key=(encode_weight(qkv, get("key.weight"), scale=softmax_scale),
             encode_bias(qkv, get("key.bias"), scale=softmax_scale)),
        value=(encode_weight(qkv, get("value.weight")), encode_bias(qkv, get("value.bias"))),
        attention_dense=(
            encode_weight_raw(get("attention.output.dense.weight"), dim=dense.dim, pack=dense.pack,
                              n_slot=dense.n_slot, group_size=dense.group_size,
                              slot_count=dense.slot_count, n_in=dense.n_in, n_out=dense.n_out,
                              slot_indices=np.arange(dense.n_blocks)),
            encode_bias(dense, get("attention.output.dense.bias"))),
        attention_norm=(encode_bias(dense, get("attention.output.LayerNorm.weight")),
                        encode_bias(dense, get("attention.output.LayerNorm.bias"))),
        intermediate=(encode_weight_ff(get("intermediate.dense.weight"), axis=0,
                                       scale=1.0 / GELU_SCALE, **ff_kwargs), intermediate_bias),
        output_dense=(encode_weight_ff(get("output.dense.weight"), axis=1, **ff_kwargs),
                      encode_bias(feedforward, get("output.dense.bias"))),
        output_norm=(encode_bias(feedforward, get("output.LayerNorm.weight")),
                     encode_bias(feedforward, get("output.LayerNorm.bias"))),
    )


class EncoderLayer:
    """Stages 01-16 over one engine, with one stage object per representation the layer passes through."""

    def __init__(self, engine, *, qkv: Geometry = THOR_BERT,
                 dense: Geometry = THOR_ATTENTION_DENSE,
                 feedforward: Geometry = THOR_FEEDFORWARD, binary_rotations: bool = False):
        self.engine = engine
        self.g_qkv, self.g_dense, self.g_ff = qkv, dense, feedforward

        qkv_low, qkv_high = block_diagonal_masks(qkv)
        dense_low, dense_high = block_diagonal_masks(dense)
        ff_low, ff_high = block_diagonal_masks(feedforward)

        #: the used-slot indicator: the softmax denominator's Goldschmidt iteration starts from it,
        #: and it doubles as the padding mask the attention takes
        self.used_slots = ((np.arange(qkv.slot_count) % qkv.n_slot) < qkv.n_blocks).astype(float)

        self.attention = Softmax(engine, qkv, masks=qkv_low, complement_masks=qkv_high,
                                 transpose=transpose_masks(qkv), copies=make_copies_masks(qkv),
                                 attention=attention_rotate_masks(qkv), ccmm=ccmm_masks(qkv),
                                 ones=engine.encrypt(self.used_slots),
                                 binary_rotations=binary_rotations)
        self.dense = DenseStages(engine, dense, masks=dense_low, complement_masks=dense_high,
                                 binary_rotations=binary_rotations)
        self.norm = LayerNormStages(engine, dense, masks=dense_low,
                                    complement_masks=dense_high,
                                    binary_rotations=binary_rotations)
        self.feedforward = FeedForwardStages(engine, feedforward, masks=ff_low,
                                             complement_masks=ff_high,
                                             binary_rotations=binary_rotations)

        #: LayerNorm's own starting point is the *slot-0* indicator, not the used-slot one
        self.norm_ones = engine.encrypt(statistic_mask(dense))

    def padding_mask(self, tokens: int | None = None):
        """The attention mask stage 07 takes: one plaintext per score ciphertext.

        It is a list, not a single mask, because the scores are held as *diagonals*: slot
        ``(group, tau, block)`` of score ciphertext ``ct`` carries key position
        ``(ct * pack + group + tau) mod dim``. A mask keyed on the slot alone can only express "this
        query token is padding"; the softmax denominator needs "this *key* position is padding", and
        that depends on the ciphertext index. Getting this wrong is quiet - the chain still runs, and
        the denominator simply sums the exponential over all 128 key positions.
        """
        g = self.g_qkv
        count = 2 * g.n_output_ciphertexts
        if tokens is None or tokens >= g.dim:
            return [self.used_slots] * count

        group = np.arange(g.slot_count) // g.group_size
        tau = (np.arange(g.slot_count) % g.group_size) // g.n_slot
        return [self.used_slots * (((ct * g.pack + group + tau) % g.dim) < tokens)
                for ct in range(count)]

    def forward(self, x, weights: LayerWeights, attention_mask, layer_index: int, *,
                trace=None, softmax_parameters=None):
        """One layer. ``x`` is the eight real ciphertexts a previous layer (or stage 01) produces.

        ``trace``, if given, is a dict that collects each stage's output for debugging.
        ``softmax_parameters`` overrides THOR's per-layer calibration, which is what a run on
        activations other than the ones THOR calibrated on needs - see :func:`softmax.calibrate`.
        """
        def keep(name, value):
            if trace is not None:
                trace[name] = value
            return value

        attention, dense, norm, ff = self.attention, self.dense, self.norm, self.feedforward

        residual, complexified = attention.stage_01_complexify_x(x, layer_index)
        rotated = keep("rotated", attention.stage_02_make_rotated_copies(complexified))
        query = keep("query", attention.stage_03_query(rotated, *weights.query))
        key = keep("key", attention.stage_04_key(rotated, *weights.key))
        value = keep("value", attention.stage_05_value(rotated, *weights.value))

        scores = keep("scores", attention.stage_06_attention_score(query, key))
        weighted = keep("softmax", attention.stage_07_softmax(scores, attention_mask, layer_index,
                                                           parameters=softmax_parameters))
        context = keep("context", attention.stage_08_attention_context(value, weighted))

        context_rotated = dense.stage_02_make_rotated_copies(context)
        attention_dense = keep("attention_dense",
                               dense.stage_10_attention_dense(context_rotated,
                                                              *weights.attention_dense))
        norm_1 = keep("norm_1", norm.stage_11_attention_layernorm(
            residual, attention_dense, *weights.attention_norm, self.norm_ones))

        intermediate = keep("intermediate",
                            ff.stage_12_intermediate_dense(norm_1, *weights.intermediate))
        activated = keep("gelu", ff.stage_13_gelu(intermediate))
        output_dense = keep("output_dense",
                            ff.stage_14_output_dense(activated, *weights.output_dense))

        norm_2_input = keep("norm_2_input", norm.stage_15_prepare_layernorm(norm_1, output_dense))
        return keep("norm_2", norm.stage_16_output_layernorm(norm_2_input, *weights.output_norm,
                                                             self.norm_ones,
                                                             layer_index=layer_index))
