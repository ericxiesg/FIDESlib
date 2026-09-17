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
from .numeric import ACTIVATION_SCALE, GELU_SCALE
from .softmax import Softmax

# `ACTIVATION_SCALE` is defined in `numeric`, beside the other two scales it has to agree with.

#: The key scaling folded into the key projection. One requirement fixes it: what reaches
#: ``he_softmax`` must be the BERT attention score itself, because a softmax is not scale-invariant
#: and ``he_softmax(x)`` approximates ``softmax(x)`` - which is the contract ``test_stage9`` pins, and
#: the units THOR's window is in (its [-27.2, 21.7] is the range of a BERT-base score).
#:
#: Counting the factors, with ``s = ACTIVATION_SCALE`` the amplitude the layer is entered at:
#:
#: * a projection gives ``s * (x @ W.T) + 2b`` - ``encode_weight`` halves the weight, the bias is
#:   added at single, and ``y + conj(y)`` doubles both - which at ``s = 2`` is ``2 *`` BERT's ``q``;
#: * stage 06 carries the product exactly, so it holds ``s**2 * (q.k) * scale``
#:   (``test_attention_score_is_exactly_q_k_transpose`` pins the stage itself to 1e-12);
#: * ``stage_07_softmax``'s bootstrap fold doubles it once more.
#:
#: So ``he_softmax`` sees ``2 * s**2 * (q.k) * scale = 8 * (q.k) * scale``, and BERT's
#: ``(q.k) / sqrt(64)`` needs ``scale = 1/64``.
#:
#: It was briefly 1/16, on an accounting that dropped the ``s**2``. What hid the factor is that
#: ``bench magnitudes --through 06`` entered the layer at amplitude 1: at ``s = 1`` the same
#: projection gives ``x @ W.T + 2b``, and against that reference the ratio reads 1.0000, so stage 06
#: looked like it carried ``(q.k) * scale`` when the real pipeline has it carry four times that.
#: At 1/16 the exponential is handed four times BERT's score and the denominator ``he_inv`` needs in
#: ``[epsilon, 1]`` reaches 534 on MRPC's first validation row - reproducible in plaintext to four
#: digits, which is what identified the factor.
#:
#: **Every layer gets the same scale, including layer 2.** THOR writes 1/512 for most layers and
#: 1/1024 for layer 2, and this port carried that factor of two across as `{2: scale / 2}`. It does
#: not belong here. The two polynomial paths land on the same exponent - `he_exp1` with `l = 2` and
#: `he_exp2` with `l = 4` both give `exp(u)` for the `u` handed to `he_softmax`, which is what the
#: module docstring of `thorfhe.softmax` derives - so both want BERT's score, and halving one of them
#: is a doubling of that layer's temperature. Measured on the real checkpoint, layer 2's wide path
#: against BERT's own softmax: 5.2e-4 handed the score, **0.61** handed half of it. THOR's constant
#: must be paying for something else in THOR; in these units it is simply wrong.
SOFTMAX_SCALES: dict[int, float] = {}
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


def encode_layer(parameters: dict, layer_index: int, *, residual_scale: float = 1.0,
                 score_refresh_scale: float = 1.0, qkv: Geometry = THOR_BERT,
                 dense: Geometry = THOR_ATTENTION_DENSE,
                 feedforward: Geometry = THOR_FEEDFORWARD) -> LayerWeights:
    """Encode one layer's BERT arrays. ``parameters`` uses HuggingFace's names, without the prefix.

    Expected keys: ``{query,key,value}.{weight,bias}``, ``attention.output.dense.{weight,bias}``,
    ``attention.output.LayerNorm.{weight,bias}``, ``intermediate.dense.{weight,bias}``,
    ``output.dense.{weight,bias}``, ``output.LayerNorm.{weight,bias}``.
    """
    def get(name):
        return np.asarray(parameters[name])

    # `score_refresh_scale` divides the scores so that stage 07 bootstraps a smaller number, and
    # stage 07 multiplies them back by an integer immediately afterwards - so `he_softmax` sees the
    # scores it always saw and no calibration moves. The key projection is where it goes because the
    # key feeds the scores and nothing else. See `Softmax.score_refresh_scale`.
    softmax_scale = SOFTMAX_SCALES.get(layer_index, DEFAULT_SOFTMAX_SCALE) / score_refresh_scale

    # `residual_scale` shrinks what stage 15 hands its bootstrap, and it is folded into plaintexts
    # rather than applied to a ciphertext, so it costs no levels and no operations.
    #
    # Three places, because `norm_1` is read twice. Stage 11's gamma and beta scale its output by
    # 1/s; stage 12's weight is scaled by s to undo that before GELU, which is non-linear and must
    # see exactly what it saw before; stage 14's weight and bias scale its output by 1/s so that
    # stage 15's `norm_1 + output_dense` comes out scaled as a whole. Stage 12's *bias* is not
    # touched: it is added after the product, which the two cancelling factors leave unchanged.
    #
    # Stage 16 absorbs the rest - it normalises, so its output is unaffected once its variance bounds
    # are divided by s^2 (see `LayerNormStages.residual_scale`).
    residual = 1.0 / residual_scale

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
        attention_norm=(encode_bias(dense, get("attention.output.LayerNorm.weight"), scale=residual),
                        encode_bias(dense, get("attention.output.LayerNorm.bias"), scale=residual)),
        intermediate=(encode_weight_ff(get("intermediate.dense.weight"), axis=0,
                                       scale=residual_scale / GELU_SCALE, **ff_kwargs),
                      intermediate_bias),
        output_dense=(encode_weight_ff(get("output.dense.weight"), axis=1, scale=residual,
                                       **ff_kwargs),
                      encode_bias(feedforward, get("output.dense.bias"), scale=residual)),
        output_norm=(encode_bias(feedforward, get("output.LayerNorm.weight")),
                     encode_bias(feedforward, get("output.LayerNorm.bias"))),
    )


class EncoderLayer:
    """Stages 01-16 over one engine, with one stage object per representation the layer passes through."""

    def __init__(self, engine, *, residual_scale: float = 1.0, score_refresh_scale: float = 1.0,
                 refresh_scale: float = 1.0, qkv: Geometry = THOR_BERT,
                 dense: Geometry = THOR_ATTENTION_DENSE,
                 feedforward: Geometry = THOR_FEEDFORWARD, binary_rotations: bool = False,
                 refresh_after_dense: bool = False, refresh_after_context: bool = False):
        self.engine = engine
        self.g_qkv, self.g_dense, self.g_ff = qkv, dense, feedforward

        #: optional ``(stage_name, ciphertexts) -> anything`` callback. When set, a trace records what
        #: it returns instead of the ciphertexts themselves, so tracing costs a decoded array per
        #: stage rather than a live copy of every stage's output.
        self.trace_sink = None

        #: optional ``(stage_name, device_memory_dict)`` callback, invoked at every stage boundary.
        #: A host-side model says which stage *should* be largest; this says what the device actually
        #: holds, which is the only way to tell a working set that is too big from a pool that is
        #: merely hoarding.
        self.memory_probe = None

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
        # Must match the `residual_scale` the weights were encoded with, or stage 16 is told to
        # expect a variance the input does not have - see `LayerNormStages.residual_scale`.
        self.norm.residual_scale = residual_scale
        # Same contract: must match what `encode_layer` was given, or the scores come out scaled.
        self.attention.score_refresh_scale = score_refresh_scale
        # `refresh` is the identity either way, so this one needs no matching change at encode time.
        self.norm.refresh_scale = refresh_scale
        self.feedforward = FeedForwardStages(engine, feedforward, masks=ff_low,
                                             complement_masks=ff_high,
                                             binary_rotations=binary_rotations)

        #: Insert a bootstrap between stages 10 and 11. Not THOR's, and semantically the
        #: identity - see `LayerNormStages.refresh`. It halves the layer's deepest level
        #: chain, which is the only thing that brings the depth inside a 32 GiB card.
        self.refresh_after_dense = refresh_after_dense
        #: refresh after stage 08 instead, where the reference implementation does.
        self.refresh_after_context = refresh_after_context

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

        The *query* side is masked as well, and that is not cosmetic. A padding query's row sums the
        exponential over the real keys, which is a perfectly finite number with no meaning - and
        `he_inv` then inverts it. On the real checkpoint those rows carry denominators up to 534
        where a real row carries 2e-3, and the device reported an inverse denominator whose *median*
        was 1.9e159: with 23 real tokens, 105 of the 128 rows are padding. Nothing downstream reads
        them, but they pass through the bootstraps, where a magnitude like that is not a rounding
        problem. Both conditions ride the same plaintext multiply, so the query side is free.
        """
        g = self.g_qkv
        count = 2 * g.n_output_ciphertexts
        if tokens is None or tokens >= g.dim:
            return [self.used_slots] * count

        group = np.arange(g.slot_count) // g.group_size
        tau = (np.arange(g.slot_count) % g.group_size) // g.n_slot
        query_is_real = tau < tokens
        return [self.used_slots * query_is_real * (((ct * g.pack + group + tau) % g.dim) < tokens)
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
                # A trace that holds ciphertexts holds every stage's output alive to the end of the
                # layer, which is the whole working set several times over - that is why --per-stage
                # used to run the card out of memory. With a sink the stage is read here, at the
                # boundary, and what survives is a small numpy array instead.
                trace[name] = self.trace_sink(name, value) if self.trace_sink is not None else value
            # A stage boundary is the one place the working set is reliably smaller than it was a
            # moment ago, so it is where draining the engine's pooled polynomials actually returns
            # memory rather than just handing it straight back out again.
            attention.release_pooled_memory()
            # Reading the device after the drain, not before, is the point: it is the memory that
            # would still be held if the next stage asked for more.
            if self.memory_probe is not None:
                self.memory_probe(name, attention.device_memory())
            return value

        # An intermediate stays alive as long as a local binds it, and `rotated` alone is 64
        # ciphertexts at nearly full level - about 2 GiB at depth 37, held through stages 03 to 06 for
        # no reason. Measured, stage 06 is the layer's true peak (5.43 GiB against a GV100's 3.7 GiB
        # of headroom), so releasing each intermediate where it dies is worth more than any single
        # rewrite inside a stage. When a trace is being taken it keeps them anyway; that mode is for
        # reading numbers, not for fitting on the card.
        def drop(*names):
            for name in names:
                scope[name] = None

        scope = {}

        attention, dense, norm, ff = self.attention, self.dense, self.norm, self.feedforward

        scope["residual"], scope["complexified"] = attention.stage_01_complexify_x(x, layer_index)
        scope["rotated"] = keep("rotated",
                                attention.stage_02_make_rotated_copies(scope["complexified"]))
        drop("complexified")
        scope["query"] = keep("query", attention.stage_03_query(scope["rotated"], *weights.query))
        scope["key"] = keep("key", attention.stage_04_key(scope["rotated"], *weights.key))
        scope["value"] = keep("value", attention.stage_05_value(scope["rotated"], *weights.value))
        drop("rotated")

        scores = keep("scores", attention.stage_06_attention_score(scope["query"], scope["key"]))
        drop("query", "key")
        weighted = keep("softmax", attention.stage_07_softmax(scores, attention_mask, layer_index,
                                                           parameters=softmax_parameters))
        scores = None
        # The softmax diagonals are the layer's largest working set. Nothing reads them after this
        # except a trace, and a trace with a sink has already read them at the boundary above - so
        # only a trace that keeps ciphertexts forces them to be held.
        traced_live = trace is not None and self.trace_sink is None
        context = keep("context", attention.stage_08_attention_context(scope["value"], weighted,
                                                                       consume=not traced_live))
        drop("value")
        weighted = None

        if self.refresh_after_context:
            # Where THOR's own schedule and the EasyFHE reference refresh - immediately after A.V,
            # before the dense layer - and it would cost two bootstraps rather than the four that
            # `refresh_after_dense` costs, because the context is two ciphertexts where the dense
            # output is eight.
            #
            # Measured, it does not work here: on the clear engine a layer fails at depth 30, 33, 36
            # and 37 alike, where refreshing after the dense layer completes at 37. The reason is
            # that the reference makes *two* coupled choices - it refreshes here *and* its LayerNorm
            # bootstraps internally (both attention_layernorm and feed_forward_layernorm take a
            # bootstrap_program). Ours never does, because a 20-level budget lets it finish without
            # one. Taking the earlier refresh without the internal ones leaves the chain from here
            # through LayerNorm longer than any depth tried can hold.
            #
            # Kept, off, with the measurement recorded, because this is an obvious-looking idea - I
            # proposed it twice - and the next person should be able to see it was tried.
            context = keep("refreshed_context", norm.refresh(context))

        context_rotated = dense.stage_02_make_rotated_copies(context)
        context = None
        attention_dense = keep("attention_dense",
                               dense.stage_10_attention_dense(context_rotated,
                                                              *weights.attention_dense))
        context_rotated = None
        if self.refresh_after_dense:
            attention_dense = keep("refreshed_dense", norm.refresh(attention_dense))
        norm_1 = keep("norm_1", norm.stage_11_attention_layernorm(
            scope["residual"], attention_dense, *weights.attention_norm, self.norm_ones))
        drop("residual")
        attention_dense = None

        intermediate = keep("intermediate",
                            ff.stage_12_intermediate_dense(norm_1, *weights.intermediate))
        activated = keep("gelu", ff.stage_13_gelu(intermediate))
        intermediate = None
        output_dense = keep("output_dense",
                            ff.stage_14_output_dense(activated, *weights.output_dense))
        activated = None

        norm_2_input = keep("norm_2_input", norm.stage_15_prepare_layernorm(norm_1, output_dense))
        norm_1 = output_dense = None
        return keep("norm_2", norm.stage_16_output_layernorm(norm_2_input, *weights.output_norm,
                                                             self.norm_ones,
                                                             layer_index=layer_index))
