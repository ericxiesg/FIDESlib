"""``python -m thorfhe.bench`` - the THOR/fideslib benchmark: speed, accuracy and fidelity on MRPC.

Four commands:

* ``info``      - what the proxy, endpoint and cache are set to, and what is already cached.
* ``fetch``     - download the checkpoint and the dataset, then stop. Warms the cache on a machine
                  that has the proxy, so the measured run does no network I/O.
* ``reference`` - the plaintext numpy model over MRPC: accuracy, F1, wall clock. This is the ground
                  truth the encrypted run is scored against, so it is worth running first.
* ``fhe``       - the encrypted pipeline. Runs ``--layers`` encoder layers under CKKS and the rest in
                  plaintext, so the cost can be paid one layer at a time; reports time, accuracy and
                  fidelity against the plaintext model.

Fidelity and accuracy are different questions and are reported separately - see :mod:`thorfhe.metrics`.

Every network call goes through :class:`~thorfhe.hub.HubClient`, which takes ``--proxy`` (or the usual
environment variables) and caches under ``--cache-dir``, so a second run is offline-capable via
``--offline``.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from .bert import BertConfig, BertForSequenceClassification, layer_parameters, layer_norm
from .checkpoint import load_state_dict
from .encoding import FF_SLOT_INDICES, decode_linear_output, encode_activations
from .layer import ACTIVATION_SCALE
from .geometry import (FEEDFORWARD_WINDOW, THOR_ATTENTION_DENSE, THOR_BERT,
                       THOR_FEEDFORWARD)
from .hub import HubClient
from .metrics import (Classification, Fidelity, OpTimings, ProbabilityAgreement, Timings,
                      as_dict, instrument)
from .tokenizer import BertTokenizer

DEFAULT_MODEL = "textattack/bert-base-uncased-MRPC"
DEFAULT_DATASET = "nyu-mll/glue"
DEFAULT_CONFIG = "mrpc"
DEFAULT_SPLIT = "validation"
#: THOR packs 128 tokens; a different length would need a different geometry.
SEQUENCE_LENGTH = THOR_BERT.dim

CHECKPOINT_CANDIDATES = ("model.safetensors", "pytorch_model.bin")
TOKENIZER_FILES = ("vocab.txt",)


# ---------------------------------------------------------------------- plumbing
@contextmanager
def timed(timings: Timings, phase: str):
    start = time.perf_counter()
    yield
    timings.add(phase, time.perf_counter() - start)


def build_hub(args) -> HubClient:
    return HubClient(cache_dir=args.cache_dir, proxy=args.proxy, token=args.token,
                     endpoint=args.hf_endpoint, rows_endpoint=args.rows_endpoint,
                     offline=args.offline, timeout=args.timeout, retries=args.retries,
                     quiet=args.quiet)


def load_model(hub: HubClient, args):
    """Checkpoint, config and tokenizer, all from the cache once ``fetch`` has run."""
    config_path = hub.file(args.model, "config.json", revision=args.revision)
    weights_path = hub.first_file(args.model, CHECKPOINT_CANDIDATES, revision=args.revision)
    vocab_path = hub.file(args.model, "vocab.txt", revision=args.revision)
    try:
        tokenizer_config = hub.file(args.model, "tokenizer_config.json", revision=args.revision)
    except Exception:                                  # noqa: BLE001 - optional file
        tokenizer_config = None

    config = BertConfig.from_file(config_path)
    state = load_state_dict(weights_path)
    model = BertForSequenceClassification(state, config)
    tokenizer = BertTokenizer.from_files(vocab_path, tokenizer_config)
    return model, tokenizer, config, weights_path


def load_rows(hub: HubClient, args):
    rows = hub.rows(args.dataset, args.config, args.split, limit=args.limit)
    return rows[:args.limit] if args.limit else rows


def encode_rows(tokenizer: BertTokenizer, rows, length: int):
    """Tokenize once; both paths then work from the same ids."""
    encoded = []
    for row in rows:
        ids, types, mask = tokenizer.encode_pair(row["sentence1"], row.get("sentence2"),
                                                 max_length=length)
        encoded.append((np.array(ids), np.array(types), np.array(mask), int(row["label"])))
    return encoded


# ---------------------------------------------------------------------- plaintext
def run_reference(model, encoded, timings: Timings, *, trace=False):
    logits, traces = [], []
    with timed(timings, "plaintext forward"):
        for ids, types, mask, _label in encoded:
            out, states = model.forward(ids, types, mask, trace=trace)
            logits.append(out)
            if trace:
                traces.append(states)
    return np.array(logits), traces


# ---------------------------------------------------------------------- encrypted
def describe_rotations(args, rotations=None):
    """One phrase naming the rotation basis, for the budget report's header line."""
    if args.extra_rotation_keys:
        count = "" if rotations is None else f", {rotations.rotations}/layer"
        return f"binary + {args.extra_rotation_keys} measured{count}"
    return "binary" if args.binary_rotations else "one key per index"


def _instrumented(args, engine):
    """Wrap the engine's primitives when --time-ops asked for it, and remember where the totals are.

    A phase total says a layer took 1131 seconds. This says which primitive did, which is the
    question a device sitting at 0% utilisation actually poses.
    """
    if not getattr(args, "time_ops", False):
        return engine
    args._op_timings = OpTimings()
    return instrument(engine, args._op_timings)


def plaintext_cache_tag(args) -> str:
    """Everything that changes what a weight encodes to, folded into the directory name.

    The scales are baked into the plaintexts by `encode_layer`, and the ring and scaling parameters
    decide the coefficients, so a cache written under one set must not be read under another. Naming
    them rather than hashing the arrays is the whole point of keying by provenance - but then the
    name has to carry everything the provenance depends on.
    """
    parts = [str(getattr(args, "model", "model")).replace("/", "_"),
             f"n{args.log_n}", f"d{args.depth}", f"sb{args.scaling_bits}",
             f"fmb{args.first_mod_bits}", f"rs{args.residual_scale:g}",
             f"srs{args.score_refresh_scale:g}"]
    return "-".join(parts)


def rotation_key_cost(args):
    """``(level -> bytes, budget in bytes or None)`` for choosing the extra rotation keys.

    Without a budget the greedy spends its whole allowance on whichever index is rotated by most,
    which on this layer is one used at depth - the most expensive key there is. Priced, it can prefer
    two keys used below the bootstrap level that together cost less and remove more.
    """
    from .budget import key_bytes
    special = getattr(args, "special_primes", 11)
    if getattr(args, "no_truncate", False):
        def cost(level, _depth=args.depth):
            return key_bytes(log_n=args.log_n, level=_depth, special_primes=special,
                             dnum=args.dnum)
    else:
        def cost(level, _depth=args.depth):
            return key_bytes(log_n=args.log_n, level=min(level + 1, _depth),
                             special_primes=special, dnum=args.dnum)
    gib = getattr(args, "rotation_key_budget", None)
    return cost, None if gib is None else int(gib * (1 << 30))


def rotation_mode(args):
    """What a layer takes as its ``binary_rotations``.

    With ``--extra-rotation-keys`` this is the basis the key plan was built from, not a flag: the
    plan and the run have to agree on which keys exist, and a run that reaches for a key the plan
    did not build fails as a wrong plaintext rather than as an error.
    """
    basis = getattr(args, "_rotation_basis", None)
    return args.binary_rotations if basis is None else basis


def make_engine(args, geometry):
    """The clear (numpy) engine, or fideslib when it is built and asked for."""
    if args.engine == "clear":
        from .clear import ClearEngine
        level = (args.depth - resolve_bootstrap_depth(args) if args.bootstrap_level is None
                 else args.bootstrap_level)
        if args.extra_rotation_keys:
            from .he import plan_rotations
            cost, budget = rotation_key_cost(args)
            # the plan has to be made on the configuration that will run: `factored_basis` chooses
            # its extra keys from measured rotation frequencies, and `compact` changes them
            args._rotation_basis = plan_rotations(
                geometry, depth=args.depth, bootstrap_level=level,
                extra_rotation_keys=args.extra_rotation_keys,
                refresh_after_dense=args.refresh_after_dense,
                compact=args.compact, key_cost=cost, rotation_key_budget=budget,
                rotation_max_steps=args.rotation_max_steps).basis
        # The noise model off by default - the clear engine's job is to prove the schedule and the
        # algebra, and exact arithmetic is what makes a failure there unambiguous. Switched on, it
        # carries the *device's* bootstrap error, which is what turns a 19-minute GPU run that comes
        # back at 7.5e17 into a five-minute laptop run that comes back at 7.5e17.
        return _instrumented(args, ClearEngine(
                           geometry, depth=args.depth, bootstrap_level=level,
                           strict=not args.lenient, noise_model=args.noise_model,
                           scaling_bits=args.scaling_bits, first_mod_bits=args.first_mod_bits,
                           bootstrap_precision_bits=args.bootstrap_precision_bits,
                           bootstrap_noise_only=args.bootstrap_noise_only))

    import pyfideslib

    from .he import plan_rotations

    # The rotation plan is computed, not hand-written: `plan_rotation_keys` runs the same layer on the
    # clear engine and records the level every rotation actually happens at, so the key set cannot
    # drift from the code that uses it. With --binary-rotations that is 15 indices instead of 210.
    # On hardware the post-bootstrap level is not a free parameter: EvalBootstrap returns a
    # ciphertext at `depth - GetBootstrapDepth()`. Planning against a different number silently
    # builds keys for levels the run never reaches - and, worse, hides that the level budget does not
    # fit at all. Derive it, and only let --bootstrap-level override it deliberately.
    key_cost, key_budget = rotation_key_cost(args)
    achievable = args.depth - resolve_bootstrap_depth(args)
    level = achievable if args.bootstrap_level is None else args.bootstrap_level
    if level > achievable and not args.quiet:
        print(f"warning: --bootstrap-level {level} is above what depth {args.depth} can give "
              f"({achievable} = depth - {resolve_bootstrap_depth(args)}). The plan will assume levels the "
              f"hardware never reaches.", file=sys.stderr)
    rotations = plan_rotations(geometry, depth=args.depth, bootstrap_level=level,
                               binary_rotations=args.binary_rotations,
                               extra_rotation_keys=args.extra_rotation_keys,
                               refresh_after_dense=args.refresh_after_dense,
                               compact=args.compact,
                               key_cost=key_cost, rotation_key_budget=key_budget,
                               rotation_max_steps=args.rotation_max_steps)
    plan = rotations.levels
    args._rotation_basis = rotations.basis if args.extra_rotation_keys else None
    if args.extra_rotation_keys and not args.quiet:
        print(f"rotation basis: {len(plan)} keys, {rotations.rotations} rotations per layer",
              file=sys.stderr)
    budget = None if args.no_bootstrap else tuple(args.bootstrap_level_budget)
    dist = (pyfideslib.SPARSE_TERNARY if args.secret_key_dist == "sparse"
            else pyfideslib.UNIFORM_TERNARY)

    if not args.quiet:
        from .budget import estimate
        predicted = estimate(log_n=args.log_n, depth=args.depth, dnum=args.dnum,
                             rotation_levels=plan, level_budget=budget)
        print(f"\npredicted GPU footprint ({len(plan)} rotation keys):", file=sys.stderr)
        print(predicted.format(args.card_gib * (1 << 30)), file=sys.stderr, flush=True)

    return _instrumented(args, pyfideslib.Engine(
                             args.device, log_n=args.log_n, depth=args.depth,
                             scaling_bits=args.scaling_bits, first_mod_bits=args.first_mod_bits,
                             dnum=args.dnum, rotation_indexes=plan,
                             bootstrap_level_budget=budget, bootstrap_level=level,
                             secret_key_dist=dist,
                             light_plaintext_cache=args.light_plaintext_cache,
                             truncate_keys=not args.no_truncate_keys,
                             allow_key_grow=args.allow_key_grow))


def decode_six_blocks(engine, ciphertexts, geometry=THOR_ATTENTION_DENSE):
    """Read a (dim, features) matrix out of the eight ciphertexts a LayerNorm leaves behind."""
    g = geometry
    out = np.zeros((g.dim, g.features))
    for ct in range(g.n_output_ciphertexts):
        slots = np.real(engine.decrypt(ciphertexts[ct]))
        for group in range(g.pack):
            for token in range(g.dim):
                for block in range(g.out_blocks):
                    feature = g.n_out * block + (ct * g.pack + group + token) % g.n_out
                    out[token, feature] = slots[g.slot(group, token, block)]
    return out


def decode_twelve_blocks(engine, ciphertexts, geometry=THOR_FEEDFORWARD):
    """The two-window layout stages 12 and 13 leave behind: 24 blocks of 128 over 2 reps x 12 slots."""
    g = geometry
    out = np.zeros((g.dim, 4 * g.features))
    for rep in range(2):
        for ct in range(g.n_output_ciphertexts):
            slots = np.real(engine.decrypt(ciphertexts[rep, ct]))
            for group in range(g.pack):
                for token in range(g.dim):
                    for position, slot in enumerate(FF_SLOT_INDICES):
                        block = 2 * FEEDFORWARD_WINDOW * rep + position
                        feature = g.n_out * block + (ct * g.pack + group + token) % g.n_out
                        out[token, feature] = slots[g.slot(group, token, slot)]
    return out


def decode_score_diagonals(engine, ciphertexts, geometry=THOR_BERT):
    """Stage 06's layout: `out[ct][g, tau, b] = S[b][tau, (ct*pack + g + tau) % dim]`."""
    g = geometry
    out = np.zeros((g.n_blocks, g.dim, g.dim))
    for ct in range(2 * g.n_output_ciphertexts):
        slots = np.real(engine.decrypt(ciphertexts[ct]))
        for group in range(g.pack):
            diagonal = ct * g.pack + group
            for tau in range(g.dim):
                for block in range(g.n_blocks):
                    out[block, tau, (diagonal + tau) % g.dim] = slots[g.slot(group, tau, block)]
    return out


def decode_broadcast_diagonals(engine, ciphertexts, geometry=THOR_BERT):
    """Stage 07's output: one ciphertext per diagonal, the weight broadcast across every group."""
    g = geometry
    out = np.zeros((g.n_blocks, g.dim, g.dim))
    for diagonal in range(g.dim):
        slots = np.real(engine.decrypt(ciphertexts[diagonal]))
        for tau in range(g.dim):
            for block in range(g.n_blocks):
                out[block, tau, (diagonal + tau) % g.dim] = slots[g.slot(0, tau, block)]
    return out


#: How each traced stage is read back, and what the plaintext model calls the same thing. The factor
#: is what THOR's own conventions put in front of it - the QKV bias is not halved so those stages give
#: `x @ w.T + 2b`, and stage 12 carries the pre-activation divided by GELU's 64.
STAGE_READERS = {
    # Under THOR's doubled-ciphertext convention every one of these should come back with a
    # best-fit scale of exactly 2 - except `softmax`, which is a probability and so is 1.
    "query": ("query", lambda e, c: decode_linear_output(THOR_BERT, [e.decrypt(x) for x in c])),
    "value": ("value", lambda e, c: decode_linear_output(THOR_BERT, [e.decrypt(x) for x in c])),
    "scores": ("scores_unmasked", decode_score_diagonals),
    "softmax": ("attention", decode_broadcast_diagonals),
    "attention_dense": ("attention_dense", decode_six_blocks),
    "norm_1": ("norm_1", decode_six_blocks),
    "intermediate": ("intermediate", decode_twelve_blocks),
    "gelu": ("gelu", decode_twelve_blocks),
    "output_dense": ("output_dense", decode_six_blocks),
    "norm_2": ("norm_2", decode_six_blocks),
}


def _real_tokens(array, tokens):
    """Drop the padded rows (and columns, for a score matrix) before comparing.

    Both paths compute something for the padding, and neither computes the same thing: the plaintext
    reference softmaxes a row of -10000s into a uniform 1/128, while THOR masks those slots to zero.
    Comparing them measures nothing, and at 128 positions against about 25 real ones it dominates
    every average.
    """
    array = np.asarray(array, dtype=np.float64)
    if array.ndim == 3:                       # (heads, query, key)
        return array[:, :tokens, :tokens]
    return array[:tokens]


def per_stage_fidelity(engine, trace, reference, tokens):
    """Fidelity at every stage that has a plaintext counterpart, with its best-fit scale.

    The scale is the point: THOR carries factors of two around (an unhalved bias here, a LayerNorm
    doubling there), so a stage can be exactly right up to a constant. A best-fit scale near a clean
    ratio with a small residual means the stage is correct and the convention is known; a scale near
    one with a large residual means it is not.
    """
    rows = []
    for name, (reference_name, reader) in STAGE_READERS.items():
        if name not in trace or reference_name not in reference:
            continue
        try:
            recorded = trace[name]
            # With a trace sink the stage was decoded at the boundary and this is already an array;
            # without one it is still ciphertexts and has to be read here.
            if isinstance(recorded, Exception):
                raise recorded
            got = recorded if isinstance(recorded, np.ndarray) else reader(engine, recorded)
        except Exception as error:                        # noqa: BLE001 - a decoder mismatch
            rows.append((name, None, None, str(error)))
            continue
        want = np.asarray(reference[reference_name], dtype=np.float64)
        if got.shape != want.shape:
            rows.append((name, None, None, f"shape {got.shape} vs {want.shape}"))
            continue
        got, want = _real_tokens(got, tokens), _real_tokens(want, tokens)
        scale = float((got * want).sum() / (want ** 2).sum()) if (want ** 2).sum() else 1.0
        rows.append((name, scale, Fidelity.between(got / (scale or 1.0), want), None))
    return rows


def softmax_parameters_for(reference_scores, layer_index, args, tokens):
    """THOR's per-layer constants, or a window derived from the plaintext scores.

    What reaches ``he_softmax`` is the BERT attention score itself - see ``SOFTMAX_SCALES``, and
    ``--per-stage`` prints the measured factor - so the calibration is over exactly those scores,
    restricted to the real tokens. The padded positions carry an arbitrary q.k that THOR masks off
    after the exponential; letting them into the window widens it for nothing and moves the centre.
    """
    from .softmax import calibrate

    if not args.calibrate or reference_scores is None:
        return None                       # let stage_07 use THOR's own table
    reaching = np.asarray(reference_scores, dtype=np.float64)[:, :tokens, :tokens]
    return calibrate(reaching)


def print_stage_rows(index, rows, stream=sys.stderr):
    """One layer's per-stage fidelity, printed where it is produced and again in the summary."""
    print(f"\nper-stage fidelity, layer {index} (sample 0), each rescaled by its best fit",
          file=stream, flush=True)
    for name, scale, fidelity, error in rows:
        if error is not None:
            print(f"  {name:<22}could not compare: {error}", file=stream, flush=True)
        else:
            print(f"  {name:<22}scale {scale:8.4f}   {fidelity.format()}", file=stream, flush=True)


def run_encrypted(model, encoded, args, timings: Timings, traces):
    """Encrypt, run ``args.layers`` encoder layers, decrypt, finish in plaintext."""
    from .layer import EncoderLayer, encode_layer
    from .plaintext_store import store_for

    config = model.config
    hidden_all, logits = [], []

    # The engine comes first now, because the plaintext store encodes through it. Nothing is built
    # here: `encode_layer` is handed the store and returns fields that encode - or load - on access.
    engine = make_engine(args, THOR_BERT)
    store = store_for(engine, args.plaintext_cache, tag=plaintext_cache_tag(args))

    with timed(timings, "encode weights"):
        # Every layer up front is 9.7 GB each at THOR's geometry - the plaintexts are one value per
        # slot and there are 32768 of them - so twelve layers is 116 GB and one is more than a laptop
        # has. `--lazy-weights` encodes each field as the layer reads it and holds none, which is
        # 3.2 GB peak and the difference between measuring accuracy here and not measuring it.
        weights = [encode_layer(layer_parameters(model.state, index), index,
                                residual_scale=args.residual_scale,
                                score_refresh_scale=args.score_refresh_scale,
                                lazy=args.lazy_weights or args.compact, store=store)
                   for index in range(args.layers)]

    layer = EncoderLayer(engine, residual_scale=args.residual_scale,
                         refresh_scale=args.refresh_scale,
                         score_refresh_scale=args.score_refresh_scale,
                         binary_rotations=rotation_mode(args),
                         refresh_after_dense=args.refresh_after_dense,
                         compact=args.compact,
                         boundary_refresh_scale=args.boundary_refresh_scale)
    layer.attention.inverse_lift = args.inverse_lift
    def magnitude_probe(name, value, mask=None):
        """Report an intermediate's magnitude and level. No reference needed, and that is the point.

        A stage that is wrong is wrong somewhere, and the plaintext model has nothing to compare its
        insides against. But a softmax numerator that should be in [0, 1] and comes back at 1e12, or
        a denominator that is constant across tokens, or a level that is not what the schedule says,
        each name a specific step without any reference at all.
        """
        flat = [ct for ct in np.asarray(value, dtype=object).ravel() if ct is not None]
        if not flat:
            return
        slots = np.concatenate([np.real(np.asarray(engine.decrypt(ct))).ravel() for ct in flat])
        finite = np.isfinite(slots)
        levels = sorted({engine.level(ct) for ct in flat})
        head = f"  [probe] {name:<26} {len(flat):3d} ct  level {levels[0] if len(levels) == 1 else levels}"
        if not finite.any():
            print(f"{head}  ALL NON-FINITE", file=sys.stderr, flush=True)
            return
        good = slots[finite]
        # Quantiles of |x| rather than a mean, and rather than a median over the non-zero slots.
        # Most slots carry no data, and on the clear engine they are exactly zero - but on the device
        # they are the encryption noise, so "non-zero" stops separating the two populations just when
        # it matters. The spread does separate them: a value that diverges in the slots that carry
        # data moves p50, while one that diverges only in the padding moves max alone. That
        # distinction is the difference between a broken circuit and a probe reading the noise.
        magnitude = np.abs(good)
        p50, p99 = np.quantile(magnitude, [0.5, 0.99])
        # How many slots are negative, and how many are above one. Every iterative numeric in the
        # port - `he_inv`, `he_invsqrt` - is derived for an input in `[epsilon, 1]` and diverges
        # outside it, in whichever slots stepped out. `min` and `max` say it happened somewhere;
        # these say how much of the vector it is, which is the difference between "a few slots are
        # out of range" and "the whole thing is wrong".
        outside = ""
        below, above = int((good < 0).sum()), int((good > 1.0).sum())
        if below or above:
            outside = f"  <0 {below}/{good.size}  >1 {above}/{good.size}"
            # Where, not just how many. The device's first `he_inv` converges in every slot but one
            # or two - `>1 1/32768` with a 99th percentile three orders below the maximum - and a
            # count cannot say whether that slot is a padding row, a group boundary or slot zero,
            # while an index can. Slots are reported modulo the geometry's `n_slot` as well, because
            # the layout repeats and the residue is usually the informative half.
            far = np.argsort(-np.abs(good))[:3]
            outside += "  worst at " + ",".join(
                f"{int(i)}(%{int(i) % THOR_BERT.n_slot})" for i in far)
        # Split by the carried slots when the caller knows them. `07a`'s maximum is 37.5 on the
        # device against 12.43 here, and whether that sits in a real token or in padding decides
        # what to look at next - per-stage fidelity only ever compares the carried ones.
        if mask is not None:
            carried = np.tile(np.asarray(mask, dtype=float).ravel(),
                              len(flat))[:slots.size][finite] > 1e-9
            if carried.any() and not carried.all():
                outside += (f"  carried max {np.abs(good[carried]).max():.4g}"
                            f"  padding max {np.abs(good[~carried]).max():.4g}")
        print(f"{head}  min {good.min():+.4g}  max {good.max():+.4g}"
              f"  |x| p50 {p50:.4g}  p99 {p99:.4g}{outside}"
              + (f"  NON-FINITE {(~finite).sum()}/{slots.size}" if not finite.all() else ""),
              file=sys.stderr, flush=True)

    def stage_sink(name, value):
        """Decode a stage at its boundary, so the trace keeps an array instead of ciphertexts.

        A stage with no reader is not compared against anything, so there is nothing to keep: return
        None and let the ciphertexts go. A decoder that fails is reported by per_stage_fidelity, so
        the exception is carried rather than raised here.
        """
        entry = STAGE_READERS.get(name)
        if entry is None:
            return None
        try:
            return entry[1](engine, value)
        except Exception as error:                        # noqa: BLE001 - reported per stage later
            return error

    if args.device_memory:
        gib = float(1 << 30)

        def probe(stage, memory):
            if not memory:
                return  # CPU engine: nothing to report
            held = memory["pooled"] - memory["in_use"]
            print(f"  [mem] {stage:<32} pool {memory['pooled'] / gib:5.2f} GiB "
                  f"(in use {memory['in_use'] / gib:5.2f}, reclaimable {held / gib:5.2f})  "
                  f"driver free {memory['driver_free'] / gib:5.2f} GiB", flush=True)

        # The baseline, before any of the circuit has run: whatever is gone by now is keys, plaintexts
        # and the engine's own tables. It is the number the per-stage lines have to be read against -
        # a stage that fails with little free is only the circuit's fault if there was room to begin
        # with, and that distinction is not visible from the failure alone.
        probe("(after key generation)", layer.attention.device_memory())
        layer.memory_probe = probe
    stage_rows = []

    for sample, (ids, types, mask, _label) in enumerate(encoded):
        hidden = model.embeddings(np.asarray(ids), np.asarray(types))
        # THOR masks the padding inside the softmax rather than with a -10000 in the score, so the
        # real token count has to reach stage 07 - an all-ones mask attends over all 128 positions.
        padding = layer.padding_mask(int(np.asarray(mask).sum()))

        with timed(timings, "encrypt"):
            # THOR's invariant: every ciphertext carries *twice* the value it represents. The weight
            # encoders halve, the bias is added before the `y + conj(y)` that doubles - so the bias
            # is single relative to a doubled input - and LayerNorm's uncancelled doubling puts the
            # next layer back on the same footing. Layer 0 has to be entered on that footing too.
            state = encode_activations(THOR_BERT, args.output_scale * hidden)
            state = np.array([engine.encrypt(message) for message in state], dtype=object)

        for index in range(args.layers):
            parameters = softmax_parameters_for(
                traces[sample][f"layer_{index}"]["scores_unmasked"] if traces else None,
                index, args, int(np.asarray(mask).sum()))
            if sample == 0 and not args.quiet:
                window = ("THOR's table" if parameters is None
                          else f"calibrated [{parameters['min_x']:.2f}, {parameters['max_x']:.2f}], "
                               f"inv_epsilon 2^{np.log2(parameters['inv_epsilon']):.0f}")
                print(f"  layer {index} softmax window: {window}", file=sys.stderr, flush=True)
            trace = {} if args.per_stage and sample == 0 else None
            # Read each stage where it is produced rather than at the end of the layer. Holding all
            # of them is the whole layer's working set several times over, which is what made
            # --per-stage impossible to run on the card that most needs it.
            layer.trace_sink = stage_sink if trace is not None else None
            # The probes look inside a stage, which only matters when a stage is the one that is
            # wrong; they cost a handful of decryptions, so they ride along with --per-stage.
            # every stage owner, not just the attention: the softmax's `07a`-`07d` were the only
            # probes for a long time because they were the only ones that could fire, and stage 11
            # is where a device run diverges
            for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
                owner.probe = magnitude_probe if trace is not None else None
                # The masks have no provenance to key on, so they stay content-addressed - but a
                # layer has 689 of them against the weights' 19,137, so the digest is affordable and
                # they need not be re-encoded on every run either.
                owner.plaintext_store = store
            with timed(timings, f"layer {index}"):
                state = layer.forward(state, weights[index], padding, index,
                                      softmax_parameters=parameters, trace=trace)
            if trace is not None:
                rows = per_stage_fidelity(engine, trace, traces[sample][f"layer_{index}"],
                                          int(np.asarray(mask).sum()))
                stage_rows.append((index, rows))
                # Print as soon as the layer is measured rather than at the end of the run. This is
                # the expensive half of a --per-stage run and the half that says which stage is
                # wrong; a failure anywhere after it - a later layer, the final decode, the
                # plaintext tail - must not be able to take it down with it.
                print_stage_rows(index, rows)

        with timed(timings, "decrypt"):
            hidden = decode_six_blocks(engine, state) / args.output_scale
        hidden_all.append(hidden)

        with timed(timings, "plaintext tail"):
            for index in range(args.layers, config.num_hidden_layers):
                hidden, _ = model.encoder_layer(hidden, index, np.asarray(mask, dtype=np.float64))
            pooled = np.tanh(hidden[0] @ model.state["bert.pooler.dense.weight"].T
                             + model.state["bert.pooler.dense.bias"])
            logits.append(pooled @ model.state["classifier.weight"].T
                          + model.state["classifier.bias"])

        if not args.quiet:
            print(f"  sample {sample + 1}/{len(encoded)} done "
                  f"({timings.total:.1f}s elapsed)", file=sys.stderr, flush=True)

    if store is not None and not args.quiet:
        print(f"  {store.describe()}", file=sys.stderr, flush=True)
    args._bootstrap_margins = getattr(engine, "bootstrap_margins", None)
    args._recoverable_bound = getattr(engine, "recoverable_bound", None)
    return np.array(logits), np.array(hidden_all), stage_rows


def print_bootstrap_margins(args, stream=sys.stdout):
    """How close each bootstrap came to q0/(2*Delta), worst first.

    The guard in `ClearEngine.bootstrap` fires on a message already over the bound. This is the
    number to look at before that: the device's slot distribution matches this engine's to the 99th
    percentile and runs wider in the extreme, so a bootstrap that only clears the bound by a small
    factor here is one the device wraps in whichever few slots ran wide - and a wrapped slot is not
    refreshed, it is replaced, after which `he_inv` and `he_invsqrt` amplify it without limit.
    """
    margins = getattr(args, "_bootstrap_margins", None)
    if not margins:
        return
    bound = args._recoverable_bound
    ratios = sorted((ratio, peak) for peak, ratio in margins)
    print(f"\nbootstrap headroom against q0/(2*Delta) = {bound:.4g} ({len(margins)} bootstraps)",
          file=stream)
    print(f"  {'':4}{'peak':>12}{'of bound':>11}{'headroom':>11}", file=stream)
    for ratio, peak in ratios[-5:][::-1]:
        print(f"  {'':4}{peak:>12.4g}{ratio:>10.1%}{1 / ratio if ratio else float('inf'):>10.2f}x",
              file=stream)
    worst = ratios[-1][0]
    print(f"  tightest {1 / worst:.2f}x. The device's tail is wider than this engine's, so anything "
          f"under about 3x is worth scaling down\n  (--score-refresh-scale for stage 07, "
          f"--refresh-scale for LayerNormStages.refresh, --residual-scale for stage 15).",
          file=stream)


# ---------------------------------------------------------------------- commands
def command_info(args):
    hub = build_hub(args)
    print(f"thorfhe benchmark\n  {hub.describe()}")
    if hub.cache.exists():
        files = sorted(p for p in hub.cache.rglob("*") if p.is_file())
        total = sum(p.stat().st_size for p in files)
        print(f"  cached {len(files)} files, {total / 2**20:.1f} MiB")
        for path in files[:20]:
            print(f"    {path.relative_to(hub.cache)}  {path.stat().st_size / 2**20:.1f} MiB")
    else:
        print("  cache is empty")
    return 0


def command_fetch(args):
    hub = build_hub(args)
    print(f"fetching into {hub.cache}")
    model, tokenizer, config, weights_path = load_model(hub, args)
    rows = load_rows(hub, args)
    print(f"  model {args.model}: {config.num_hidden_layers} layers, "
          f"{config.num_labels} labels, act {config.hidden_act}")
    print(f"  weights {weights_path.name} ({weights_path.stat().st_size / 2**20:.1f} MiB), "
          f"{len(model.state)} tensors")
    print(f"  vocabulary {len(tokenizer.vocab)} tokens, lower_case={tokenizer.do_lower_case}")
    print(f"  dataset {args.dataset}/{args.config}/{args.split}: {len(rows)} rows")
    return 0


def command_reference(args):
    hub = build_hub(args)
    model, tokenizer, config, _ = load_model(hub, args)
    rows = load_rows(hub, args)
    encoded = encode_rows(tokenizer, rows, args.max_length)

    timings = Timings(samples=len(encoded))
    logits, _ = run_reference(model, encoded, timings)
    labels = np.array([entry[3] for entry in encoded])
    score = Classification.score(logits.argmax(-1), labels)

    print(f"\nplaintext reference: {args.model} on {args.config}/{args.split}")
    print(score.format("labels"))
    print(timings.format())
    _write_json(args, dict(command="reference", model=args.model, samples=len(encoded),
                           **as_dict(score, timings)))
    return 0


def command_fhe(args):
    hub = build_hub(args)
    model, tokenizer, config, _ = load_model(hub, args)
    rows = load_rows(hub, args)
    encoded = encode_rows(tokenizer, rows, args.max_length)
    args.layers = min(args.layers, config.num_hidden_layers)

    timings = Timings(samples=len(encoded))
    reference_logits, traces = run_reference(model, encoded, timings, trace=True)

    print(f"\nencrypted run: {args.layers} of {config.num_hidden_layers} layers on "
          f"{args.engine}, {len(encoded)} samples")
    fhe_logits, fhe_hidden, stage_rows = run_encrypted(model, encoded, args, timings, traces)

    labels = np.array([entry[3] for entry in encoded])
    reference_score = Classification.score(reference_logits.argmax(-1), labels)
    fhe_score = Classification.score(fhe_logits.argmax(-1), labels)
    logit_fidelity = Fidelity.between(fhe_logits, reference_logits)
    agreement = ProbabilityAgreement.between(fhe_logits, reference_logits)

    reference_hidden = np.array([traces[i][f"layer_{args.layers - 1}"]["norm_2"]
                                 for i in range(len(encoded))])

    # Over the tokens that carry a word, which is the only place the two models compute the same
    # thing. `padding_mask` masks the query side, so a padding row's attention comes out empty and
    # its hidden state is whatever an empty context normalises to; the plaintext model has no such
    # notion and carries [PAD] through like any other token. Comparing all 128 rows on MRPC's first
    # sample reports relRMSE 0.22 for a layer whose every *stage* is accurate to 5e-3 or better - the
    # padding is two thirds of the rows. `per_stage_fidelity` has always taken the token count; this
    # is the same restriction, and it is what makes the number comparable to thor-openfhe's, which
    # skips padding explicitly (44 x 768 = 33792 values, not 128 x 768). Both are reported, because
    # a large gap between them is itself worth seeing.
    counts = [int(np.asarray(entry[2]).sum()) for entry in encoded]
    carried = np.concatenate([h[:n] for h, n in zip(fhe_hidden, counts)])
    reference_carried = np.concatenate([r[:n] for r, n in zip(reference_hidden, counts)])

    hidden_fidelity = Fidelity.between(carried, reference_carried)
    padded_fidelity = Fidelity.between(fhe_hidden, reference_hidden)
    # The best-fit scale between the two. A THOR LayerNorm returns a constant multiple of the real
    # one, and getting that constant wrong looks exactly like a broken layer, so report it rather
    # than let --output-scale hide it.
    fitted = float((carried * reference_carried).sum() / (reference_carried ** 2).sum())
    scaled_fidelity = Fidelity.between(carried / fitted, reference_carried)

    print(f"\naccuracy (against the dataset labels)")
    print("  " + reference_score.format("plaintext"))
    print("  " + fhe_score.format("encrypted"))
    print(f"\nfidelity (against the plaintext model)")
    print("  " + hidden_fidelity.format(f"hidden after layer {args.layers - 1}"))
    print("  " + padded_fidelity.format("  ... with padding"))
    print(f"  {'best-fit scale':<24}{fitted * args.output_scale:.4f} of the plaintext hidden state "
          f"(--output-scale is {args.output_scale})")
    print("  " + scaled_fidelity.format("  ... rescaled by it"))
    print("  " + logit_fidelity.format("logits"))
    print("  " + agreement.format("probabilities"))

    for index, rows in stage_rows:
        print_stage_rows(index, rows, stream=sys.stdout)
    print(f"\ntiming")
    print(timings.format())
    print_bootstrap_margins(args)
    ops = getattr(args, "_op_timings", None)
    if ops is not None:
        print(f"\nengine primitives ({args.layers} encrypted layers, {len(encoded)} samples)")
        print(ops.format())

    _write_json(args, dict(command="fhe", model=args.model, engine=args.engine,
                           layers=args.layers, samples=len(encoded),
                           reference_accuracy=reference_score.accuracy,
                           **as_dict(fhe_score, logit_fidelity, agreement, timings),
                           hidden_fidelity=hidden_fidelity.__dict__,
                           hidden_fidelity_rescaled=scaled_fidelity.__dict__,
                           hidden_best_fit_scale=fitted * args.output_scale))
    return 0


def _write_json(args, payload):
    if args.json:
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")


# ---------------------------------------------------------------------- argv
def resolve_bootstrap_depth(args) -> int:
    """Levels EvalBootstrap consumes, measured where we have measured it.

    Not a constant, and not guessable: it moves with the level budget, and for this port it also
    includes the level EvalCoeffsToSlots spends aligning the ciphertext to its diagonals. An explicit
    --bootstrap-depth wins; otherwise take the measured value, and say so if there is none.
    """
    if getattr(args, "bootstrap_depth", None) is not None:
        return args.bootstrap_depth
    from .budget import bootstrap_depth
    budget = None if getattr(args, "no_bootstrap", False) else tuple(args.bootstrap_level_budget)
    measured = bootstrap_depth(log_n=getattr(args, "log_n", 16), level_budget=budget or (3, 3))
    if measured is None:
        raise SystemExit(
            f"level budget {budget} has no measured bootstrap depth, so the post-bootstrap level "
            f"cannot be derived. Pass --bootstrap-depth explicitly, or add the measurement to "
            f"thorfhe.budget.MEASURED_BOOTSTRAP.")
    return measured


def level_budget(text: str) -> tuple[int, int]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("level budget is two integers, e.g. 3,3")
    return (int(parts[0]), int(parts[1]))


#: The input `magnitudes` measures on, fixed so two machines report comparable numbers. The
#: sentence pair is MRPC-shaped; what matters is that it is the same one on both sides.
MAGNITUDE_SENTENCES = ("The company said it will cut 500 jobs .",
                       "About 500 positions will be eliminated , the company said .")


def command_magnitudes(args):
    """What each bootstrap in a layer is handed, by site, on the real checkpoint.

    A CKKS bootstrap only recovers a message well inside `q0 / Delta`; past it the sine has wrapped
    and what comes back is an unrelated number. Which sites crowd that bound is a property of the
    real activations - random weights understate it sevenfold - so it has to be measured, and it has
    to be measured the same way on both sides. Two independent measurements of stage 07 disagreed by
    a factor of four because they were not: this exists so the next disagreement is about numbers
    rather than about what was run.

    `--through 06` stops after the attention score, which is the cheap part and needs no memory to
    speak of; the default runs the whole layer.
    """
    import collections
    import os
    import traceback

    import numpy as np

    from .bert import layer_parameters
    from .clear import ClearEngine
    from .encoding import encode_activations
    from .layer import DEFAULT_SOFTMAX_SCALE, SOFTMAX_SCALES, EncoderLayer, encode_layer

    g = THOR_BERT
    model, tokenizer, _, _ = load_model(build_hub(args), args)

    ids, types, mask = tokenizer.encode_pair(*MAGNITUDE_SENTENCES, max_length=g.dim)
    ids, types, mask = np.asarray(ids), np.asarray(types), np.asarray(mask)
    tokens = int(mask.sum())
    state = model.state
    embedded = (state["bert.embeddings.word_embeddings.weight"][ids]
                + state["bert.embeddings.position_embeddings.weight"][: g.dim]
                + state["bert.embeddings.token_type_embeddings.weight"][types])
    centred = embedded - embedded.mean(-1, keepdims=True)
    x = (centred / np.sqrt(embedded.var(-1, keepdims=True) + 1e-12)
         * state["bert.embeddings.LayerNorm.weight"] + state["bert.embeddings.LayerNorm.bias"])

    print(f"checkpoint  {args.model}", file=sys.stderr)
    print(f"input       {tokens}/{g.dim} tokens, LayerNorm'd embedding |x| max {np.abs(x).max():.4g}"
          f"  p50 {np.median(np.abs(x)):.4g}", file=sys.stderr)
    print(f"scales      softmax {SOFTMAX_SCALES.get(args.layer, DEFAULT_SOFTMAX_SCALE)}  "
          f"residual {args.residual_scale}  refresh {args.refresh_scale}  "
          f"score_refresh {args.score_refresh_scale}", file=sys.stderr, flush=True)

    seen = []

    class Probe(ClearEngine):
        def bootstrap(self, ct, keep_levels=None):
            frames = traceback.extract_stack()[-4:-1]
            site = "|".join(f"{f.filename.rsplit(os.sep, 1)[-1]}:{f.lineno}" for f in frames)
            seen.append((site, float(np.max(np.abs(ct.slots))), ct.level))
            return super().bootstrap(ct, keep_levels)

    level = args.depth - resolve_bootstrap_depth(args)
    engine = Probe(g, depth=args.depth, bootstrap_level=level)
    engine.bootstrap_message_margin = 1e9    # measure the magnitude rather than refuse it
    layer = EncoderLayer(engine, residual_scale=args.residual_scale,
                         refresh_scale=args.refresh_scale,
                         score_refresh_scale=args.score_refresh_scale,
                         binary_rotations=True, refresh_after_dense=args.refresh_after_dense)
    for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
        owner.check_ranges = False

    # lazy: one encoded layer is 9.7 GiB and this only needs one field at a time
    weights = encode_layer(layer_parameters(state, args.layer), args.layer,
                           residual_scale=args.residual_scale,
                           score_refresh_scale=args.score_refresh_scale, lazy=True)
    # The layer is entered at `ACTIVATION_SCALE`, and that amplitude squares into the attention
    # score. Measuring at 1 instead is what made stage 06 look like it carried `(q.k) * scale`
    # when it carries four times that - see `SOFTMAX_SCALES`.
    packed = np.array([engine.encrypt(m) for m in encode_activations(g, args.output_scale * x)],
                      dtype=object)

    if args.through == "06":
        _, complexified = layer.attention.stage_01_complexify_x(packed, layer_index=args.layer)
        rotated = layer.attention.stage_02_make_rotated_copies(complexified)
        scores = layer.attention.stage_06_attention_score(
            layer.attention.stage_03_query(rotated, *weights.query),
            layer.attention.stage_04_key(rotated, *weights.key))
        magnitudes = [float(np.max(np.abs(np.asarray(engine.decrypt(ct))))) for ct in scores]
        half = len(scores) // 2
        packed_pairs = [float(np.max(np.abs(np.asarray(engine.decrypt(scores[i]))
                                            + 1j * np.asarray(engine.decrypt(scores[i + half])))))
                        for i in range(half)]
        print(f"\nstage 06 outputs      {'  '.join(f'{m:.4g}' for m in magnitudes)}")
        print(f"stage 07 would refresh {'  '.join(f'{m:.4g}' for m in packed_pairs)}")

        # The magnitude alone does not travel between machines - it depends on the input. The ratio to
        # the plaintext score does, and it is the quantity actually in dispute: whether stage 06
        # carries the score or some multiple of it. Checked at the slot where the plaintext score is
        # largest, since that is the one a bound has to accommodate.
        scale = SOFTMAX_SCALES.get(args.layer, DEFAULT_SOFTMAX_SCALE)
        weight = layer_parameters(state, args.layer)
        # `s * (x @ W.T) + 2b`, not `x @ W.T + b`: `encode_weight` halves the weights to pay for the
        # `y + conj(y)` that makes the result real, and the bias is added between the two, so it is
        # not scaled by `s` - see `test_qkv_computes_xw_plus_bias`. Both terms have to be right: a
        # reference with the single bias reads 0.90 where the stage is exact, and one that drops `s`
        # reads 1.0000 where the stage carries four times the score.
        entry = args.output_scale
        query = entry * (x @ weight["query.weight"].T) + 2 * weight["query.bias"]
        key = (entry * (x @ weight["key.weight"].T) + 2 * weight["key.bias"]) * scale
        heads = np.stack([query[:, h * g.n_out:(h + 1) * g.n_out]
                          @ key[:, h * g.n_out:(h + 1) * g.n_out].T for h in range(g.n_blocks)])
        head, token, other = np.unravel_index(np.abs(heads).argmax(), heads.shape)
        diagonal = (int(other) - int(token)) % g.dim
        held = np.asarray(engine.decrypt(scores[diagonal // g.pack]))[
            g.slot(diagonal % g.pack, int(token), int(head))]
        expected = float(heads[head, token, other])
        print(f"\nplaintext score (with softmax_scale) max {abs(expected):.5g} "
              f"at head {head}, ({token}, {other}) - diagonal {diagonal}")
        print(f"the ciphertext holds {abs(held):.5g} there, a ratio of "
              f"{abs(held) / abs(expected):.4f}")
        return 0

    layer.forward(packed, weights, layer.padding_mask(tokens), args.layer)

    grouped = collections.defaultdict(list)
    for site, magnitude, in_level in seen:
        grouped[site].append((magnitude, in_level))
    print(f"\n{len(seen)} bootstraps, by site:")
    print(f"  {'site':<52} {'n':>3} {'min':>10} {'max':>10} {'% of bound 2':>13}")
    for site, values in sorted(grouped.items(), key=lambda kv: -max(v[0] for v in kv[1])):
        low = min(v[0] for v in values)
        high = max(v[0] for v in values)
        print(f"  {site:<52} {len(values):3d} {low:10.4g} {high:10.4g} {100 * high / 2:12.1f}%")
    worst = max(m for _, m, _ in seen)
    print(f"\nlargest {worst:.4g}; a bound of 2 leaves it at {100 * worst / 2:.1f}%, "
          f"a bound of 32 at {100 * worst / 32:.1f}%")
    return 0


def command_workingset(args):
    """Measure how many ciphertexts each stage keeps alive, and what that costs on the device.

    The key and plaintext footprint is what `budget` predicts; this is the other half - what the
    circuit itself holds while it runs. Both have to fit, and it was the second that the GPU runs kept
    dying of, one stage at a time, without anyone knowing which stage was actually the largest.
    """
    import collections

    import numpy as np

    from .encoding import encode_activations
    from .layer import EncoderLayer, encode_layer
    from .plaintext_store import store_for
    from .workingset import tracking_engine

    g = THOR_BERT
    square = np.zeros((g.features, g.features))
    vector = np.zeros(g.features)
    dummy = {"query.weight": square, "query.bias": vector, "key.weight": square,
             "key.bias": vector, "value.weight": square, "value.bias": vector,
             "attention.output.dense.weight": square, "attention.output.dense.bias": vector,
             "attention.output.LayerNorm.weight": vector, "attention.output.LayerNorm.bias": vector,
             "intermediate.dense.weight": np.zeros((4 * g.features, g.features)),
             "intermediate.dense.bias": np.zeros(4 * g.features),
             "output.dense.weight": np.zeros((g.features, 4 * g.features)),
             "output.dense.bias": vector,
             "output.LayerNorm.weight": vector, "output.LayerNorm.bias": vector}

    level = args.depth - resolve_bootstrap_depth(args)
    engine, tracker = tracking_engine(g, depth=args.depth, bootstrap_level=level)
    # `compact` belongs here as much as anywhere: `stream_qkv` exists precisely to stop the layer
    # holding the 64 rotated copies, and this is the tool that says what holding them costs. Without
    # the flag this measured the path nobody runs any more.
    layer = EncoderLayer(engine, binary_rotations=rotation_mode(args),
                         refresh_after_dense=args.refresh_after_dense,
                         compact=args.compact)
    # Dummy weights are zeros, so every value in the layer is an artefact of that: the variance a
    # LayerNorm sees is 3e-12 and no window covers it. What is being measured here is how many
    # ciphertexts each stage holds, which the values do not affect - so turn the range checks off,
    # exactly as `plan_rotations` does for the same reason.
    for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
        owner.check_ranges = False

    for owner, names in ((layer.attention, ("stage_01_complexify_x", "stage_02_make_rotated_copies",
                                            "stage_03_query", "stage_04_key", "stage_05_value",
                                            "stage_06_attention_score", "stage_07_softmax",
                                            "stage_08_attention_context")),
                         (layer.dense, ("stage_10_attention_dense",)),
                         (layer.norm, ("stage_11_attention_layernorm", "refresh",
                                       "stage_15_prepare_layernorm", "stage_16_output_layernorm")),
                         (layer.feedforward, ("stage_12_intermediate_dense", "stage_13_gelu",
                                              "stage_14_output_dense"))):
        for name in names:
            bound = getattr(owner, name, None)
            if bound is None:
                continue

            def labelled(bound=bound, name=name):
                def call(*a, **kw):
                    tracker.at(name)
                    return bound(*a, **kw)
                return call
            setattr(owner, name, labelled())

    state = np.array([engine.encrypt(m)
                      for m in encode_activations(g, np.zeros((g.dim, g.features)))], dtype=object)
    layer.forward(state, encode_layer(dummy, 0), layer.padding_mask(g.dim), 0)

    print(f"ciphertext working set, one layer, depth={args.depth}, bootstrap level {level}, "
          f"N=2^{args.log_n}")
    print(tracker.report(g.slot_count, args.depth))
    # By bytes, not by count: the two disagree, and it is the bytes that run a card out of memory.
    # Stage 07 holds the most ciphertexts and stage 06 the most memory, because stage 06 works near
    # full level while stage 07's ciphertexts have already spent theirs.
    peak = tracker.peak
    print(f"\n  peak {peak.nbytes / (1 << 30):.2f} GiB in {peak.where} "
          f"({peak.count} live)")
    print("  (light-plaintext expansions and the engine's own scratch are on top of this)")
    # Where in the level range that stage's bytes sit. A ciphertext at level 36 costs six times one at
    # level 5, so a handful of near-full ones outweighs a crowd of spent ones - and a ciphertext held
    # at a level higher than its next use needs is pure waste that a level_down would return.
    ranked = sorted(tracker.per_label.values(), key=lambda p: -p.nbytes)
    for row in [r for r in ranked if r.nbytes >= (1 << 30)]:
        histogram = collections.Counter(row.levels)
        held = " ".join(f"L{lvl}x{n}" for lvl, n in sorted(histogram.items(), reverse=True))
        print(f"  {row.where}: {held}")
    return 0


def command_budget(args):
    """Predict the GPU footprint of a parameter set without building a context."""
    from .budget import estimate
    from .he import plan_rotations

    plan = None
    rotations = None
    key_cost, key_budget = rotation_key_cost(args)
    if not args.keys:
        level = (args.depth - resolve_bootstrap_depth(args) if args.bootstrap_level is None
                 else args.bootstrap_level)
        rotations = plan_rotations(THOR_BERT, depth=args.depth, bootstrap_level=level,
                                   binary_rotations=args.binary_rotations,
                                   extra_rotation_keys=args.extra_rotation_keys,
                                   refresh_after_dense=args.refresh_after_dense,
                                   key_cost=key_cost, rotation_key_budget=key_budget,
                                   rotation_max_steps=args.rotation_max_steps)
        plan = rotations.levels
    predicted = estimate(log_n=args.log_n, depth=args.depth, dnum=args.dnum,
                         rotation_levels=plan, rotation_keys=args.keys,
                         level_budget=None if args.no_bootstrap else tuple(args.bootstrap_level_budget),
                         special_primes=args.special_primes, truncate=not args.no_truncate)
    print(f"log_n={args.log_n} depth={args.depth} dnum={args.dnum} "
          f"level_budget={args.bootstrap_level_budget} "
          f"{describe_rotations(args, rotations)} rotations")
    print(predicted.format(args.card_gib * (1 << 30)))
    return 0


def build_parser():
    parser = argparse.ArgumentParser(prog="python -m thorfhe.bench",
                                     description="THOR on fideslib: speed, accuracy and fidelity.")
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p):
        network = p.add_argument_group("network and cache")
        network.add_argument("--proxy", default=None,
                             help="proxy URL for both http and https; falls back to "
                                  "HTTPS_PROXY/HTTP_PROXY/ALL_PROXY")
        network.add_argument("--hf-endpoint", default=None,
                             help="Hugging Face endpoint or mirror (env HF_ENDPOINT)")
        network.add_argument("--rows-endpoint", default=None, help="datasets-server endpoint")
        network.add_argument("--token", default=None, help="Hugging Face token (env HF_TOKEN)")
        network.add_argument("--cache-dir", default=None,
                             help="download cache (env THORFHE_CACHE, default ~/.cache/thorfhe)")
        network.add_argument("--offline", action="store_true",
                             help="use the cache only; fail rather than reach the network")
        network.add_argument("--timeout", type=float, default=30.0)
        network.add_argument("--retries", type=int, default=3)

        data = p.add_argument_group("model and data")
        data.add_argument("--model", default=DEFAULT_MODEL)
        data.add_argument("--revision", default="main")
        data.add_argument("--dataset", default=DEFAULT_DATASET)
        data.add_argument("--config", default=DEFAULT_CONFIG)
        data.add_argument("--split", default=DEFAULT_SPLIT)
        data.add_argument("--limit", type=int, default=None, help="use only the first N rows")
        data.add_argument("--max-length", type=int, default=SEQUENCE_LENGTH,
                          help=f"sequence length; THOR's packing fixes it at {SEQUENCE_LENGTH}")

        p.add_argument("--json", default=None, help="write the metrics to this file")
        p.add_argument("--quiet", action="store_true")

    info = sub.add_parser("info", help="show the proxy, cache and what is already downloaded")
    common(info)
    info.set_defaults(handler=command_info)

    fetch = sub.add_parser("fetch", help="download and cache the checkpoint and the dataset")
    common(fetch)
    fetch.set_defaults(handler=command_fetch)

    reference = sub.add_parser("reference", help="the plaintext model: accuracy, F1, time")
    common(reference)
    reference.set_defaults(handler=command_reference)

    fhe = sub.add_parser("fhe", help="the encrypted pipeline: time, accuracy and fidelity")
    common(fhe)
    engine = fhe.add_argument_group("engine")
    engine.add_argument("--engine", choices=("clear", "fideslib"), default="clear",
                        help="'clear' is the exact numpy mirror under the same level and scale "
                             "contract; 'fideslib' is the CUDA backend")
    engine.add_argument("--layers", type=int, default=1,
                        help="how many encoder layers to run encrypted; the rest run in plaintext")
    # 37 is the level budget a layer actually needs, measured: the deepest inter-bootstrap
    # segment is 19 and a bootstrap costs 17. It only fits with --refresh-after-dense; without
    # that flag the chain runs out and the clear engine says so before any GPU time is spent.
    engine.add_argument("--depth", type=int, default=37,
                        help="multiplicative depth. The measured working point is 37 together "
                             "with --refresh-after-dense (docs/thor_port.md).")
    engine.add_argument("--bootstrap-level", type=int, default=None,
                        help="level a bootstrap restores to. Defaults to depth - --bootstrap-depth, "
                             "which is what the hardware actually gives; override only to explore")
    engine.add_argument("--bootstrap-depth", type=int, default=None,
                        help="levels EvalBootstrap itself consumes, including the one "
                             "EvalCoeffsToSlots spends aligning to its diagonals. Defaults to the "
                             "measured value for the level budget (17 for 3,3; 18 for 4,4)")
    engine.add_argument("--lenient", action="store_true",
                        help="do not enforce the FIXEDMANUAL level and scale contract")
    engine.add_argument("--calibrate", action="store_true",
                        help="derive the softmax window from the plaintext scores instead of using "
                             "THOR's per-layer table")
    engine.add_argument("--refresh-after-dense", action="store_true",
                        help="insert a bootstrap between stages 10 and 11. Not a THOR stage and "
                             "semantically the identity, but it halves the layer's deepest level "
                             "chain: minimum depth 52 -> 34, which is what fits a 32 GiB card")
    engine.add_argument("--binary-rotations", action="store_true",
                        help="perform every rotation as a sequence of power-of-two rotations: 15 "
                             "rotation keys instead of 210 (3.6 GiB instead of 51), at 4.5x the "
                             "rotation count. The only way a full layer's keys fit a 32 GB card")
    engine.add_argument("--extra-rotation-keys", type=int, default=0, metavar="N",
                        help="keep N rotation keys beyond the powers of two, chosen by measuring "
                             "which indices the layer rotates by most. Six of them cut the "
                             "rotation count from 8258 to 3465 for 1.0 GiB more key memory; past nine "
                             "the keys stop fitting a 32 GB card. Implies --binary-rotations. A cap, "
                             "not a target: with --rotation-key-budget the greedy stops early rather "
                             "than overrun")
    engine.add_argument("--boundary-refresh-scale", type=float, default=4.0, metavar="S",
                        help="divide the hidden state by S before the bootstrap that starts every "
                             "layer after the first; 0 refreshes nothing there. A layer costs 36 "
                             "levels (in at 37, out at 1), so without this the second layer dies in "
                             "pcmm at level -1 - which is why every run so far has been --layers 1. "
                             "S is needed because the state leaves a layer at about 19 and grows, "
                             "against a recoverable bound of 16")
    engine.add_argument("--inverse-lift", type=int, default=1, metavar="N",
                        help="multiply the softmax denominator by N before he_inv refreshes it, and "
                             "divide it back out afterwards. The bootstrap's error is absolute - "
                             "0.017 on the device against layer 0's epsilon of 2^-6 - so a "
                             "denominator low in [epsilon, 1] is mostly error. Level-free (the "
                             "scalar is an integer); refused if N would push the denominator past 1. "
                             "Measured on the clear engine with 0.017 injected: 35.6 of relative "
                             "error at 1, 0.63 at 3")
    engine.add_argument("--plaintext-cache", default=None, metavar="DIR",
                        help="keep encoded weights under DIR and reuse them. Encoding is 96%% of a "
                             "layer on the device (19,137 calls at 58 ms) and the weights do not "
                             "depend on the sample, so this is paid once: about 9.3 GiB a layer, "
                             "112 GiB for all twelve. Keyed by checkpoint, layer, field and the "
                             "scales that are baked in, so it cannot be read back under parameters "
                             "it was not written under")
    engine.add_argument("--rotation-key-budget", type=float, default=None, metavar="GIB",
                        help="how much key memory the extra rotation keys may bring the set to. "
                             "Given one, they are chosen by rotations removed per byte instead of "
                             "per key - keys are level-truncated, so an index rotated below the "
                             "bootstrap level is a fraction of the price of one rotated at depth")
    engine.add_argument("--rotation-max-steps", type=int, default=4, metavar="N",
                        help="how many keys one rotation index may be reached through. Two is a pair "
                             "of keys and the binary expansion for everything else; four meets in "
                             "the middle and costs 3465 rotations a layer instead of 4237 for the "
                             "same keys. Planning cost only - the decomposition is cached")
    engine.add_argument("--special-primes", type=int, default=11, metavar="K",
                        help="how many special primes the context has; only used to price keys. "
                             "Recover it from a run's key-memory line, it depends on the digits")
    engine.add_argument("--time-ops", action="store_true",
                        help="count and time every engine primitive, and print the table at the end. "
                             "A phase total says the layer took 1131s; this says which of its 53,385 "
                             "calls did. Costs about 0.2 us a call, against 19 ms measured on the "
                             "device, so it does not move what it measures")
    engine.add_argument("--device-memory", action="store_true",
                        help="print the device pool at every stage boundary. An OOM says which stage "
                             "was unlucky, not which one was large; this says where the memory is.")
    engine.add_argument("--per-stage", action="store_true",
                        help="decrypt every stage of the first sample and report its fidelity and "
                             "best-fit scale against the plaintext model - the diagnostic that says "
                             "which stage a divergence comes from")
    # The three level-free scalings that keep a bootstrap's input inside `q0/Delta`. They are folded
    # into plaintexts, so they cost nothing at all, and they are what the 22-site magnitude
    # measurement settled at: residual 256, refresh 4, score_refresh 16. Defaults are 1.0 because
    # they are only needed once the magnitudes are measured against the bound the run actually has.
    engine.add_argument("--residual-scale", type=float, default=1.0,
                        help="divide what stage 15 hands its bootstrap; 256 on the real checkpoint, "
                             "where the residual reaches 132")
    engine.add_argument("--refresh-scale", type=float, default=1.0,
                        help="the same for `LayerNormStages.refresh`; 4 on the real checkpoint")
    engine.add_argument("--score-refresh-scale", type=float, default=1.0,
                        help="the same for stage 07's score bootstrap; 16 on the real checkpoint")
    engine.add_argument("--compact", action="store_true",
                        help="every option that trades recomputation for held memory, together: "
                             "streamed QKV copies and lazily encoded weights. Exact - no output "
                             "value changes - and worth 1.07 GiB on the device, which is the margin "
                             "--extra-rotation-keys needs. THORFHE_DEBUG=1 reports what it did")
    engine.add_argument("--lazy-weights", action="store_true",
                        help="encode each of a layer's weight fields as it is read rather than "
                             "holding the layer. One encoded layer is 9.7 GB at THOR's geometry and "
                             "`forward` reads each field once, so this is a 3x smaller peak for the "
                             "cost of re-encoding per layer")
    engine.add_argument("--noise-model", action="store_true",
                        help="give the clear engine the device's bootstrap error instead of exact "
                             "arithmetic. The error a bootstrap leaves is set by q0/Delta and not by "
                             "the value it is handed, so it is the same absolute size whatever it "
                             "refreshes - which is why it destroys the softmax denominator and "
                             "nothing else. Ignored by --engine fideslib, which has the real thing")
    engine.add_argument("--bootstrap-noise-only", action="store_true",
                        help="with --noise-model, model the bootstrap's error and skip the rest. A "
                             "key-switch is 2^-40 of the scale and a layer's rotations accumulate "
                             "8.2e-11 against the bootstrap's 4.9e-04, so this drops one part in 5.9 "
                             "million and the allocation that makes a noise-modelled layer not fit")
    engine.add_argument("--bootstrap-precision-bits", type=int, default=11,
                        help="bits of q0/Delta the bootstrap reproduces, for --noise-model. The "
                             "default is the device's measured floor (0.015 at Delta=2^50, i.e. "
                             "0.015/32 = 2^-11), not the 22 the clear engine assumes on its own")
    engine.add_argument("--output-scale", type=float, default=ACTIVATION_SCALE,
                        help="the amplitude every activation ciphertext carries relative to the real "
                             "value; THOR's final doubling is never cancelled, so this is 2. It is "
                             "not only a decode scale: the layer is entered on it, so it squares "
                             "into the attention score and `SOFTMAX_SCALES` is derived from it")
    device = fhe.add_argument_group("fideslib engine")
    device.add_argument("--device", default="cuda:0")
    device.add_argument("--log-n", type=int, default=16)
    device.add_argument("--scaling-bits", type=int, default=50)
    device.add_argument("--first-mod-bits", type=int, default=55)
    device.add_argument("--dnum", type=int, default=4,
                        help="raise only as far as the MAXP check needs: a key is 2*dnum polynomials, "
                             "so dnum costs key memory linearly while only shrinking K")
    device.add_argument("--bootstrap-level-budget", type=level_budget, default=(3, 3),
                        metavar="E,D", help="EvalBootstrapSetup level budget, e.g. 4,4")
    device.add_argument("--no-bootstrap", action="store_true",
                        help="build the context without bootstrap keys - a layer cannot finish "
                             "without them, but it isolates the rest")
    device.add_argument("--secret-key-dist", choices=("sparse", "uniform"), default="sparse")
    device.add_argument("--light-plaintext-cache", type=int, default=8,
                        help="expanded light plaintexts kept resident; each is about "
                             "(level+1) * N * 8 bytes, so 64 is nearly a GiB. Measured against one "
                             "layer's 21195 plaintext multiplies: 8 entries hit 93.0%% for 124 MiB, "
                             "32 hit 95.4%% for 486 MiB, and holding all 959 hits 95.5%% for 7.4 GiB. "
                             "Raising it buys almost nothing - the accesses come in bursts, so a "
                             "small cache already catches them")
    device.add_argument("--no-truncate-keys", action="store_true",
                        help="store every key complete. Costs memory; use it to rule the level plan "
                             "in or out when a truncated key is suspected of being too small")
    device.add_argument("--allow-key-grow", action="store_true",
                        help="reload a truncated key that is used above its planned level instead of "
                             "raising. Run once with this to validate a level plan: the key memory "
                             "report then counts how many keys had to grow, and 0 means the plan is "
                             "right. Leave it off otherwise - silent regrowth is a per-call cost")
    device.add_argument("--card-gib", type=float, default=32.0,
                        help="card size the predicted footprint is checked against")
    fhe.set_defaults(handler=command_fhe)

    budget = sub.add_parser("budget", help="predict the GPU footprint without building a context")
    budget.add_argument("--log-n", type=int, default=16)
    budget.add_argument("--depth", type=int, default=50)
    budget.add_argument("--dnum", type=int, default=4)
    budget.add_argument("--bootstrap-level", type=int, default=None)
    budget.add_argument("--bootstrap-depth", type=int, default=None)
    budget.add_argument("--bootstrap-level-budget", type=level_budget, default=(3, 3), metavar="E,D")
    budget.add_argument("--no-bootstrap", action="store_true")
    budget.add_argument("--binary-rotations", action="store_true")
    budget.add_argument("--extra-rotation-keys", type=int, default=0, metavar="N")
    budget.add_argument("--rotation-key-budget", type=float, default=None, metavar="GIB")
    budget.add_argument("--rotation-max-steps", type=int, default=4, metavar="N")
    budget.add_argument("--refresh-after-dense", action="store_true")
    budget.add_argument("--keys", type=int, default=None,
                        help="skip the (slow) rotation plan and assume this many untruncated keys")
    budget.add_argument("--special-primes", type=int, default=11,
                        help="K; recover it from a run's key-memory line, it depends on the digits")
    budget.add_argument("--no-truncate", action="store_true")
    budget.add_argument("--card-gib", type=float, default=32.0)
    budget.add_argument("--json", default=None)
    budget.add_argument("--quiet", action="store_true")
    budget.set_defaults(handler=command_budget)

    magnitudes = sub.add_parser(
        "magnitudes",
        help="what each bootstrap in a layer is handed, by site, on the real checkpoint")
    common(magnitudes)
    magnitudes.add_argument("--layer", type=int, default=0)
    magnitudes.add_argument("--depth", type=int, default=37)
    magnitudes.add_argument("--log-n", type=int, default=16)
    magnitudes.add_argument("--bootstrap-level-budget", type=level_budget, default=(3, 3),
                            metavar="E,D")
    magnitudes.add_argument("--bootstrap-depth", type=int, default=None)
    magnitudes.add_argument("--no-bootstrap", action="store_true")
    magnitudes.add_argument("--refresh-after-dense", action="store_true", default=True)
    magnitudes.add_argument("--residual-scale", type=float, default=1.0)
    magnitudes.add_argument("--refresh-scale", type=float, default=1.0)
    magnitudes.add_argument("--score-refresh-scale", type=float, default=1.0)
    magnitudes.add_argument("--output-scale", type=float, default=ACTIVATION_SCALE,
                            help="the amplitude the layer is entered at; it squares into the "
                                 "attention score, so measuring at 1 measures a different pipeline")
    magnitudes.add_argument("--through", choices=("06", "layer"), default="layer",
                            help="stop after the attention score, which needs far less memory")
    magnitudes.set_defaults(handler=command_magnitudes)

    working = sub.add_parser("workingset",
                             help="measure the ciphertext working set of one layer, per stage")
    working.add_argument("--depth", type=int, default=37)
    working.add_argument("--log-n", type=int, default=16)
    working.add_argument("--bootstrap-level-budget", type=level_budget, default=(3, 3), metavar="E,D")
    working.add_argument("--bootstrap-depth", type=int, default=None)
    working.add_argument("--bootstrap-level", type=int, default=None)
    working.add_argument("--no-bootstrap", action="store_true")
    working.add_argument("--binary-rotations", action="store_true", default=True)
    working.add_argument("--extra-rotation-keys", type=int, default=0, metavar="N")
    working.add_argument("--compact", action="store_true",
                         help="measure the path `--compact` actually runs: streamed QKV copies, so "
                              "the 64 rotated copies are never all alive at once")
    working.add_argument("--refresh-after-dense", action="store_true", default=True)
    working.add_argument("--quiet", action="store_true")
    working.set_defaults(handler=command_workingset)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "max_length", SEQUENCE_LENGTH) != SEQUENCE_LENGTH:
        print(f"note: THOR's packing fixes the sequence length at {SEQUENCE_LENGTH}; "
              f"--max-length {args.max_length} only affects the plaintext path", file=sys.stderr)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
