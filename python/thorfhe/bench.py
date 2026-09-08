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
from .geometry import (FEEDFORWARD_WINDOW, THOR_ATTENTION_DENSE, THOR_BERT,
                       THOR_FEEDFORWARD)
from .hub import HubClient
from .metrics import Classification, Fidelity, ProbabilityAgreement, Timings, as_dict
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
def make_engine(args, geometry):
    """The clear (numpy) engine, or fideslib when it is built and asked for."""
    if args.engine == "clear":
        from .clear import ClearEngine
        return ClearEngine(geometry, depth=args.depth, bootstrap_level=args.bootstrap_level,
                           strict=not args.lenient)

    import pyfideslib

    from .he import plan_rotation_keys

    # The rotation plan is computed, not hand-written: `plan_rotation_keys` runs the same layer on the
    # clear engine and records the level every rotation actually happens at, so the key set cannot
    # drift from the code that uses it. With --binary-rotations that is 15 indices instead of 210.
    plan = plan_rotation_keys(geometry, depth=args.depth, bootstrap_level=args.bootstrap_level,
                              binary_rotations=args.binary_rotations)
    budget = None if args.no_bootstrap else tuple(args.bootstrap_level_budget)
    dist = (pyfideslib.SPARSE_TERNARY if args.secret_key_dist == "sparse"
            else pyfideslib.UNIFORM_TERNARY)

    if not args.quiet:
        from .budget import estimate
        predicted = estimate(log_n=args.log_n, depth=args.depth, dnum=args.dnum,
                             rotation_levels=plan, level_budget=budget)
        print(f"\npredicted GPU footprint ({len(plan)} rotation keys):", file=sys.stderr)
        print(predicted.format(args.card_gib * (1 << 30)), file=sys.stderr, flush=True)

    return pyfideslib.Engine(args.device, log_n=args.log_n, depth=args.depth,
                             scaling_bits=args.scaling_bits, first_mod_bits=args.first_mod_bits,
                             dnum=args.dnum, rotation_indexes=plan,
                             bootstrap_level_budget=budget, secret_key_dist=dist,
                             light_plaintext_cache=args.light_plaintext_cache)


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
            got = reader(engine, trace[name])
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


def run_encrypted(model, encoded, args, timings: Timings, traces):
    """Encrypt, run ``args.layers`` encoder layers, decrypt, finish in plaintext."""
    from .layer import EncoderLayer, encode_layer

    config = model.config
    hidden_all, logits = [], []

    with timed(timings, "encode weights"):
        weights = [encode_layer(layer_parameters(model.state, index), index)
                   for index in range(args.layers)]

    engine = make_engine(args, THOR_BERT)
    layer = EncoderLayer(engine, binary_rotations=args.binary_rotations)
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
            with timed(timings, f"layer {index}"):
                state = layer.forward(state, weights[index], padding, index,
                                      softmax_parameters=parameters, trace=trace)
            if trace is not None:
                stage_rows.append((index, per_stage_fidelity(
                    engine, trace, traces[sample][f"layer_{index}"], int(np.asarray(mask).sum()))))

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

    return np.array(logits), np.array(hidden_all), stage_rows


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
    hidden_fidelity = Fidelity.between(fhe_hidden, reference_hidden)
    # The best-fit scale between the two. A THOR LayerNorm returns a constant multiple of the real
    # one, and getting that constant wrong looks exactly like a broken layer, so report it rather
    # than let --output-scale hide it.
    fitted = float((fhe_hidden * reference_hidden).sum() / (reference_hidden ** 2).sum())
    scaled_fidelity = Fidelity.between(fhe_hidden / fitted, reference_hidden)

    print(f"\naccuracy (against the dataset labels)")
    print("  " + reference_score.format("plaintext"))
    print("  " + fhe_score.format("encrypted"))
    print(f"\nfidelity (against the plaintext model)")
    print("  " + hidden_fidelity.format(f"hidden after layer {args.layers - 1}"))
    print(f"  {'best-fit scale':<24}{fitted * args.output_scale:.4f} of the plaintext hidden state "
          f"(--output-scale is {args.output_scale})")
    print("  " + scaled_fidelity.format("  ... rescaled by it"))
    print("  " + logit_fidelity.format("logits"))
    print("  " + agreement.format("probabilities"))

    for index, rows in stage_rows:
        print(f"\nper-stage fidelity, layer {index} (sample 0), each rescaled by its best fit")
        for name, scale, fidelity, error in rows:
            if error is not None:
                print(f"  {name:<22}could not compare: {error}")
            else:
                print(f"  {name:<22}scale {scale:8.4f}   {fidelity.format()}")
    print(f"\ntiming")
    print(timings.format())

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
def level_budget(text: str) -> tuple[int, int]:
    parts = text.replace(",", " ").split()
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("level budget is two integers, e.g. 3,3")
    return (int(parts[0]), int(parts[1]))


def command_budget(args):
    """Predict the GPU footprint of a parameter set without building a context."""
    from .budget import estimate
    from .he import plan_rotation_keys

    plan = None
    if not args.keys:
        plan = plan_rotation_keys(THOR_BERT, depth=args.depth, bootstrap_level=args.bootstrap_level,
                                  binary_rotations=args.binary_rotations)
    predicted = estimate(log_n=args.log_n, depth=args.depth, dnum=args.dnum,
                         rotation_levels=plan, rotation_keys=args.keys,
                         level_budget=None if args.no_bootstrap else tuple(args.bootstrap_level_budget),
                         special_primes=args.special_primes, truncate=not args.no_truncate)
    print(f"log_n={args.log_n} depth={args.depth} dnum={args.dnum} "
          f"level_budget={args.bootstrap_level_budget} "
          f"{'binary' if args.binary_rotations else 'one key per index'} rotations")
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
    engine.add_argument("--depth", type=int, default=90)
    engine.add_argument("--bootstrap-level", type=int, default=80)
    engine.add_argument("--lenient", action="store_true",
                        help="do not enforce the FIXEDMANUAL level and scale contract")
    engine.add_argument("--calibrate", action="store_true",
                        help="derive the softmax window from the plaintext scores instead of using "
                             "THOR's per-layer table")
    engine.add_argument("--binary-rotations", action="store_true",
                        help="perform every rotation as a sequence of power-of-two rotations: 15 "
                             "rotation keys instead of 210 (3.6 GiB instead of 51), at 4.5x the "
                             "rotation count. The only way a full layer's keys fit a 32 GB card")
    engine.add_argument("--per-stage", action="store_true",
                        help="decrypt every stage of the first sample and report its fidelity and "
                             "best-fit scale against the plaintext model - the diagnostic that says "
                             "which stage a divergence comes from")
    engine.add_argument("--output-scale", type=float, default=2.0,
                        help="what a THOR LayerNorm returns relative to the real one; THOR's final "
                             "doubling is never cancelled, so this is 2")
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
                             "(depth+1) * N * 8 bytes, so 64 is over a GiB")
    device.add_argument("--card-gib", type=float, default=32.0,
                        help="card size the predicted footprint is checked against")
    fhe.set_defaults(handler=command_fhe)

    budget = sub.add_parser("budget", help="predict the GPU footprint without building a context")
    budget.add_argument("--log-n", type=int, default=16)
    budget.add_argument("--depth", type=int, default=50)
    budget.add_argument("--dnum", type=int, default=4)
    budget.add_argument("--bootstrap-level", type=int, default=40)
    budget.add_argument("--bootstrap-level-budget", type=level_budget, default=(3, 3), metavar="E,D")
    budget.add_argument("--no-bootstrap", action="store_true")
    budget.add_argument("--binary-rotations", action="store_true")
    budget.add_argument("--keys", type=int, default=None,
                        help="skip the (slow) rotation plan and assume this many untruncated keys")
    budget.add_argument("--special-primes", type=int, default=11,
                        help="K; recover it from a run's key-memory line, it depends on the digits")
    budget.add_argument("--no-truncate", action="store_true")
    budget.add_argument("--card-gib", type=float, default=32.0)
    budget.add_argument("--json", default=None)
    budget.add_argument("--quiet", action="store_true")
    budget.set_defaults(handler=command_budget)

    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    if getattr(args, "max_length", SEQUENCE_LENGTH) != SEQUENCE_LENGTH:
        print(f"note: THOR's packing fixes the sequence length at {SEQUENCE_LENGTH}; "
              f"--max-length {args.max_length} only affects the plaintext path", file=sys.stderr)
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
