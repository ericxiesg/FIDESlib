"""Softmax step by step: the device against exact plaintext arithmetic, one probe at a time.

Everything needed for this was already in the tree except the comparison. `Stages.probed` reports
every interesting intermediate of the softmax - the refreshed scores, the exponential, both
denominators, the inverse, and each Goldschmidt half-step inside `he_inv` - and `ClearEngine` runs
exactly the same code with exact arithmetic under the same FIXEDMANUAL level and scale contract. But
the only probe ever wired up is `bench.magnitude_probe`, which reports how big each intermediate is,
not whether it is right. So a device run says "07d reached 1e6" and nothing says where 1e6 entered.

This runs the same `stage_07_softmax` twice, on the same scores, and lines the two probe streams up
by name. That splits the error into the two halves that need different fixes:

    device vs clear   the FHE error - noise, bootstrap precision, a wrong kernel
    clear vs numpy    the algorithm's own error - the degree-15 fit, Goldschmidt truncation

Reading only the end of the softmax cannot tell those apart, and reading only magnitudes cannot tell
either of them from a large but correct intermediate. The first probe where device and clear diverge
is the answer; every probe after it is downstream of that one.

`test_softmax_algorithm_against_true_softmax` needs no device and is the control: it holds the clear
column against numpy, so when the device column is bad the algorithm has already been cleared.
"""
import os

import numpy as np
import pytest

from thorfhe import THOR_BERT, ClearEngine, block_diagonal_masks
from thorfhe.attention import attention_rotate_masks, ccmm_masks, make_copies_masks, transpose_masks
from thorfhe.encoding import ACTIVATION_SCALE
from thorfhe.softmax import Softmax

# The score layout and the broadcast-diagonal layout belong to stages 06 and 07, not to this test.
# They are already written and already held by test_stage9, so they are imported rather than copied:
# a second transcription of `(diagonal + tau) % dim` is a second chance to get it wrong, and a wrong
# layout here would be indistinguishable from a device error.
from test_stage9_thor_softmax import (decode_broadcast_diagonals, encode_score_diagonals,
                                      true_softmax, used_slots)

G = THOR_BERT
DEPTH = 60

BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"),
                           reason="set PYFIDESLIB_BENCH_PARAMS=1 to build a device engine")


@pytest.fixture(scope="module")
def mask_families():
    return (block_diagonal_masks(G), transpose_masks(G), make_copies_masks(G),
            attention_rotate_masks(G), ccmm_masks(G))


@pytest.fixture
def goldschmidt_probes(monkeypatch):
    """Turn on the probes inside `he_inv`, which are the ones that matter here.

    `07c` and `07d` bracket the whole inversion, and on the device they have disagreed by seventeen
    orders of magnitude with the input already correct - so the question is which Goldschmidt step,
    and nothing between those two says. `numeric._debug` reads `THORFHE_DEBUG` per call precisely so
    a test can set it; without it the six outer probes fire and every half-step inside the iteration
    is invisible.
    """
    monkeypatch.setenv("THORFHE_DEBUG", "1")


def build_stages(engine, mask_families):
    (low, high), transpose, copies, attention, ccmm = mask_families
    return Softmax(engine, G, masks=low, complement_masks=high, transpose=transpose, copies=copies,
                   attention=attention, ccmm=ccmm, ones=engine.encrypt(used_slots()))


def synthetic_scores(seed=61):
    """The scores `test_stage9_thor_softmax.test_softmax_matches_the_real_thing` uses.

    Same generator, same seed, same window as the test that already holds this construction to
    5e-3, so the control below inherits a bar that is known to be met rather than one invented
    here. It matters that they are inside `Softmax.NARROW`: the large scores test_stage9 uses for
    its end-to-end test overflow the degree-15 fit deliberately, which is harmless on exact
    arithmetic but would put the device outside the window and make a calibration problem look
    like a device error.
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(-8, 8, size=(G.n_blocks, G.dim, G.dim))


def feed(engine, per_head):
    """Score ciphertexts for `stage_07_softmax`, carrying half of what `he_softmax` should see.

    `stage_07_softmax` packs the two real ciphertexts into one complex one, bootstraps it, and
    unpacks `2*Re` and `2*Im` - so whatever is encoded here reaches `he_softmax` doubled (THOR's
    invariant that a ciphertext carries twice the value it represents; `score_refresh_scale` is 1).
    Halving here means `he_softmax` is handed exactly the scores that
    `test_softmax_matches_the_real_thing` hands it directly, so `Softmax.NARROW` is the right window
    and its 5e-3 result is the one to expect.
    """
    return list(encode_score_diagonals(engine, per_head / 2.0))


# ------------------------------------------------------------------ the real input

#: The MRPC row this runs on. Pinned rather than swept: THOR's softmax window is calibrated to the
#: score range real activations produce (`Softmax.NARROW` is [-27.25, 21.73], measured on 64 MRPC
#: validation sentences), so a synthetic range exercises the construction somewhere it was never
#: tuned for and a device comparison then mixes a calibration problem into the FHE error. One fixed
#: row also means two machines report comparable numbers, which is why `MAGNITUDE_SENTENCES` exists.
MRPC = ("nyu-mll/glue", "mrpc", "validation")
MRPC_ROW = 0


def mrpc_example():
    """The pinned MRPC row, or a skip when the dataset is not reachable.

    Read through `HubClient` rather than checked in: the sentences are the dataset's, and a copy in
    this file would be a second source of truth that nothing keeps honest.
    """
    from thorfhe.hub import HubClient
    try:
        rows = HubClient(quiet=True).rows(*MRPC, limit=MRPC_ROW + 1)
    except Exception as exc:                                   # noqa: BLE001 - offline is expected
        pytest.skip(f"{MRPC[0]}/{MRPC[1]} is not reachable here: {exc}")
    if len(rows) <= MRPC_ROW:
        pytest.skip(f"{MRPC[1]}/{MRPC[2]} returned {len(rows)} rows")
    row = rows[MRPC_ROW]
    return row["sentence1"], row["sentence2"], row.get("label")


def real_scores(layer_index=0, model_name="textattack/bert-base-uncased-MRPC"):
    """Layer-`layer_index` attention scores for the pinned MRPC row, as stage 06 produces them.

    Runs the real path - embeddings, LayerNorm, stages 01 to 06 on exact arithmetic - rather than
    recomputing `q @ k.T` here. The scaling between the embedding and the score runs through
    `ACTIVATION_SCALE`, `output_scale` and `SOFTMAX_SCALES`, and getting it wrong by a factor would
    reproduce exactly the miscalibration this test exists to avoid. Using the pipeline means the
    scaling is right by construction.

    Returns `(per_head, score_ciphertexts, tokens, engine, stages)`.
    """
    from thorfhe.bert import layer_parameters
    from thorfhe.encoding import encode_activations
    from thorfhe.hub import HubClient
    from thorfhe.layer import EncoderLayer, encode_layer

    sentence1, sentence2, _ = mrpc_example()
    hub = HubClient(quiet=True)
    try:
        from thorfhe.bench import load_model

        class _Args:
            model = model_name
            revision = None
        model, tokenizer, _, _ = load_model(hub, _Args())
    except Exception as exc:                                   # noqa: BLE001 - offline is expected
        pytest.skip(f"checkpoint {model_name} is not available here: {exc}")

    ids, types, mask = tokenizer.encode_pair(sentence1, sentence2, max_length=G.dim)
    ids, types, mask = np.asarray(ids), np.asarray(types), np.asarray(mask)
    tokens = int(mask.sum())

    state = model.state
    embedded = (state["bert.embeddings.word_embeddings.weight"][ids]
                + state["bert.embeddings.position_embeddings.weight"][:G.dim]
                + state["bert.embeddings.token_type_embeddings.weight"][types])
    centred = embedded - embedded.mean(-1, keepdims=True)
    x = (centred / np.sqrt(embedded.var(-1, keepdims=True) + 1e-12)
         * state["bert.embeddings.LayerNorm.weight"] + state["bert.embeddings.LayerNorm.bias"])

    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    engine.bootstrap_message_margin = 1e9
    layer = EncoderLayer(engine, binary_rotations=True)
    for owner in (layer.attention, layer.dense, layer.norm, layer.feedforward):
        owner.check_ranges = False
    weights = encode_layer(layer_parameters(state, layer_index), layer_index, lazy=True)

    packed = np.array([engine.encrypt(m) for m in encode_activations(G, ACTIVATION_SCALE * x)],
                      dtype=object)
    _, complexified = layer.attention.stage_01_complexify_x(packed, layer_index=layer_index)
    rotated = layer.attention.stage_02_make_rotated_copies(complexified)
    scores = layer.attention.stage_06_attention_score(
        layer.attention.stage_03_query(rotated, *weights.query),
        layer.attention.stage_04_key(rotated, *weights.key))

    return decode_score_diagonals(engine, scores), scores, tokens, engine, layer.attention


def decode_score_diagonals(engine, scores):
    """Inverse of `encode_score_diagonals`; `test_layouts_round_trip` holds the two together."""
    out = np.zeros((G.n_blocks, G.dim, G.dim))
    for ct in range(len(scores)):
        slots = np.asarray(engine.decrypt(scores[ct]), dtype=complex)
        for group in range(G.pack):
            diagonal = ct * G.pack + group
            for tau in range(G.dim):
                for block in range(G.n_blocks):
                    out[block, tau, (diagonal + tau) % G.dim] = slots[G.slot(group, tau, block)].real
    return out


def test_layouts_round_trip():
    """`decode_score_diagonals` really is the inverse of the encoder it is paired with.

    Without this the decoder is a second transcription of `(diagonal + tau) % dim` and a wrong one
    would misreport every theoretical comparison below as a divergence.
    """
    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    want = np.random.default_rng(3).uniform(-8, 8, (G.n_blocks, G.dim, G.dim))
    got = decode_score_diagonals(engine, encode_score_diagonals(engine, want))
    assert np.max(np.abs(got - want)) < 1e-12


class ProbeRecorder:
    """Collects every probed intermediate, decrypted, keyed by probe name.

    A probe name can repeat - `he_inv` tags its iterations per call site and `update_inv_D` runs
    several times - so repeats are suffixed rather than overwritten. Losing the second occurrence
    would hide exactly the divergence this test is looking for.
    """

    def __init__(self, engine):
        self.engine = engine
        self.values: dict = {}
        self.masks: dict = {}
        self._counts: dict = {}

    def __call__(self, name, value, mask=None):
        seen = self._counts.get(name, 0)
        self._counts[name] = seen + 1
        key = name if seen == 0 else f"{name}#{seen + 1}"
        self.values[key] = self._decrypt(value)
        self.masks[key] = None if mask is None else np.asarray(mask, dtype=float)

    def _decrypt(self, value):
        items = value if isinstance(value, (list, tuple, np.ndarray)) else [value]
        flat = np.ravel(np.asarray(items, dtype=object))
        rows = [np.asarray(self.engine.decrypt(ct), dtype=complex) for ct in flat]
        return np.stack(rows) if rows else np.zeros((0, 0))


def compare(name, got, want, mask=None):
    """Max and RMS absolute error over the carried slots, plus the magnitude it sits against."""
    if got.shape != want.shape:
        return dict(name=name, shape_mismatch=(got.shape, want.shape))
    err = np.abs(got - want)
    scale = np.abs(want)
    if mask is not None and mask.size == err.shape[-1]:
        carried = mask > 0
        err = err[..., carried]
        scale = scale[..., carried]
    if err.size == 0:
        return dict(name=name, empty=True)
    denom = max(float(np.max(scale)), 1e-300)
    return dict(name=name, max=float(np.max(err)), rms=float(np.sqrt(np.mean(err ** 2))),
                rel=float(np.max(err) / denom), magnitude=float(np.max(scale)))


def report(rows, left, right):
    print(f"\n{'probe':<46} {'max err':>11} {'rms':>11} {'rel':>9} {'magnitude':>11}")
    print("-" * 92)
    for r in rows:
        if "shape_mismatch" in r:
            print(f"{r['name']:<46} shape {r['shape_mismatch'][0]} vs {r['shape_mismatch'][1]}")
        elif r.get("empty"):
            print(f"{r['name']:<46} (no carried slots)")
        else:
            print(f"{r['name']:<46} {r['max']:>11.4g} {r['rms']:>11.4g} {r['rel']:>9.2e} "
                  f"{r['magnitude']:>11.4g}")
    print(f"({left} vs {right}; 'rel' is max error over the largest carried value in {right})")


# ------------------------------------------------------------------ theoretical truth, per step

def theory_checks(probes, per_head, masks):
    """Recompute each step from the *measured* previous step and report the residual.

    This is the part that locates a divergence rather than just showing one. Comparing every probe
    against an end-to-end theoretical value blames each step for everything upstream of it, so the
    whole tail lights up and the first real fault is indistinguishable from its own consequences.
    Recomputing a step from what actually entered it asks a local question - "given this input, is
    this output right?" - and the first step that fails it is the one that is broken.

    Only relations that need no `delta` bookkeeping are checked. `he_inv` carries its scale in a
    `DeltaCiphertext` beside the value and the probes report the ciphertext alone, so a raw
    Goldschmidt recurrence on probe values would be comparing `value * delta` against `value`. The
    inverse is checked by its defining property instead - `inv_D * D == 1` - which is delta-free and
    is the strongest single statement available about the step that fails on the device.
    """
    carried = masks[0] > 0
    out = []

    def check(name, got, want, note):
        err = np.abs(np.asarray(got) - np.asarray(want))
        out.append(dict(name=name, max=float(np.max(err)), rms=float(np.sqrt(np.mean(err ** 2))),
                        rel=float(np.max(err) / max(float(np.max(np.abs(want))), 1e-300)),
                        magnitude=float(np.max(np.abs(want))), note=note))

    if "07a.refreshed_scores" in probes and "07a0.score_refresh_input" in probes:
        packed = probes["07a0.score_refresh_input"]
        refreshed = probes["07a.refreshed_scores"]
        half = refreshed.shape[0] // 2
        check("07a = 2*Re(07a0)", refreshed[:half][..., carried],
              2 * packed.real[..., carried], "the bootstrap fold's doubling")
        check("07a = 2*Im(07a0)", refreshed[half:][..., carried],
              2 * packed.imag[..., carried], "the bootstrap fold's doubling")

    if "07b.exp" in probes and "07c.denominator" in probes:
        # `_sum_over_groups` adds the score ciphertexts then folds with `interval_sum(_, group_size)`,
        # which rotates by `-group_size * 2**step`. So the fold is *strided*, not contiguous:
        # out[i] = sum_j x[i + j*group_size]. Reshaping (pack, group_size) and summing axis 0 is that
        # sum; reshaping the other way round sums neighbours instead and reports ~100% residual on
        # exact arithmetic, which is how this comment came to be written.
        exp_total = probes["07b.exp"].sum(axis=0)
        folded = exp_total.reshape(G.pack, G.group_size).sum(axis=0)
        want = np.tile(folded, G.pack)
        check("07c = groupwise sum of 07b", probes["07c.denominator"][0][carried],
              want[carried], "_sum_over_groups")

    if "07c.denominator" in probes and "07d.inverse_denominator" in probes:
        # `he_inv` returns `(inv_D, delta, precision)` and the value it represents is
        # `inv_D * delta`; the probe reports the ciphertext alone. So `D * inv_D` is not 1 - it is
        # `1/delta`, one scalar for every slot. That still gives a delta-free statement, and a
        # sharper one than a magnitude: whatever delta is, `D * inv_D` has to be the *same number*
        # on every carried slot. A slot where the inversion went wrong breaks the constancy, and
        # the size of the break is the error relative to the rest.
        D = probes["07c.denominator"][0][carried].real
        inv = probes["07d.inverse_denominator"][0][carried].real
        product = D * inv
        level = float(np.median(product))
        check("07d * 07c constant across slots", product, np.full(product.shape, level),
              f"he_inv up to its delta; the constant is 1/delta = {level:.6g}")

    return out


def goldschmidt_trace(probes):
    """|b| per iteration, which is the iteration's own convergence story, read delta-free.

    Not monotone, and it is worth saying so rather than asserting otherwise: on exact arithmetic it
    drops sharply for two iterations and then sits on a plateau, because `_restore_magnitude`
    rescales `a` and `b` together between steps. What matters on the device is the comparison with
    this column, not a shape claimed in advance - a run where |b| climbs away instead of settling is
    diverging, and the iteration where the two columns part is where it began.
    """
    rows = []
    for name in sorted(k for k in probes if "_b" in k and "inv_iter" in k and "_times" not in k):
        rows.append((name, float(np.max(np.abs(probes[name])))))
    return rows


def report_theory(rows, title):
    if not rows:
        return
    print(f"\n{title}")
    print(f"{'relation':<40} {'max resid':>11} {'rms':>11} {'rel':>9}   note")
    print("-" * 100)
    for r in rows:
        print(f"{r['name']:<40} {r['max']:>11.4g} {r['rms']:>11.4g} {r['rel']:>9.2e}   {r['note']}")


# ------------------------------------------------------------------ control: no device needed

def test_softmax_algorithm_against_true_softmax(mask_families, goldschmidt_probes):
    """The clear column on its own: THOR's construction against numpy, and every step against theory.

    Exact arithmetic, so whatever error it shows is the algorithm's - the degree-15 exponential fit
    and the Goldschmidt truncation - and it is the floor the device cannot beat. Run it before
    reading any device number, so a device failure is not attributed to noise when the construction
    was already off. It runs on synthetic scores, because it is checking relations that hold at any
    input range; `test_softmax_on_a_real_mrpc_example` is the one that has to use the real range.
    """
    per_head = synthetic_scores()
    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    stages = build_stages(engine, mask_families)
    recorder = ProbeRecorder(engine)
    stages.probe = recorder

    masks = [used_slots()] * 2 * G.n_output_ciphertexts
    weights = stages.stage_07_softmax(feed(engine, per_head), masks, layer_index=0,
                                      parameters=Softmax.NARROW)

    print(f"\nprobes captured: {len(recorder.values)}")
    report_theory(theory_checks(recorder.values, per_head, masks), "step recomputed from its input:")
    trace = goldschmidt_trace(recorder.values)
    if trace:
        print("\nGoldschmidt |b| per iteration (must fall toward zero):")
        for name, value in trace:
            print(f"  {name:<40} {value:.6g}")

    got = decode_broadcast_diagonals(engine, weights)
    want = true_softmax(per_head)
    err = float(np.max(np.abs(got - want)))
    rows = float(np.max(np.abs(got.sum(axis=2) - 1.0)))
    print(f"\nclear vs true softmax: max abs error {err:.6g}; rows sum to 1 within {rows:.6g}")
    # The same bar test_stage9 holds the construction to; it is approximate by design.
    assert err < 5e-3, "THOR's softmax construction is off before any FHE noise is involved"
    assert rows < Softmax.NARROW["output_alpha"]

    # Measured on this engine, not guessed: `07c` comes out at 1e-16 (float64 roundoff) and the
    # inverse's constancy at 7e-4, which is the Goldschmidt residual and inside `output_alpha`.
    # They are asserted so that a change to `interval_sum` or to `he_inv` that breaks the relation
    # fails here rather than being read off a device table as an FHE problem.
    by_name = {r["name"]: r for r in theory_checks(recorder.values, per_head, masks)}
    assert by_name["07c = groupwise sum of 07b"]["rel"] < 1e-12
    assert by_name["07d * 07c constant across slots"]["rel"] < Softmax.NARROW["output_alpha"]


def test_softmax_on_a_real_mrpc_example(mask_families, goldschmidt_probes):
    """The same construction on the range it was actually calibrated for.

    `Softmax.NARROW` is [-27.25, 21.73], measured on real MRPC activations, and `he_exp`'s guard
    refuses a score outside it - a minimax fit 1.7x outside its interval does not degrade, it takes
    off. Synthetic scores in [-8, 8] never approach either end, so they exercise neither the guard
    nor the fit's edges. This runs the pinned MRPC row through the real checkpoint and reports where
    in the window it actually lands, which is the number the device comparison needs alongside it.
    """
    sentence1, sentence2, label = mrpc_example()
    per_head, scores, tokens, engine, _ = real_scores()
    print(f"\nMRPC {MRPC[2]}[{MRPC_ROW}] (label {label}), {tokens}/{G.dim} tokens")
    print(f"  1: {sentence1}")
    print(f"  2: {sentence2}")

    lo, hi = float(np.min(2 * per_head)), float(np.max(2 * per_head))
    window = (Softmax.NARROW["min_x"], Softmax.NARROW["max_x"])
    print(f"  doubled score range [{lo:.3f}, {hi:.3f}] against NARROW {window}")

    stages = build_stages(engine, mask_families)
    recorder = ProbeRecorder(engine)
    stages.probe = recorder
    masks = [used_slots()] * len(scores)
    weights = stages.he_softmax(list(scores), masks, **Softmax.NARROW)

    report_theory(theory_checks(recorder.values, per_head, masks), "step recomputed from its input:")
    got = decode_broadcast_diagonals(engine, weights)
    want = true_softmax(per_head)
    print(f"\nclear vs true softmax on real data: max abs error "
          f"{float(np.max(np.abs(got - want))):.6g}")

    assert lo >= window[0] and hi <= window[1], (
        f"the real doubled scores [{lo:.3f}, {hi:.3f}] fall outside NARROW {window}; "
        "the window and the data disagree, which is a calibration problem, not an FHE one")


# ------------------------------------------------------------------ the device comparison

@BENCH
def test_device_softmax_matches_plaintext_step_by_step(device, mask_families, goldschmidt_probes):
    """The same softmax on the device and on exact arithmetic, probe by probe.

    No per-probe threshold is asserted. The useful output is the table: the first row where the error
    jumps is where the device diverges, and everything below it is downstream of that. A per-probe
    bound would have to be invented per row and would turn a diagnostic into a guess.
    """
    import pyfideslib as pf
    from test_bootstrap_noise_level import bench_params

    per_head = synthetic_scores()
    parameters = Softmax.NARROW
    masks = [used_slots()] * 2 * G.n_output_ciphertexts

    # Clear first: it doubles as the dry run that says which rotation keys the device needs.
    clear = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    clear_stages = build_stages(clear, mask_families)
    clear_probe = ProbeRecorder(clear)
    clear_stages.probe = clear_probe
    clear_stages.stage_07_softmax(feed(clear, per_head), masks,
                                  layer_index=0, parameters=parameters)
    rotations = dict(clear.rotation_levels)
    print(f"\nrotation keys the softmax needs: {len(rotations)}")

    params = dict(bench_params())
    params["rotation_indexes"] = rotations
    engine = pf.Engine(device, **params)
    device_stages = build_stages(engine, mask_families)
    device_probe = ProbeRecorder(engine)
    device_stages.probe = device_probe
    device_stages.stage_07_softmax(feed(engine, per_head), masks,
                                   layer_index=0, parameters=parameters)

    shared = [k for k in clear_probe.values if k in device_probe.values]
    missing = [k for k in clear_probe.values if k not in device_probe.values]
    rows = [compare(k, device_probe.values[k], clear_probe.values[k], clear_probe.masks.get(k))
            for k in shared]
    report(rows, "device", "clear")
    if missing:
        print(f"probes the device never reached (it stopped early): {missing}")

    # The second column, and the one that localises. device-vs-clear says *that* the device drifted;
    # recomputing each step from the device's own measured input says *which step* did it, because a
    # step is then judged on what actually entered it rather than on everything upstream.
    report_theory(theory_checks(device_probe.values, per_head, masks),
                  "device, each step recomputed from its own measured input:")
    report_theory(theory_checks(clear_probe.values, per_head, masks),
                  "clear, same relations (the baseline the device column is read against):")

    device_b = goldschmidt_trace(device_probe.values)
    clear_b = goldschmidt_trace(clear_probe.values)
    if device_b:
        print("\nGoldschmidt |b| per iteration:")
        print(f"{'iteration':<40} {'device':>14} {'clear':>14}")
        clear_by_name = dict(clear_b)
        for name, value in device_b:
            print(f"{name:<40} {value:>14.6g} {clear_by_name.get(name, float('nan')):>14.6g}")

    assert shared, "no probe fired on both engines - the two runs did not take the same path"
