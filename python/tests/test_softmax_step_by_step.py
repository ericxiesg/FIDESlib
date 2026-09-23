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


# ------------------------------------------------------------------ control: no device needed

def test_softmax_algorithm_against_true_softmax(mask_families, goldschmidt_probes):
    """The clear column on its own: THOR's construction against numpy, and every probe listed.

    This is exact arithmetic, so whatever error it shows is the algorithm's - the degree-15
    exponential fit and the Goldschmidt truncation - and it is the floor the device cannot beat. Run
    it before reading any device number, so a device failure is not attributed to noise when the
    construction was already off.
    """
    per_head = synthetic_scores()
    engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    stages = build_stages(engine, mask_families)
    recorder = ProbeRecorder(engine)
    stages.probe = recorder

    weights = stages.stage_07_softmax(feed(engine, per_head),
                                      [used_slots()] * 2 * G.n_output_ciphertexts,
                                      layer_index=0, parameters=Softmax.NARROW)

    print(f"\nprobes captured: {len(recorder.values)}")
    for name, row in recorder.values.items():
        print(f"  {name:<46} shape {str(row.shape):<14} max |.| {np.max(np.abs(row)):.6g}")

    got = decode_broadcast_diagonals(engine, weights)
    want = true_softmax(per_head)
    err = float(np.max(np.abs(got - want)))
    print(f"\nclear engine vs true softmax: max abs error {err:.6g}")
    # The same bar test_stage9 holds the construction to; it is approximate by design.
    assert err < 5e-3, "THOR's softmax construction is off before any FHE noise is involved"


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

    assert shared, "no probe fired on both engines - the two runs did not take the same path"
