"""Which CKKS parameters the network survives, measured against the device's own bootstrap error.

A bootstrap's error is *absolute*: it is set by `q0/Delta` and not by the value being refreshed, so
refreshing a small number is far less accurate in relative terms than refreshing a large one. The
smallest numbers in a THOR layer are the softmax denominator and LayerNorm's normalised variance, and
both are inverted immediately afterwards - which is why the choice of `scaling_bits` is a correctness
question and not a precision-tuning one.

`ClearEngine(noise_model=True)` carries that error, so the question is settled here rather than on a
GPU: a 19-minute device run becomes a few seconds. The precision is *not* a constant - it is measured
per `q0/Delta` - so each configuration below uses the figure measured at its own bound:

    sb=50 fmb=55   q0/Delta = 32   1.56e-02   11.0 bits of the bound
    sb=55 fmb=60   q0/Delta = 32   4.88e-04   16.0
    sb=59 fmb=60   q0/Delta =  2   1.55e-06   20.3

Bit counts are relative to the bound and so are not comparable between rows; the absolute error is.
"""
import numpy as np
import pytest

from thorfhe import THOR_BERT, ClearEngine, block_diagonal_masks
from thorfhe.attention import attention_rotate_masks, ccmm_masks, make_copies_masks, transpose_masks
from thorfhe.softmax import Softmax

G = THOR_BERT
DEPTH = 60

#: (scaling_bits, first_mod_bits, bits of the bound the device reproduces there).
DEVICE = {"sb=50": (50, 55, 11), "sb=59": (59, 60, 20)}


@pytest.fixture(scope="module")
def mask_families():
    return (block_diagonal_masks(G), transpose_masks(G), make_copies_masks(G),
            attention_rotate_masks(G), ccmm_masks(G))


def used_slots():
    return ((np.arange(G.slot_count) % G.n_slot) < G.n_blocks).astype(float)


def run_softmax(mask_families, parameters, configuration):
    (low, high), transpose, copies, attention, ccmm = mask_families
    if configuration is None:
        engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH)
    else:
        sb, fmb, bits = DEVICE[configuration]
        engine = ClearEngine(G, depth=DEPTH, bootstrap_level=DEPTH, noise_model=True,
                             scaling_bits=sb, first_mod_bits=fmb, bootstrap_precision_bits=bits)
    engine.bootstrap_message_margin = 1e9      # measuring precision here, not magnitude
    stages = Softmax(engine, G, masks=low, complement_masks=high, transpose=transpose, copies=copies,
                     attention=attention, ccmm=ccmm, ones=engine.encrypt(used_slots()))
    stages.check_ranges = False                # let it diverge, so the size of the failure shows

    # a score distribution shaped like the real checkpoint's: mostly low, a few high. Uniform scores
    # put the denominator somewhere `LAYERS` was never calibrated for, and then this measures nothing.
    rng = np.random.default_rng(101)
    scores = rng.normal(size=(G.n_blocks, G.dim, G.dim)) * 3.0 - 2.0
    encoded = []
    for ct in range(2 * G.n_output_ciphertexts):
        message = np.zeros(G.slot_count, dtype=complex)
        for group in range(G.pack):
            diagonal = ct * G.pack + group
            for tau in range(G.dim):
                for block in range(G.n_blocks):
                    message[G.slot(group, tau, block)] = scores[block, tau, (diagonal + tau) % G.dim]
        encoded.append(engine.encrypt(message))

    out = stages.he_softmax(encoded, [used_slots()] * 2 * G.n_output_ciphertexts, **parameters)
    got = np.zeros((G.n_blocks, G.dim, G.dim))
    for diagonal in range(G.dim):
        slots = np.real(engine.decrypt(out[diagonal]))
        for tau in range(G.dim):
            for block in range(G.n_blocks):
                got[block, tau, (diagonal + tau) % G.dim] = slots[G.slot(0, tau, block)]
    shifted = np.exp(scores - scores.max(axis=2, keepdims=True))
    return float(np.abs(got - shifted / shifted.sum(axis=2, keepdims=True)).max())


def test_the_softmax_survives_the_device_at_fifty_nine_bits(mask_families):
    """At `sb=59` the device's own bootstrap error is invisible in the softmax.

    This is the measurement the whole parameter migration rests on, so it is pinned rather than
    argued. On this input it comes out at parity to four significant figures - 0.007460 exact against
    0.007463 with the device's error - and on the real checkpoint, at three: 0.00709 against 0.00709.
    """
    parameters = Softmax.LAYERS[0]
    exact = run_softmax(mask_families, parameters, None)
    device = run_softmax(mask_families, parameters, "sb=59")
    assert device < 0.05, f"the softmax does not survive sb=59: {device:.4g}"
    assert device < 4 * exact, (
        f"sb=59 costs more than it should: {exact:.4g} exact -> {device:.4g} with the device's error")


def test_the_softmax_does_not_survive_the_device_at_fifty_bits(mask_families):
    """And at `sb=50` it does not - which is why the GPU's first end-to-end run came back at 7.5e17.

    The failure is not a loss of accuracy, it is a divergence: the bootstrap's error at `sb=50` is
    1.56e-2 absolute, the denominator it refreshes is smaller than that, and a denominator that comes
    back near zero or negative sends Goldschmidt to infinity. A test that only asserted "less
    accurate" would pass on a broken pipeline.
    """
    # 4.6e10 on this input; 1.8e+184 on the real checkpoint, and NaN on layer 8. Which of those it
    # lands on is not the point - that it is not a softmax is.
    device = run_softmax(mask_families, Softmax.LAYERS[0], "sb=50")
    assert not (device < 1.0), (
        f"sb=50 came back at {device:.4g}, i.e. it survived. The bootstrap error there is 1.56e-2 "
        f"against a denominator of the same size, so either the noise model or `LAYERS` has changed "
        f"and the parameter choice needs re-measuring")


def test_meta_bts_buys_its_bits_back(mask_families):
    """Two bootstraps and one level for `meta_bts_bits` bits of precision.

    `b0 = BTS(x)` is `x + e`; reducing it to the input's modulus and subtracting isolates `-e`, which
    an integer multiply lifts back into a range the bootstrap resolves well, and scaling the refreshed
    residual down again returns the message with the error `k` bits smaller. Measured against a single
    bootstrap at the device's own precision, the gain is exactly `2^k`.

    This is what thor-openfhe gets from `EvalBootstrap(ct, numIterations=2, precision=10)`. FIDESlib's
    GPU path accepts both arguments and forwards them only on its CPU fallback, so passing them there
    does nothing - but every primitive the iteration needs is exposed, which is what this pins.
    """
    from thorfhe.stages import Stages

    (low, high), *_ = mask_families
    rng = np.random.default_rng(3)
    values = rng.uniform(-0.4, 0.4, G.slot_count)

    def error(k):
        engine = ClearEngine(G, depth=40, bootstrap_level=35, noise_model=True, scaling_bits=59,
                             first_mod_bits=60, bootstrap_precision_bits=15)
        engine.bootstrap_message_margin = 1e9
        stages = Stages(engine, G, masks=low, complement_masks=high)
        stages.meta_bts_bits = k
        ct = engine.level_down(engine.encrypt(values), by=30)
        once = stages.bootstrap(ct)
        if k is None:
            return float(np.abs(np.real(engine.decrypt(once)) - values).max()), 0
        out = stages.bootstrap_twice(ct)
        return (float(np.abs(np.real(engine.decrypt(out)) - values).max()),
                engine.level(once) - engine.level(out))

    single, _ = error(None)
    for k in (6, 10):
        meta, levels = error(k)
        assert single / meta == pytest.approx(2 ** k, rel=0.15), (
            f"k={k} should buy {2 ** k}x, got {single / meta:.0f}x")
        assert levels == 1, f"k={k} cost {levels} levels, not one"

    # The ceiling is the message bound: the lifted residual has to stay inside what the sine can
    # recover, so `2^k * |e| < q0/(2*Delta)`. At 15 bits that is 2.5e-04 * 2^13 = 2.07 against 1.0,
    # and the residual comes back as its own residue instead - no gain at all, not a smaller one.
    too_far, _ = error(13)
    assert too_far > single / 2, (
        f"k=13 lifts the residual past the bound and should stop working, but gave "
        f"{single / too_far:.0f}x")
