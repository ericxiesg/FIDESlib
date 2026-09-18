"""Benchmark metrics: how fast, how right, and how close to the plaintext model.

Three questions, deliberately kept apart:

* **Accuracy** is against the dataset's labels - what the task scores. MRPC is unbalanced (about 68%
  positive), so accuracy alone flatters a degenerate classifier and GLUE reports F1 beside it.
* **Fidelity** is against the *plaintext model's own* outputs. This is the number that says whether the
  encrypted pipeline is faithful, and it is the one that moves when a polynomial is mis-calibrated.
  A run can be 100% faithful and 70% accurate, or 0% faithful and accidentally accurate.
* **Agreement** is the bridge: how often the two paths pick the same label. Fidelity in probability
  space can look fine while a handful of near-tie samples flip.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np


def softmax(x, axis=-1):
    shifted = np.asarray(x, dtype=np.float64)
    shifted = shifted - shifted.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


@dataclass
class Fidelity:
    """Elementwise closeness of one array to a reference."""

    mae: float
    rmse: float
    max_abs: float
    rel_rmse: float
    count: int

    @classmethod
    def between(cls, got, want) -> "Fidelity":
        got = np.asarray(got, dtype=np.float64).ravel()
        want = np.asarray(want, dtype=np.float64).ravel()
        if got.shape != want.shape:
            raise ValueError(f"shape mismatch: {got.shape} vs {want.shape}")
        error = got - want
        scale = np.sqrt((want ** 2).mean()) or 1.0
        return cls(mae=float(np.abs(error).mean()),
                   rmse=float(np.sqrt((error ** 2).mean())),
                   max_abs=float(np.abs(error).max()) if error.size else 0.0,
                   rel_rmse=float(np.sqrt((error ** 2).mean()) / scale),
                   count=int(error.size))

    def format(self, name: str = "") -> str:
        label = f"{name:<22}" if name else ""
        return (f"{label}MAE {self.mae:.3e}  RMSE {self.rmse:.3e}  "
                f"max {self.max_abs:.3e}  relRMSE {self.rel_rmse:.3e}")


@dataclass
class Classification:
    """Accuracy, F1 and the confusion counts behind them."""

    accuracy: float
    f1: float
    true_positive: int
    false_positive: int
    false_negative: int
    true_negative: int
    total: int

    @classmethod
    def score(cls, predictions, labels, positive: int = 1) -> "Classification":
        predictions = np.asarray(predictions)
        labels = np.asarray(labels)
        tp = int(((predictions == positive) & (labels == positive)).sum())
        fp = int(((predictions == positive) & (labels != positive)).sum())
        fn = int(((predictions != positive) & (labels == positive)).sum())
        tn = int(((predictions != positive) & (labels != positive)).sum())
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        return cls(accuracy=float((predictions == labels).mean()) if labels.size else 0.0, f1=f1,
                   true_positive=tp, false_positive=fp, false_negative=fn, true_negative=tn,
                   total=int(labels.size))

    def format(self, name: str = "") -> str:
        return (f"{name:<22}accuracy {self.accuracy:6.2%}  F1 {self.f1:6.2%}  "
                f"(tp {self.true_positive} fp {self.false_positive} "
                f"fn {self.false_negative} tn {self.true_negative}, n={self.total})")


@dataclass
class ProbabilityAgreement:
    """How far the encrypted class probabilities are from the plaintext ones."""

    l1_mean: float
    l1_max: float
    label_agreement: float
    logit_gap_min: float

    @classmethod
    def between(cls, got_logits, want_logits) -> "ProbabilityAgreement":
        got = softmax(np.asarray(got_logits, dtype=np.float64))
        want = softmax(np.asarray(want_logits, dtype=np.float64))
        l1 = np.abs(got - want).sum(axis=-1)
        agree = (got.argmax(-1) == want.argmax(-1))
        # the smallest plaintext margin among the samples that flipped - a flip at a tiny margin is
        # noise, a flip at a large one is a real divergence
        flipped = np.asarray(want_logits, dtype=np.float64)[~agree]
        gap = float(np.abs(flipped[:, 0] - flipped[:, 1]).min()) if flipped.size else float("inf")
        return cls(l1_mean=float(l1.mean()), l1_max=float(l1.max()) if l1.size else 0.0,
                   label_agreement=float(agree.mean()) if agree.size else 1.0, logit_gap_min=gap)

    def format(self, name: str = "probability") -> str:
        gap = "n/a" if self.logit_gap_min == float("inf") else f"{self.logit_gap_min:.3f}"
        return (f"{name:<22}L1 mean {self.l1_mean:.3e}  max {self.l1_max:.3e}  "
                f"label agreement {self.label_agreement:6.2%}  smallest flipped margin {gap}")


@dataclass
class Timings:
    """Wall-clock, split by phase. Encryption and decryption are reported apart from evaluation."""

    phases: dict = field(default_factory=dict)
    samples: int = 0

    def add(self, phase: str, seconds: float):
        self.phases[phase] = self.phases.get(phase, 0.0) + seconds

    @property
    def total(self) -> float:
        return sum(self.phases.values())

    def format(self) -> str:
        lines = []
        for phase, seconds in sorted(self.phases.items(), key=lambda kv: -kv[1]):
            share = seconds / self.total if self.total else 0.0
            lines.append(f"  {phase:<24}{seconds:9.2f}s  {share:5.1%}")
        lines.append(f"  {'total':<24}{self.total:9.2f}s")
        if self.samples:
            lines.append(f"  {'per sample':<24}{self.total / self.samples:9.2f}s "
                         f"({self.samples} samples)")
        return "\n".join(lines)


@dataclass
class OpTimings:
    """Wall-clock and call count per engine primitive.

    `Timings` splits a run by phase, which says a layer took 1131 seconds but not what it spent them
    on. A layer issues about 53,000 engine calls and they are not alike: a plaintext multiply and a
    bootstrap differ by four orders of magnitude, and a device at 0% utilisation is a statement about
    where the *other* time went. Recorded around the engine rather than inside it, so it costs one
    dictionary update per call and works the same on both engines.
    """

    counts: dict = field(default_factory=dict)
    seconds: dict = field(default_factory=dict)

    def record(self, name: str, elapsed: float):
        self.counts[name] = self.counts.get(name, 0) + 1
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed

    @property
    def total(self) -> float:
        return sum(self.seconds.values())

    def format(self) -> str:
        total = self.total
        lines = [f"  {'primitive':<18}{'calls':>9}{'seconds':>11}{'share':>8}{'us/call':>11}"]
        for name, seconds in sorted(self.seconds.items(), key=lambda kv: -kv[1]):
            count = self.counts[name]
            lines.append(f"  {name:<18}{count:>9}{seconds:>11.2f}{seconds / total if total else 0:>8.1%}"
                         f"{seconds / count * 1e6:>11.1f}")
        lines.append(f"  {'total':<18}{sum(self.counts.values()):>9}{total:>11.2f}")
        return "\n".join(lines)


#: What `--time-ops` wraps. Everything that reaches the device, plus the two host-side encodings,
#: because the point of the measurement is to tell those apart.
TIMED_PRIMITIVES = (
    "multiply", "add", "add_inplace", "subtract", "negate", "rotate", "rescale", "relinearize",
    "square", "conjugate", "multiply_1j", "bootstrap", "level_down", "encode", "encrypt",
    "decrypt", "encode_to_light_plaintext", "noise_level", "level")


def instrument(engine, ops: "OpTimings"):
    """Wrap ``engine``'s primitives in place so each call is counted and timed.

    Instance attributes, so nothing is patched globally and a second engine in the same process is
    unaffected. Returns the engine for chaining.
    """
    import time

    for name in TIMED_PRIMITIVES:
        bound = getattr(engine, name, None)
        if bound is None or not callable(bound):
            continue

        def wrapper(*args, _call=bound, _name=name, **kwargs):
            start = time.perf_counter()
            try:
                return _call(*args, **kwargs)
            finally:
                ops.record(_name, time.perf_counter() - start)

        setattr(engine, name, wrapper)
    return engine


def as_dict(*records) -> dict:
    """Flatten a set of metric records into one JSON-serialisable dict."""
    out = {}
    for record in records:
        if record is None:
            continue
        name = type(record).__name__.lower()
        out[name] = asdict(record) if hasattr(record, "__dataclass_fields__") else record
    return out
