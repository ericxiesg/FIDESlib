"""A numpy BERT-for-sequence-classification, used as the benchmark's ground truth.

This is the plaintext model the encrypted one is measured against, so it follows HuggingFace exactly
rather than THOR: in particular ``hidden_act`` is honoured, and the default ``"gelu"`` is the erf form,
not the tanh approximation THOR's polynomial is fitted to. The two differ by about 1e-3 at the
pre-activations BERT actually produces, which is part of what the fidelity numbers are measuring.

:func:`layer_parameters` re-keys one layer into the short names :func:`thorfhe.layer.encode_layer`
takes, so the same checkpoint drives both paths.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: The short names one encoder layer is addressed by, once the `bert.encoder.layer.N.` prefix and the
#: `attention.self.` infix are stripped.
LAYER_KEYS = (
    "query.weight", "query.bias", "key.weight", "key.bias", "value.weight", "value.bias",
    "attention.output.dense.weight", "attention.output.dense.bias",
    "attention.output.LayerNorm.weight", "attention.output.LayerNorm.bias",
    "intermediate.dense.weight", "intermediate.dense.bias",
    "output.dense.weight", "output.dense.bias",
    "output.LayerNorm.weight", "output.LayerNorm.bias",
)


def _erf(x):
    """Abramowitz-Stegun 7.1.26, so the reference does not need scipy. ~1.5e-7 absolute."""
    sign = np.sign(x)
    z = np.abs(x)
    t = 1.0 / (1.0 + 0.3275911 * z)
    poly = t * (0.254829592 + t * (-0.284496736 + t * (1.421413741
                + t * (-1.453152027 + t * 1.061405429))))
    return sign * (1.0 - poly * np.exp(-z * z))


def gelu(x):
    return 0.5 * x * (1.0 + _erf(x / math.sqrt(2.0)))


def gelu_tanh(x):
    """The ``gelu_new`` / ``gelu_pytorch_tanh`` form - the one THOR's polynomial approximates."""
    return 0.5 * x * (1.0 + np.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x ** 3)))


ACTIVATIONS = {"gelu": gelu, "gelu_new": gelu_tanh, "gelu_pytorch_tanh": gelu_tanh,
               "gelu_fast": gelu_tanh}


def layer_norm(x, gamma, beta, eps=1e-12):
    """LayerNorm, with the mean and variance taken in float64 whatever the activations are.

    HuggingFace does the same (torch's LayerNorm accumulates in float32 for a half-precision input),
    and it matters here: a float32 sum of 768 terms loses about two digits, which would show up in the
    fidelity numbers as a difference the encrypted path did not cause.
    """
    wide = np.asarray(x, dtype=np.float64)
    mean = wide.mean(axis=-1, keepdims=True)
    variance = ((wide - mean) ** 2).mean(axis=-1, keepdims=True)
    normalised = (wide - mean) / np.sqrt(variance + eps)
    return (normalised.astype(x.dtype, copy=False) * gamma + beta)


def softmax(x, axis=-1):
    shifted = x - x.max(axis=axis, keepdims=True)
    exponentials = np.exp(shifted)
    return exponentials / exponentials.sum(axis=axis, keepdims=True)


@dataclass
class BertConfig:
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 512
    type_vocab_size: int = 2
    layer_norm_eps: float = 1e-12
    hidden_act: str = "gelu"
    num_labels: int = 2

    @classmethod
    def from_file(cls, path) -> "BertConfig":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        labels = raw.get("id2label")
        return cls(
            hidden_size=raw.get("hidden_size", 768),
            num_hidden_layers=raw.get("num_hidden_layers", 12),
            num_attention_heads=raw.get("num_attention_heads", 12),
            intermediate_size=raw.get("intermediate_size", 3072),
            max_position_embeddings=raw.get("max_position_embeddings", 512),
            type_vocab_size=raw.get("type_vocab_size", 2),
            layer_norm_eps=raw.get("layer_norm_eps", 1e-12),
            hidden_act=raw.get("hidden_act", "gelu"),
            num_labels=len(labels) if labels else raw.get("num_labels", 2),
        )


def normalise_keys(state: dict) -> dict:
    """Give every parameter the ``bert.``-prefixed name this module addresses it by.

    Checkpoints saved from ``BertModel`` rather than ``BertForSequenceClassification`` have no prefix,
    and some carry a ``model.`` one instead. Renaming once here beats guessing at every lookup.
    """
    if any(key.startswith("bert.") for key in state):
        return dict(state)
    out = {}
    for key, value in state.items():
        stripped = key[len("model."):] if key.startswith("model.") else key
        if stripped.startswith(("embeddings.", "encoder.", "pooler.")):
            stripped = "bert." + stripped
        out[stripped] = value
    return out


def layer_parameters(state: dict, index: int, *, prefix: str = "bert.encoder.layer") -> dict:
    """One layer's arrays, keyed by :data:`LAYER_KEYS`."""
    base = f"{prefix}.{index}."
    out = {}
    for key in LAYER_KEYS:
        source = base + ("attention.self." + key if key.startswith(("query", "key", "value"))
                         else key)
        if source not in state:
            raise KeyError(f"checkpoint has no {source}")
        out[key] = state[source]
    return out


class BertForSequenceClassification:
    """The plaintext model. ``forward`` optionally returns every intermediate the benchmark compares."""

    def __init__(self, state: dict, config: BertConfig, dtype=np.float32):
        # One dtype for weights *and* activations. Mixing them is not a rounding question: numpy
        # cannot call BLAS on a float64 @ float32 product and falls back to a generic loop, which
        # costs about 15 seconds a sequence instead of a tenth of one.
        self.dtype = np.dtype(dtype)
        self.state = {k: np.ascontiguousarray(v, dtype=self.dtype)
                      for k, v in normalise_keys(state).items()}
        self.config = config
        self.activation = ACTIVATIONS.get(config.hidden_act, gelu)
        missing = [k for k in ("classifier.weight", "classifier.bias") if k not in self.state]
        if missing:
            raise KeyError(f"checkpoint is not a sequence classifier: missing {missing}")

    def _get(self, name):
        return self.state[name]

    def head_parameters(self):
        """The pooler and classifier arrays, under the names the encrypted head's encoders take."""
        return dict(pooler_weight=self._get("bert.pooler.dense.weight"),
                    pooler_bias=self._get("bert.pooler.dense.bias"),
                    classifier_weight=self._get("classifier.weight"),
                    classifier_bias=self._get("classifier.bias"))

    def embeddings(self, input_ids, token_type_ids):
        input_ids = np.asarray(input_ids)
        token_type_ids = np.asarray(token_type_ids)
        words = self._get("bert.embeddings.word_embeddings.weight")[input_ids]
        positions = self._get("bert.embeddings.position_embeddings.weight")[:input_ids.shape[-1]]
        types = self._get("bert.embeddings.token_type_embeddings.weight")[token_type_ids]
        return layer_norm(words + positions + types,
                          self._get("bert.embeddings.LayerNorm.weight"),
                          self._get("bert.embeddings.LayerNorm.bias"),
                          self.config.layer_norm_eps)

    def encoder_layer(self, hidden, index, attention_mask):
        p = layer_parameters(self.state, index)
        heads = self.config.num_attention_heads
        head_dim = self.config.hidden_size // heads
        tokens = hidden.shape[0]

        q = hidden @ p["query.weight"].T + p["query.bias"]
        k = hidden @ p["key.weight"].T + p["key.bias"]
        v = hidden @ p["value.weight"].T + p["value.bias"]
        reshape = lambda t: t.reshape(tokens, heads, head_dim).transpose(1, 0, 2)  # noqa: E731
        qh, kh, vh = reshape(q), reshape(k), reshape(v)

        scores = qh @ kh.transpose(0, 2, 1) / math.sqrt(head_dim)
        unmasked = scores
        if attention_mask is not None:
            mask = np.asarray(attention_mask, dtype=self.dtype)
            scores = scores + (1.0 - mask)[None, None, :] * np.array(-10000.0, self.dtype)
        weights = softmax(scores)
        context = (weights @ vh).transpose(1, 0, 2).reshape(tokens, self.config.hidden_size)

        dense = context @ p["attention.output.dense.weight"].T + p["attention.output.dense.bias"]
        norm_1 = layer_norm(hidden + dense, p["attention.output.LayerNorm.weight"],
                            p["attention.output.LayerNorm.bias"], self.config.layer_norm_eps)

        intermediate = norm_1 @ p["intermediate.dense.weight"].T + p["intermediate.dense.bias"]
        activated = self.activation(intermediate)
        output = activated @ p["output.dense.weight"].T + p["output.dense.bias"]
        norm_2 = layer_norm(norm_1 + output, p["output.LayerNorm.weight"],
                            p["output.LayerNorm.bias"], self.config.layer_norm_eps)

        # THOR does not halve the QKV bias, so its projections give `x @ w.T + 2b`; carrying the
        # convention here keeps the benchmark's per-stage comparison honest about what it expects.
        return norm_2, dict(query=q, key=k, value=v, scores=scores, scores_unmasked=unmasked,
                            query_thor=q + p["query.bias"], key_thor=k + p["key.bias"],
                            value_thor=v + p["value.bias"], attention=weights,
                            context=context, attention_dense=dense, norm_1=norm_1,
                            intermediate=intermediate, gelu=activated, output_dense=output,
                            norm_2=norm_2)

    def forward(self, input_ids, token_type_ids, attention_mask=None, *, trace=False):
        """One sequence. Returns ``(logits, states)``; ``states`` is empty unless ``trace``."""
        input_ids = np.asarray(input_ids)
        token_type_ids = np.asarray(token_type_ids)
        mask = None if attention_mask is None else np.asarray(attention_mask, dtype=self.dtype)

        hidden = self.embeddings(input_ids, token_type_ids)
        states = {"embeddings": hidden} if trace else {}
        for index in range(self.config.num_hidden_layers):
            hidden, inner = self.encoder_layer(hidden, index, mask)
            if trace:
                states[f"layer_{index}"] = inner

        pooled = np.tanh(hidden[0] @ self._get("bert.pooler.dense.weight").T
                         + self._get("bert.pooler.dense.bias"))
        logits = pooled @ self._get("classifier.weight").T + self._get("classifier.bias")
        if trace:
            states["pooled"] = pooled
            states["logits"] = logits
        return logits, states
