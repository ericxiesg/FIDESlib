# The benchmark: `python -m thorfhe.bench`

Speed, accuracy and fidelity for the THOR-on-fideslib pipeline, on real BERT weights and real MRPC
data. Four commands:

| command | what it does |
|---|---|
| `info` | the proxy, endpoint and cache in effect, and what is already downloaded |
| `fetch` | download the checkpoint and the dataset, then stop |
| `reference` | the plaintext numpy model over MRPC: accuracy, F1, wall clock |
| `fhe` | the encrypted pipeline: time, accuracy, and fidelity against the plaintext model |

```bash
python -m thorfhe.bench fetch --proxy http://proxy:3128
python -m thorfhe.bench reference --limit 64 --offline
python -m thorfhe.bench fhe --layers 1 --limit 4 --per-stage --offline
```

## Network, proxy and cache

Every request goes through `thorfhe.hub.HubClient`.

* `--proxy URL` sets both http and https. Without it, `HTTPS_PROXY` / `HTTP_PROXY` / `ALL_PROXY` are
  read in that order, in either case.
* `--hf-endpoint URL` (or `HF_ENDPOINT`) points at a mirror.
* `--cache-dir` (or `THORFHE_CACHE`, default `~/.cache/thorfhe`) holds everything, keyed by repo and
  revision. `fetch` warms it; `--offline` then refuses to touch the network at all, so a measured run
  has no network in it.
* 4xx responses are permanent and are not retried; everything else backs off up to `--retries` times.

## No torch, no transformers

The benchmark machine has numpy and `requests`. Pulling in `transformers` would drag torch along for
two things that are a file format and an HTTP endpoint, so both are done directly:

* **`thorfhe.checkpoint`** reads safetensors, torch's zip format, *and* the pre-1.6 pickle layout that
  the usual MRPC checkpoints on the Hub are still saved in. Pickles are code, so the unpickler
  resolves only an explicit allow-list of globals and refuses anything else rather than importing it.
* **`thorfhe.tokenizer`** is BERT WordPiece: normalise, split on whitespace and punctuation, then
  greedy longest-match-first with `##`. `do_lower_case` comes from the checkpoint's own
  `tokenizer_config.json` rather than being assumed.
* **`thorfhe.bert`** is the plaintext model, following HuggingFace rather than THOR - `hidden_act` is
  honoured, and the default `gelu` is the erf form, not the tanh approximation THOR's polynomial is
  fitted to.

Sanity check: 64 rows of MRPC validation give **84.4% accuracy, 88.6% F1**, in line with the published
numbers for `textattack/bert-base-uncased-MRPC`.

One trap worth knowing: keep the model's weights and activations in the same dtype. numpy cannot call
BLAS on a `float64 @ float32` product and falls back to a generic loop - 15 seconds a sequence instead
of a tenth of one.

## Three kinds of number, deliberately kept apart

* **Accuracy** is against the dataset labels. MRPC is about 68% positive, so accuracy alone flatters a
  degenerate classifier; GLUE reports F1 beside it and so does this.
* **Fidelity** is against the *plaintext model's own outputs* - MAE, RMSE, max absolute, relative
  RMSE. This is what says whether the encrypted pipeline is faithful, and it is what moves when a
  polynomial is mis-calibrated.
* **Probability agreement** is the bridge: the L1 distance between the two class distributions, how
  often they pick the same label, and the smallest plaintext margin among the samples that flipped.
  A flip at a tiny margin is noise; a flip at a large one is a real divergence.

A run can be perfectly faithful and 70% accurate, or wildly unfaithful and accidentally accurate.
Reporting one number would hide both.

## `--per-stage`

Decrypts every stage of the first sample, reads it back in its own layout, and reports its fidelity
*and its best-fit scale* against the plaintext model.

The scale is the point. THOR carries factors of two around - see the doubled-ciphertext convention in
`thor_port.md` - so a stage can be exactly right up to a constant. A best-fit scale at a clean ratio
with a small residual means the stage is correct and the convention is understood; a scale near one
with a large residual means it is not. Every convention documented in `thor_port.md` was found this
way, including the three softmax bugs, each of which ran to completion and produced plausible output.

Padded positions are excluded before comparing: the plaintext reference softmaxes a row of `-10000`s
into a uniform `1/128` while THOR masks those slots to zero, and at 128 positions against about 25
real ones that difference dominates every average while measuring nothing.

## Running the encrypted path

`--layers N` runs the first N encoder layers under CKKS and the rest in plaintext, so the cost can be
paid one layer at a time and the fidelity measured at the boundary. `--engine clear` (the default) is
the numpy mirror under the same level and scale contract - exact arithmetic, so a failure is a
schedule or algebra error and never CKKS noise. `--engine fideslib` is the CUDA backend.

On the clear engine one layer costs about 130 s per sample, plus about 65 s per layer to encode its
weights (6144 slot vectors for each feed-forward matrix). Both are numpy, not CKKS; they say nothing
about GPU throughput.
