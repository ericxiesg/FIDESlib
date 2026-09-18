"""Encoded weights on disk, keyed by where they came from rather than by what they contain.

Encoding is what a layer spends its time on. `bench --time-ops` on a GV100 put 96.2% of a layer in
`encode_to_light_plaintext`: 19,137 calls at 58 ms is 1109 s of 1154 s, against 4.16 s for all 22
bootstraps. Those 19,137 are the layer's weights, one encode each, and they do not depend on the
sample - the same twelve layers are re-encoded for every run and every sentence.

So persist them. A layer is about 9.3 GiB of light plaintexts (19,137 x N int64, half a MiB each) and
twelve layers about 112 GiB, which is what `docs/light_plaintext.md` traded away to fit the *device*;
on disk it is only disk.

**Keyed by provenance.** The obvious cache is content-addressed - hash the array, name the file after
the digest - and it is the wrong one here. A stable digest has to read every byte: measured on this
machine, sha1 is 648 us per array and blake2b 1018 us, so 12 s to 20 s a layer spent hashing arrays
we already know the names of. A weight is identified by the checkpoint it came from, the layer, the
field and its position in that field, and all four are free. `Stages.plaintext`'s content cache stays
for the masks, which are rebuilt at their call sites and genuinely need it.

**The build is a thunk.** `store(index, name, build)` takes the function that would construct the
numpy arrays, not the arrays, so a hit never builds them: a warm cache skips the encode *and* the
9.7 GiB of host arrays that feed it. That is also why this hangs off `encode_layer`'s fields rather
than off `Stages.plaintext`, which only ever sees an array that has already been built.

It removes the held memory too. A field that arrives as light plaintexts is not an ndarray, so
`Stages.plaintext` passes it straight through and never adds it to `_plaintexts` - whose entries are
half a MiB each and which nothing evicts.
"""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import numpy as np

#: Bumped when the on-disk layout or the manifest changes, so a stale tree is ignored rather than
#: half-read. It is part of the directory name, so old and new can coexist.
LAYOUT = "v1"


def _flatten(value, leaves: list):
    """Describe ``value``'s shape as JSON and collect its arrays, in a fixed order.

    Tuples, lists and object arrays are structure; a numeric array is a leaf. Anything else is a
    scalar the layer folded in, and small enough to keep in the manifest.
    """
    if isinstance(value, np.ndarray) and value.dtype == object:
        return {"t": "obj", "shape": list(value.shape),
                "items": [_flatten(item, leaves) for item in value.ravel()]}
    if isinstance(value, np.ndarray):
        leaves.append(value)
        return {"t": "leaf", "i": len(leaves) - 1}
    if isinstance(value, (tuple, list)):
        return {"t": "seq", "tuple": isinstance(value, tuple),
                "items": [_flatten(item, leaves) for item in value]}
    if value is None or isinstance(value, (int, float, bool, str)):
        return {"t": "const", "v": value}
    raise TypeError(f"a layer field holds {type(value).__name__}, which has no place on disk")


def _graft(tree, leaves):
    """Rebuild what :func:`_flatten` described, with ``leaves[i]`` in place of each array."""
    kind = tree["t"]
    if kind == "leaf":
        return leaves[tree["i"]]
    if kind == "const":
        return tree["v"]
    if kind == "seq":
        items = [_graft(item, leaves) for item in tree["items"]]
        return tuple(items) if tree["tuple"] else items
    if kind == "obj":
        out = np.empty(len(tree["items"]), dtype=object)
        for index, item in enumerate(tree["items"]):
            out[index] = _graft(item, leaves)
        return out.reshape(tree["shape"])
    raise ValueError(f"unknown node {kind!r} in a stored field")


class PlaintextStore:
    """Encoded fields under ``root``, one directory per (tag, layer, field).

    ``encode``, ``write`` and ``read`` are the three engine calls this needs, passed in rather than
    taken from an engine so the store can be tested without one.
    """

    def __init__(self, root, encode, write, read, *, tag: str = "default"):
        self.root = Path(root)
        self.encode = encode
        self.write = write
        self.read = read
        self.tag = tag
        #: (hits, misses) in fields, not in plaintexts - one field is thousands of them.
        self.hits = 0
        self.misses = 0

    def directory(self, layer_index: int, name: str) -> Path:
        return self.root / LAYOUT / self.tag / f"layer{layer_index:02d}" / name

    def __call__(self, layer_index: int, name: str, build):
        """The encoded field, from disk when it is there and written there when it is not."""
        where = self.directory(layer_index, name)
        manifest = where / "tree.json"
        if manifest.is_file():
            try:
                tree = json.loads(manifest.read_text(encoding="utf-8"))
                leaves = [self.read(where / f"{index:06d}.flpt")
                          for index in range(tree["leaves"])]
            except (OSError, ValueError, KeyError) as error:
                # A half-written or stale directory is a cache problem, not a run problem: say so and
                # rebuild it. Raising here would make a corrupted cache look like a broken model.
                print(f"thorfhe: ignoring unreadable plaintext cache at {where} ({error})")
            else:
                self.hits += 1
                return _graft(tree["tree"], leaves)

        value = build()
        arrays: list = []
        tree = _flatten(value, arrays)
        encoded = [self.encode(array) for array in arrays]

        where.mkdir(parents=True, exist_ok=True)
        for index, light in enumerate(encoded):
            self.write(light, where / f"{index:06d}.flpt")
        # The manifest is written last and atomically, so its presence means every leaf beside it is
        # complete. A run killed midway leaves files without a manifest, which the next run overwrites.
        self._write_atomically(manifest,
                               json.dumps({"leaves": len(arrays), "tree": tree}).encode("utf-8"))
        self.misses += 1
        return _graft(tree, encoded)

    @staticmethod
    def _write_atomically(path: Path, payload: bytes):
        handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
            os.replace(temporary, path)
        except BaseException:
            os.unlink(temporary)
            raise

    def describe(self) -> str:
        return (f"plaintext cache {self.root / LAYOUT / self.tag}: "
                f"{self.hits} fields loaded, {self.misses} encoded")


def store_for(engine, root, *, tag: str = "default") -> "PlaintextStore | None":
    """A store over ``engine``'s light-plaintext calls, or ``None`` on an engine without them.

    ``ClearEngine`` has no encoded form - its plaintexts are the arrays themselves - so there is
    nothing to persist and the caller gets `None` rather than a store that pretends.
    """
    if root is None or not hasattr(engine, "encode_to_light_plaintext"):
        return None
    return PlaintextStore(root, engine.encode_to_light_plaintext,
                          lambda light, path: engine.write_light_plaintext(light, str(path)),
                          lambda path: engine.read_light_plaintext(str(path)), tag=tag)
