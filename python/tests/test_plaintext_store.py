"""Encoded weights on disk: the same field comes back, and a hit builds nothing.

`--time-ops` on the device put 96.2% of a layer in `encode_to_light_plaintext` - 19,137 calls at
58 ms - and the weights do not depend on the sample, so they are encoded once per run for nothing.
The store persists them. What has to hold is that a loaded field is indistinguishable from an encoded
one, and that loading it does not run the thunk: on a hit neither the plaintexts nor the 9.7 GB of
arrays that feed them should be built at all.
"""
import pickle

import numpy as np

from thorfhe.layer import encode_layer
from thorfhe.plaintext_store import LAYOUT, PlaintextStore, store_for


class FakeLight:
    """Stands in for a LightPlaintext: an opaque handle that is not an ndarray."""

    def __init__(self, values):
        self.values = np.asarray(values)

    def __eq__(self, other):
        return isinstance(other, FakeLight) and np.array_equal(self.values, other.values)


def make_store(root, counter=None):
    def encode(array):
        if counter is not None:
            counter["encoded"] += 1
        return FakeLight(array)

    def write(light, path):
        path.write_bytes(pickle.dumps(light.values))

    def read(path):
        return FakeLight(pickle.loads(path.read_bytes()))

    return PlaintextStore(root, encode, write, read, tag="test")


def sample_field():
    weight = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            weight[i, j] = np.arange(4.0) + 10 * i + j
    return weight, (np.ones(3), 2.5)


def test_a_loaded_field_is_the_field_that_was_stored(tmp_path):
    counter = {"encoded": 0}
    store = make_store(tmp_path, counter)
    built = store(0, "query", sample_field)
    # four weight blocks and one bias array are leaves; the 2.5 is a scalar and stays in the manifest
    assert counter["encoded"] == 5

    again = make_store(tmp_path, counter)(0, "query", sample_field)
    assert counter["encoded"] == 5, "a hit must not encode anything"

    assert isinstance(again, tuple) and again[0].shape == (2, 2)
    for i in range(2):
        for j in range(2):
            assert again[0][i, j] == built[0][i, j]
    assert again[1][0] == built[1][0]
    assert again[1][1] == 2.5


def test_a_hit_does_not_run_the_thunk(tmp_path):
    """The point of taking a thunk: on a hit the arrays are never constructed."""
    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return sample_field()

    make_store(tmp_path)(3, "intermediate", build)
    assert calls["n"] == 1
    make_store(tmp_path)(3, "intermediate", build)
    assert calls["n"] == 1, "the field was rebuilt despite being on disk"


def test_fields_and_layers_do_not_collide(tmp_path):
    store = make_store(tmp_path)
    first = store(0, "query", lambda: (np.zeros(2),))
    second = store(0, "value", lambda: (np.ones(2),))
    third = store(1, "query", lambda: (np.full(2, 7.0),))
    assert first[0].values.tolist() == [0, 0]
    assert second[0].values.tolist() == [1, 1]
    assert third[0].values.tolist() == [7, 7]


def test_a_directory_without_a_manifest_is_rebuilt(tmp_path):
    """The manifest is written last, so a run killed midway leaves leaves without one."""
    store = make_store(tmp_path)
    store(0, "query", sample_field)
    (store.directory(0, "query") / "tree.json").unlink()

    calls = {"n": 0}

    def build():
        calls["n"] += 1
        return sample_field()

    make_store(tmp_path)(0, "query", build)
    assert calls["n"] == 1, "a field with no manifest must be re-encoded, not half-read"


def test_a_corrupt_manifest_is_reported_and_rebuilt(tmp_path, capsys):
    store = make_store(tmp_path)
    store(0, "query", sample_field)
    (store.directory(0, "query") / "tree.json").write_text("{not json", encoding="utf-8")

    got = make_store(tmp_path)(0, "query", sample_field)
    assert "ignoring unreadable plaintext cache" in capsys.readouterr().out
    assert got[1][1] == 2.5


def test_a_missing_leaf_is_reported_and_rebuilt(tmp_path, capsys):
    store = make_store(tmp_path)
    store(0, "query", sample_field)
    (store.directory(0, "query") / "000000.flpt").unlink()

    got = make_store(tmp_path)(0, "query", sample_field)
    assert "ignoring unreadable plaintext cache" in capsys.readouterr().out
    assert got[0][0, 0] == FakeLight(np.arange(4.0))


def test_the_tag_separates_incompatible_parameters(tmp_path):
    """The scales are baked into the plaintexts, so two tags must not share a directory."""
    one = PlaintextStore(tmp_path, FakeLight, lambda l, p: p.write_bytes(pickle.dumps(l.values)),
                         lambda p: FakeLight(pickle.loads(p.read_bytes())), tag="rs1")
    two = PlaintextStore(tmp_path, FakeLight, lambda l, p: p.write_bytes(pickle.dumps(l.values)),
                         lambda p: FakeLight(pickle.loads(p.read_bytes())), tag="rs256")
    one(0, "query", lambda: (np.zeros(2),))
    assert not (two.directory(0, "query") / "tree.json").is_file()
    assert one.directory(0, "query") != two.directory(0, "query")
    assert LAYOUT in str(one.directory(0, "query"))


def test_store_for_declines_an_engine_without_light_plaintexts():
    """`ClearEngine`'s plaintexts are the arrays themselves; there is nothing to persist."""
    class Clear:
        pass

    assert store_for(Clear(), "/tmp/whatever") is None
    assert store_for(object(), None) is None


def test_encode_layer_routes_every_field_through_the_store(tmp_path):
    """The store has to see all eight fields, or the ones it misses are re-encoded every run."""
    seen = []

    def store(layer_index, name, build):
        seen.append((layer_index, name))
        return build()

    from thorfhe.geometry import THOR_BERT

    features = THOR_BERT.features
    dummy = {"query.weight": np.zeros((features, features)), "query.bias": np.zeros(features),
             "key.weight": np.zeros((features, features)), "key.bias": np.zeros(features),
             "value.weight": np.zeros((features, features)), "value.bias": np.zeros(features),
             "attention.output.dense.weight": np.zeros((features, features)),
             "attention.output.dense.bias": np.zeros(features),
             "attention.output.LayerNorm.weight": np.zeros(features),
             "attention.output.LayerNorm.bias": np.zeros(features),
             "intermediate.dense.weight": np.zeros((4 * features, features)),
             "intermediate.dense.bias": np.zeros(4 * features),
             "output.dense.weight": np.zeros((features, 4 * features)),
             "output.dense.bias": np.zeros(features),
             "output.LayerNorm.weight": np.zeros(features),
             "output.LayerNorm.bias": np.zeros(features)}

    weights = encode_layer(dummy, 5, lazy=True, store=store)
    for name in ("query", "key", "value", "attention_dense", "attention_norm",
                 "intermediate", "output_dense", "output_norm"):
        getattr(weights, name)
    assert sorted(name for _, name in seen) == sorted(
        ["query", "key", "value", "attention_dense", "attention_norm",
         "intermediate", "output_dense", "output_norm"])
    assert {index for index, _ in seen} == {5}


def test_a_mask_is_content_addressed_and_survives_the_run(tmp_path):
    """Masks have no provenance to key on, so they are hashed - but only 689 a layer, so it is cheap."""
    counter = {"encoded": 0}
    mask = np.arange(16.0)

    first = make_store(tmp_path, counter).plaintext(mask)
    assert counter["encoded"] == 1
    second = make_store(tmp_path, counter).plaintext(mask.copy())
    assert counter["encoded"] == 1, "an identical mask was re-encoded across runs"
    assert first == second

    make_store(tmp_path, counter).plaintext(mask + 1)
    assert counter["encoded"] == 2, "a different mask must not collide with the first"


def test_stages_routes_masks_through_the_store_when_it_has_one(tmp_path):
    from thorfhe import SMALL, ClearEngine, Stages

    class LightEngine(ClearEngine):
        def encode_to_light_plaintext(self, message, level=None):
            return FakeLight(message)

    engine = LightEngine(SMALL, depth=12)
    stages = Stages(engine, SMALL, masks={}, complement_masks={})
    counter = {"encoded": 0}
    stages.plaintext_store = make_store(tmp_path, counter)

    mask = np.ones(SMALL.slot_count)
    assert isinstance(stages.plaintext(mask), FakeLight)
    assert counter["encoded"] == 1
    stages.plaintext(mask.copy())          # in-memory hit, no digest, no store call
    assert counter["encoded"] == 1

    fresh = Stages(engine, SMALL, masks={}, complement_masks={})
    fresh.plaintext_store = make_store(tmp_path, counter)
    fresh.plaintext(mask.copy())           # new process would hit the disk, not re-encode
    assert counter["encoded"] == 1
