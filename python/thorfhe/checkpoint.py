"""Reading a Hugging Face checkpoint into numpy arrays, without torch or safetensors.

The benchmark machine has numpy and little else, and the two formats a BERT checkpoint comes in are
both simple enough to read directly:

* **safetensors** is a u64 header length, a JSON header of ``{name: {dtype, shape, data_offsets}}``,
  and one contiguous data buffer. Nothing to it.
* **``pytorch_model.bin``** is a pickle either way. Since torch 1.6 it is a zip holding one pickle
  and one flat file per storage; before that it was five pickles followed by the storages back to
  back. Both are handled, because checkpoints uploaded to the Hub years ago - the usual MRPC ones
  included - are still in the old layout. Reading either needs an unpickler that resolves
  ``torch._utils._rebuild_tensor_v2`` and the storage persistent-ids to numpy views; pickles are
  code, so the one here refuses every global not on an explicit allow-list rather than trusting the
  file.

Both give ``{parameter name: np.ndarray}``, and both convert to float32 - a bf16 or fp16 checkpoint is
widened on the way in, since everything downstream is float arithmetic anyway.
"""
from __future__ import annotations

import json
import pickle
import struct
import zipfile
from pathlib import Path

import numpy as np

#: safetensors dtype names to numpy. BF16 has no numpy equivalent and is widened by hand.
SAFETENSORS_DTYPES = {
    "F64": np.float64, "F32": np.float32, "F16": np.float16,
    "I64": np.int64, "I32": np.int32, "I16": np.int16, "I8": np.int8, "U8": np.uint8,
    "BOOL": np.bool_,
}

#: torch storage class names to numpy, for the pickle path.
TORCH_STORAGE_DTYPES = {
    "DoubleStorage": np.float64, "FloatStorage": np.float32, "HalfStorage": np.float16,
    "LongStorage": np.int64, "IntStorage": np.int32, "ShortStorage": np.int16,
    "CharStorage": np.int8, "ByteStorage": np.uint8, "BoolStorage": np.bool_,
    "BFloat16Storage": "bfloat16",
}


class CheckpointError(RuntimeError):
    pass


def _widen_bfloat16(raw: bytes, count: int) -> np.ndarray:
    """bfloat16 is the top 16 bits of a float32, so widening is a shift, not a conversion."""
    halves = np.frombuffer(raw, dtype=np.uint16, count=count)
    return (halves.astype(np.uint32) << 16).view(np.float32)


# ---------------------------------------------------------------------- safetensors
def load_safetensors(path) -> dict[str, np.ndarray]:
    path = Path(path)
    with open(path, "rb") as handle:
        (header_length,) = struct.unpack("<Q", handle.read(8))
        header = json.loads(handle.read(header_length))
        buffer = handle.read()

    out = {}
    for name, entry in header.items():
        if name == "__metadata__":
            continue
        start, end = entry["data_offsets"]
        raw = buffer[start:end]
        shape = tuple(entry["shape"])
        count = int(np.prod(shape)) if shape else 1

        if entry["dtype"] == "BF16":
            values = _widen_bfloat16(raw, count)
        else:
            dtype = SAFETENSORS_DTYPES.get(entry["dtype"])
            if dtype is None:
                raise CheckpointError(f"{name}: unsupported safetensors dtype {entry['dtype']}")
            values = np.frombuffer(raw, dtype=dtype, count=count)
        out[name] = values.reshape(shape).astype(np.float32, copy=True)
    return out


# ---------------------------------------------------------------------- pytorch_model.bin
class _Storage:
    """A placeholder for a torch storage: the pickle names it, the zip holds its bytes."""

    def __init__(self, key, dtype, count):
        self.key, self.dtype, self.count = key, dtype, count


class _StateDict(dict):
    """A dict with a ``__dict__``: torch pickles an OrderedDict and then BUILDs ``_metadata`` onto it,
    which a bare ``dict`` instance cannot carry."""


def _rebuild_tensor_v2(storage, storage_offset, size, stride, *_):
    return ("tensor", storage, storage_offset, tuple(size), tuple(stride))


def _rebuild_from_type_v2(func, _new_type, args, _state):
    return func(*args)


class _RestrictedUnpickler(pickle.Unpickler):
    """Resolves only what a plain state_dict needs. Anything else is a refusal, not an import."""

    ALLOWED = {
        ("torch._utils", "_rebuild_tensor_v2"): _rebuild_tensor_v2,
        ("torch._utils", "_rebuild_tensor"): _rebuild_tensor_v2,
        ("torch._tensor", "_rebuild_from_type_v2"): _rebuild_from_type_v2,
        ("collections", "OrderedDict"): _StateDict,
    }

    def find_class(self, module, name):
        if (module, name) in self.ALLOWED:
            return self.ALLOWED[(module, name)]
        if module.startswith("torch") and name in TORCH_STORAGE_DTYPES:
            return TORCH_STORAGE_DTYPES[name]
        if module == "torch" and name in ("float32", "float16", "bfloat16", "int64", "device"):
            return name
        raise CheckpointError(
            f"refusing to resolve {module}.{name} while unpickling a checkpoint - this file is "
            f"not a plain state_dict")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and pid and pid[0] == "storage"):
            raise CheckpointError(f"unexpected persistent id {pid!r}")
        # zip format: ("storage", storage_type, key, location, numel)
        # legacy format: the same with a trailing view_metadata
        _, storage_type, key, _location, count = pid[:5]
        return _Storage(str(key), storage_type, int(count))


def load_torch_bin(path) -> dict[str, np.ndarray]:
    path = Path(path)
    if not zipfile.is_zipfile(path):
        return load_torch_legacy(path)

    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        pickles = [n for n in names if n.endswith("data.pkl")]
        if not pickles:
            raise CheckpointError(f"{path.name} has no data.pkl")
        prefix = pickles[0][: -len("data.pkl")]

        with archive.open(pickles[0]) as handle:
            state = _RestrictedUnpickler(handle).load()

        out = {}
        for name, value in state.items():
            if not (isinstance(value, tuple) and value and value[0] == "tensor"):
                continue
            _, storage, offset, shape, stride = value
            with archive.open(f"{prefix}data/{storage.key}") as handle:
                raw = handle.read()
            out[name] = _materialise(raw, storage, offset, shape, stride)
    return out


def _materialise(storage_bytes, storage, offset, shape, stride) -> np.ndarray:
    """One tensor out of its storage: dtype, offset, shape, and a stride only if it is not trivial."""
    if storage.dtype == "bfloat16":
        flat = _widen_bfloat16(storage_bytes, len(storage_bytes) // 2)
    else:
        flat = np.frombuffer(storage_bytes, dtype=storage.dtype)

    count = int(np.prod(shape)) if shape else 1
    if stride and shape and tuple(stride) != _contiguous_stride(shape):
        values = np.lib.stride_tricks.as_strided(
            flat[offset:], shape=shape, strides=tuple(s * flat.dtype.itemsize for s in stride))
    else:
        values = flat[offset:offset + count].reshape(shape)
    return np.ascontiguousarray(values, dtype=np.float32)


def _storage_itemsize(storage):
    return 2 if storage.dtype == "bfloat16" else np.dtype(storage.dtype).itemsize


def load_torch_legacy(path) -> dict[str, np.ndarray]:
    """The pre-1.6 ``torch.save`` layout: five pickles, then the storages back to back.

    ``_legacy_save`` writes a magic number, a protocol version and a ``sys_info`` dict, then the
    object itself, then the list of storage keys in the order their bytes follow - each prefixed with
    an int64 element count. Checkpoints uploaded to the Hub years ago are still in this format, the
    usual MRPC ones included, which is why it is worth the forty lines rather than requiring torch.
    """
    with open(path, "rb") as handle:
        magic = pickle.load(handle)
        if magic != 0x1950A86A20F9469CFC6C:
            raise CheckpointError(f"{Path(path).name}: not a torch checkpoint (magic {magic!r})")
        pickle.load(handle)                                    # protocol version
        pickle.load(handle)                                    # sys_info

        state = _RestrictedUnpickler(handle).load()
        keys = pickle.load(handle)
        descriptors = _storages_by_key(state)

        storages = {}
        for key in keys:
            (count,) = struct.unpack("<q", handle.read(8))
            descriptor = descriptors.get(str(key))
            itemsize = _storage_itemsize(descriptor) if descriptor is not None else 4
            storages[str(key)] = handle.read(count * itemsize)

    out = {}
    for name, value in state.items():
        if not (isinstance(value, tuple) and value and value[0] == "tensor"):
            continue
        _, storage, offset, shape, stride = value
        out[name] = _materialise(storages[storage.key], storage, offset, shape, stride)
    return out


def _storages_by_key(state):
    return {value[1].key: value[1] for value in state.values()
            if isinstance(value, tuple) and value and value[0] == "tensor"}


def _contiguous_stride(shape):
    stride, running = [], 1
    for size in reversed(shape):
        stride.append(running)
        running *= size
    return tuple(reversed(stride))


def load_state_dict(path) -> dict[str, np.ndarray]:
    """Read a checkpoint by extension: ``.safetensors`` or a torch ``.bin``."""
    path = Path(path)
    if path.suffix == ".safetensors":
        return load_safetensors(path)
    return load_torch_bin(path)
