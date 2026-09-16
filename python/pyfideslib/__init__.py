"""pyfideslib — Python access to the fideslib CKKS API.

The same context runs on CPU (OpenFHE) or CUDA depending on the device list; ``Engine`` wraps the
common setup and exposes the THOR-style primitive set (see docs/thor-engine-requirements.md).
Messages cross the boundary as numpy arrays; ciphertexts/plaintexts are opaque handles.
"""
from __future__ import annotations

import numpy as np

from ._core import *  # noqa: F401,F403
from . import _core


def parse_device(device: str | int | None) -> list[int]:
    """'cpu' -> [] (OpenFHE backend); 'cuda' / 'cuda:1' / 1 -> [gpu id]."""
    if device is None or device == "cpu":
        return []
    if isinstance(device, int):
        return [device]
    if device.startswith("cuda"):
        _, _, idx = device.partition(":")
        return [int(idx) if idx else 0]
    raise ValueError(f"unknown device {device!r}")


class Engine:
    """Thin THOR-style engine over one fideslib context.

    Parameters mirror the draft OpenFHE port (HEStd_NotSet functional baseline): N=2^16, 32768 slots,
    FIXEDMANUAL rescaling so the caller controls rescale/level exactly like desilofhe.
    """

    def __init__(
        self,
        device: str | int | None = "cpu",
        *,
        log_n: int = 16,
        depth: int = 33,
        scaling_bits: int = 50,
        first_mod_bits: int = 55,
        dnum: int = 3,
        scaling_technique=_core.FIXEDMANUAL,
        security=_core.HEStd_NotSet,
        secret_key_dist=_core.UNIFORM_TERNARY,
        bootstrap_level_budget: tuple[int, int] | None = None,
        bootstrap_level: int | None = None,
        rotation_indexes: dict[int, int] | list[int] | None = None,
        truncate_keys: bool = True,
        allow_key_grow: bool = False,
        light_plaintext_cache: int = 64,
    ):
        self.devices = parse_device(device)
        self.scaling_technique = scaling_technique
        #: level-truncated keys only exist on the GPU backend; OpenFHE always keeps complete keys.
        self.on_gpu = bool(self.devices)
        self.slots = 1 << (log_n - 1)
        self.depth = depth

        p = _core.CCParams()
        p.SetSecretKeyDist(secret_key_dist)
        p.SetSecurityLevel(security)
        p.SetRingDim(1 << log_n)
        p.SetBatchSize(self.slots)
        p.SetNumLargeDigits(dnum)
        p.SetKeySwitchTechnique(_core.HYBRID)
        p.SetScalingModSize(scaling_bits)
        p.SetFirstModSize(first_mod_bits)
        p.SetScalingTechnique(scaling_technique)
        p.SetMultiplicativeDepth(depth)
        p.SetCKKSDataType(_core.COMPLEX)
        p.SetDevices(list(self.devices))
        self.cc = _core.GenCryptoContext(p)
        for f in (_core.PKE, _core.KEYSWITCH, _core.LEVELEDSHE, _core.ADVANCEDSHE, _core.FHE):
            self.cc.Enable(f)
        self.cc.truncate_keys = truncate_keys
        # A truncated key used above its declared level means the (delta -> level) table is wrong; by default
        # that raises instead of silently reloading the key on every call.
        self.cc.allow_key_grow = allow_key_grow
        # How many expanded light plaintexts to keep on the device between calls.
        self.cc.light_plaintext_cache_capacity = light_plaintext_cache

        self.keys = self.cc.KeyGen()
        self.cc.EvalMultKeyGen(self.keys.secretKey)
        self.cc.EvalConjugateKeyGen(self.keys.secretKey)

        if rotation_indexes:
            if isinstance(rotation_indexes, dict):
                idx = sorted(int(k) for k in rotation_indexes)
                self.cc.SetRotationKeyLevels({int(k): int(v) for k, v in rotation_indexes.items()})
            else:
                idx = sorted(int(k) for k in rotation_indexes)
            self.cc.EvalRotateKeyGen(self.keys.secretKey, idx)

        self.bootstrap_enabled = bootstrap_level_budget is not None
        #: The level the caller's level plan assumes a bootstrap restores to. Not a request - the
        #: hardware decides - but the number the plan was built against, so `bootstrap` can say so
        #: the first time the two disagree. Without it a shortfall is silent: the rotation keys were
        #: truncated to levels the run never reaches, and the deficit surfaces many stages later as
        #: a rescale or level-reduce failure, naming the wrong operation in the wrong place.
        self.bootstrap_level = bootstrap_level
        if self.bootstrap_enabled:
            self.cc.EvalBootstrapSetup(list(bootstrap_level_budget), [0, 0], self.slots, 0)
            self.cc.EvalBootstrapKeyGen(self.keys.secretKey, self.slots)

        self.cc.LoadContext(self.keys.publicKey)

    # ---- numpy bridge ----
    def encode(self, msg, level: int = 0, noise_scale_deg: int = 1):
        """level counts consumed levels (OpenFHE convention): 0 = fresh, depth-1 = one level left."""
        arr = np.asarray(msg)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        return self.cc.MakeCKKSPackedPlaintext(arr, noise_scale_deg, level, self.slots)

    def encrypt(self, msg, level: int = 0):
        return self.cc.Encrypt(self.keys.publicKey, self.encode(msg, level))

    def decrypt(self, ct, length: int | None = None):
        return self.cc.Decrypt(self.keys.secretKey, ct, length or self.slots)

    def decrypt_real(self, ct, length: int | None = None):
        return np.real(self.decrypt(ct, length))

    # ---- THOR primitives ----
    def add(self, x, y):
        if isinstance(x, (int, float)):
            return self.cc.EvalAddScalar(y, float(x))
        if isinstance(x, np.ndarray):
            pt = self.encode(x, level=self.depth - self.level(y))
            return self.cc.EvalAddPt(y, pt)
        if isinstance(y, (int, float)):
            return self.cc.EvalAddScalar(x, float(y))
        if isinstance(y, np.ndarray):
            pt = self.encode(y, level=self.depth - self.level(x))
            return self.cc.EvalAddPt(x, pt)
        if isinstance(y, _core.LightPlaintext):
            return self.cc.EvalAddLightPt(x, y)
        if isinstance(y, _core.Plaintext):
            return self.cc.EvalAddPt(x, y)
        return self.cc.EvalAdd(x, y)

    def add_inplace(self, x, y):
        self.cc.EvalAddInPlace(x, y)
        return x

    def subtract(self, x, y):
        if isinstance(x, (int, float)):
            return self.cc.EvalScalarSub(float(x), y)
        if isinstance(x, np.ndarray):
            pt = self.encode(x, level=self.depth - self.level(y))
            return self.cc.EvalNegate(self.cc.EvalSubPt(y, pt))
        if isinstance(y, (int, float)):
            return self.cc.EvalSubScalar(x, float(y))
        if isinstance(y, np.ndarray):
            pt = self.encode(y, level=self.depth - self.level(x))
            return self.cc.EvalSubPt(x, pt)
        if isinstance(y, _core.Plaintext):
            return self.cc.EvalSubPt(x, y)
        return self.cc.EvalSub(x, y)

    def multiply(self, x, y, *, relin: bool = False):
        """ct*ct is lazy (degree-2) unless relin=True; int scalars are level-free; floats consume a level."""
        if isinstance(y, (bool, int)) and not isinstance(y, float):
            return self.cc.EvalMultByInteger(x, int(y))
        if isinstance(y, float):
            return self.cc.EvalMultScalar(x, y)
        if isinstance(y, np.ndarray):
            pt = self.encode(y, level=self.depth - self.level(x))
            return self.cc.EvalMultPt(x, pt)
        if isinstance(y, _core.LightPlaintext):
            return self.cc.EvalMultLightPt(x, y)
        if isinstance(y, _core.Plaintext):
            return self.cc.EvalMultPt(x, y)
        return self.cc.EvalMult(x, y) if relin else self.cc.EvalMultNoRelin(x, y)

    def square(self, x, *, relin: bool = False):
        return self.cc.EvalSquare(x) if relin else self.cc.EvalSquareNoRelin(x)

    def relinearize(self, x):
        return self.cc.EvalRelinearize(x)

    def multiply_1j(self, x):
        return self.cc.EvalMultByI(x)

    def conjugate(self, x):
        return self.cc.EvalConjugate(x)

    def rotate(self, x, delta: int):
        if int(delta) == 0:
            return x
        return self.cc.EvalRotate(x, int(delta))

    def rescale(self, x):
        return self.cc.Rescale(x)

    def level_down(self, x, by: int):
        return self.cc.EvalLevelReduce(x, int(by))

    def level(self, x) -> int:
        """Remaining multiplicative levels (desilofhe `.level`)."""
        return int(self.cc.GetRemainingLevels(x))

    def noise_level(self, x) -> int:
        """Scale degree: 1 when canonical, 2 between a multiplication and its rescale.

        The other half of a ciphertext's FIXEDMANUAL state, and the half that used to be
        unreadable - so whether an operation was scale-neutral had to be inferred from a wrong
        answer several stages later rather than asked.
        """
        return int(self.cc.GetNoiseLevel(x))

    def bootstrap(self, x, keep_levels: int | None = None):
        """Refresh ``x``. ``keep_levels`` drops the result to exactly that level.

        Asking for more than the bootstrap produced is an error rather than a quiet shortfall. It
        used to return whatever it got, and the deficit then surfaced several stages later as
        ``EvalLevelReduce would drop every RNS limb`` - a message about the wrong operation, in the
        wrong place. How many levels a bootstrap leaves depends on the level budget, the secret key
        distribution and (see ``EvalCoeffsToSlots``) the level its precomputed diagonals were encoded
        at, so it is not something a caller can assume.
        """
        if self.scaling_technique == _core.FIXEDMANUAL:
            entering = self.noise_level(x)
            if entering != 1:
                raise ValueError(
                    f"bootstrap wants a canonical ciphertext, got scale degree {entering}. Rescale "
                    f"first: how many levels the bootstrap costs depends on what it is handed, and "
                    f"the level plan assumes a fixed cost.")
        out = self.cc.EvalBootstrap(x)
        # The GPU bootstrap is supposed to return a canonical ciphertext - `approxModReduction`
        # rescales once under FIXEDMANUAL for exactly that reason (ApproxModEval.cu) - but it comes
        # back at scale degree 2, so it is not meeting its own contract. Under FIXEDMANUAL nothing
        # corrects that later: the degree *adds* on every multiply and only comes down on rescale, so
        # starting at 2 reaches 12108 by the end of a softmax and the first addPt fails on mismatched
        # scales. Before that it is silent, and he_exp simply returns 1e124.
        #
        # A loop rather than a single rescale, because the number to correct is the one measured, not
        # the one assumed: `noise_level` reads the field, so this costs exactly as many levels as the
        # ciphertext is actually off by - and on the CPU path, where OpenFHE's own bootstrap returns
        # a canonical ciphertext, it costs none. Remove it once the device meets the contract; the
        # level it spends is not free (see `thorfhe.budget`: it is the difference between depth 37
        # fitting and not).
        if self.scaling_technique == _core.FIXEDMANUAL:
            for _ in range(8):   # bounded: a runaway would eat the level budget without saying so
                if self.noise_level(out) <= 1:
                    break
                out = self.cc.Rescale(out)
            remaining = self.noise_level(out)
            if remaining != 1:
                raise RuntimeError(
                    f"bootstrap returned scale degree {remaining} and rescaling did not bring it to "
                    f"1. Every multiplication after this point compounds it, and the failure "
                    f"surfaces as a scale mismatch several stages away.")
        got = self.level(out)
        if self.bootstrap_level is not None and got != self.bootstrap_level:
            raise ValueError(
                f"bootstrap restored level {got}, but the level plan was built assuming "
                f"{self.bootstrap_level}. Every rotation key was truncated to the levels that plan "
                f"predicted, so continuing would spend keys that do not reach. Re-measure what "
                f"EvalBootstrap plus the FIXEDMANUAL rescale actually cost and put it in "
                f"thorfhe.budget.MEASURED_BOOTSTRAP, or pass --bootstrap-depth "
                f"{self.depth - got}.")
        if keep_levels is not None:
            if got < keep_levels:
                raise ValueError(
                    f"bootstrap left level {got}, but keep_levels={keep_levels} asked for more. "
                    f"Raise the depth, or lower what the circuit expects after a bootstrap.")
            if got > keep_levels:
                out = self.cc.EvalLevelReduce(out, got - keep_levels)
        return out

    def bootstrap_stage(self, x, stage: int):
        """Run a bootstrap up to ``stage`` (1..4) and stop, for measuring where accuracy is lost.

        1 is ModRaise and the constant scale, 2 CoeffsToSlots, 3 the modular reduction, 4
        SlotsToCoeffs. The result is **not** a usable refreshed ciphertext - its level, scale and
        slot layout are mid-flight - so none of `bootstrap`'s scale correction or level checking
        applies here, and neither is done.

        A bootstrap is four steps and from outside it is one, which is why its accuracy has nowhere
        to be pinned: refreshing zeros comes back with a deterministic, message-independent error of
        about 0.015 and no stage owns it. Every stage of a bootstrap of zeros should decrypt to
        roughly zero; the first that does not is where it is made.
        """
        if not 1 <= int(stage) <= 4:
            raise ValueError(f"stage must be 1..4, got {stage}; use bootstrap() to run all of it")
        return self.cc.EvalBootstrap(x, stopAfterStage=int(stage))

    # ---- light plaintexts (THOR weight storage, docs/light_plaintext.md) ----
    def encode_to_light_plaintext(self, msg, level: int | None = None):
        """Compact, level-agnostic encoding: N int64 coefficients instead of the (L+1) RNS towers.

        `level` is THOR's remaining-level convention (the level the weight is used at) and is only a
        recorded hint under FIXEDMANUAL; pass it so a FLEXIBLE* context can reject a wrong expansion.
        """
        arr = np.asarray(msg)
        if arr.ndim != 1:
            arr = arr.reshape(-1)
        hint = -1 if level is None else self.depth - int(level)
        return self.cc.MakeLightPlaintext(arr, self.slots, hint)

    def write_light_plaintext(self, light, path: str):
        light.save(str(path))

    def read_light_plaintext(self, path: str):
        return _core.LightPlaintext.load(str(path))

    def expand_light_plaintext(self, light, level: int):
        """Materialise `light` as a normal plaintext at `level` remaining levels."""
        return self.cc.ExpandLightPlaintext(light, self.depth - int(level))

    def trim_auxiliary_polys(self, keep: int = 0):
        """Drain the pooled auxiliary polynomials, keeping at most ``keep``.

        A destroyed ciphertext does not free its polynomials - it parks them in the context's pool for
        the next one to reuse - and nothing drains that pool, so it settles at the high-water mark of
        live ciphertexts and stays there. This is the step that lets memory actually leave: the
        discarded polynomials free their limbs to the device pool's free lists, and a slab whose blocks
        are all free can then go back to the driver. Without it, slab reclamation finds nothing to
        return. Worth calling at a stage boundary, where the working set really is smaller.
        """
        self.cc.TrimAuxiliaryPolys(int(keep))

    def auxiliary_poly_count(self) -> int:
        return int(self.cc.GetAuxiliaryPolyCount())

    def device_memory(self) -> dict:
        """Device pool bytes: ``pooled``, ``in_use``, ``driver_free``, ``driver_total``.

        Empty on a CPU-only context, so a caller can print it unconditionally. ``pooled - in_use`` is
        memory the pool holds but nobody is using: if that is large when an allocation fails, the
        circuit's working set is not what ran the card out and reclamation is what to look at.
        """
        pooled, in_use, driver_free, driver_total = self.cc.GetDeviceMemory()
        if not driver_total:
            return {}
        return {"pooled": pooled, "in_use": in_use,
                "driver_free": driver_free, "driver_total": driver_total}

    def clear_light_plaintext_cache(self):
        self.cc.ClearLightPlaintextCache()

    # ntt/intt are identity: fideslib keeps ciphertexts in the NTT domain.
    def ntt(self, x):
        return x

    def intt(self, x):
        return x
