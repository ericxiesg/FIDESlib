"""Test stability of encrypt, multScalar, and stage-1-before-Accumulate.

Stage 1 (ModRaise) is 99.97% non-deterministic on GPU. Isolate which component
introduces the randomness:
1. encrypt(b0) — should be deterministic (no nonce in CKKS)
2. stage 1 up to but not including Accumulate — ModRaise proper
"""
import os, numpy as np, pytest

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"), reason="set PYFIDESLIB_BENCH_PARAMS=1")

def _engine(device, **ov):
    from test_bootstrap_noise_level import bench_params
    p = dict(bench_params()); p.update(ov)
    return pf.Engine(device, **p)

@BENCH
def test_encrypt_stability(device):
    """Is encrypt(b0) deterministic on GPU?"""
    engine = _engine(device)
    b0 = np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)
    N_RUNS = 4

    runs = []
    for i in range(N_RUNS):
        ct = engine.encrypt(b0)
        vals = np.real(np.asarray(engine.decrypt(ct)))[:b0.size]
        runs.append(vals)

    runs = np.array(runs)
    spread = np.std(runs, axis=0)
    unstable = np.sum(spread > 1e-6)
    print(f"\n[encrypt] unstable(>1e-6): {unstable}/{runs.shape[1]}  median spread {np.median(spread):.4g}")
    slot0 = " ".join(f"{runs[r][0]:.8g}" for r in range(N_RUNS))
    print(f"    slot0=[{slot0}]")

@BENCH
def test_stage1_vs_encrypt_stability(device):
    """Compare encrypt stability vs stage-1 stability to isolate ModRaise's contribution."""
    engine = _engine(device)
    b0 = np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)
    N_RUNS = 4

    # Encrypt stability
    enc_runs = []
    for i in range(N_RUNS):
        ct = engine.encrypt(b0)
        vals = np.real(np.asarray(engine.decrypt(ct)))[:b0.size]
        enc_runs.append(vals)

    # Stage 1 stability
    s1_runs = []
    for i in range(N_RUNS):
        ct = engine.encrypt(b0)
        out = engine.bootstrap_stage(ct, 1)
        vals = np.real(np.asarray(engine.decrypt(out)))[:b0.size]
        s1_runs.append(vals)

    enc_runs = np.array(enc_runs)
    s1_runs = np.array(s1_runs)
    enc_spread = np.std(enc_runs, axis=0)
    s1_spread = np.std(s1_runs, axis=0)

    enc_unstable = np.sum(enc_spread > 1e-6)
    s1_unstable = np.sum(s1_spread > 1e-6)

    print(f"\n[encrypt]      unstable(>1e-6): {enc_unstable:5d}/{enc_runs.shape[1]}  median spread {np.median(enc_spread):.4g}")
    print(f"[stage 1]      unstable(>1e-6): {s1_unstable:5d}/{s1_runs.shape[1]}  median spread {np.median(s1_spread):.4g}")

    enc_slot0 = " ".join(f"{enc_runs[r][0]:.8g}" for r in range(N_RUNS))
    s1_slot0 = " ".join(f"{s1_runs[r][0]:.8g}" for r in range(N_RUNS))
    print(f"    encrypt slot0=[{enc_slot0}]")
    print(f"    stage1  slot0=[{s1_slot0}]")

    if enc_unstable == 0 and s1_unstable > 0:
        print(f"    -> encrypt is deterministic, ModRaise introduces non-determinism")
    elif enc_unstable > 0:
        print(f"    -> encrypt itself is non-deterministic (unexpected!)")
