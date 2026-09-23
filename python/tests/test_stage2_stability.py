"""Run stage 2 repeatedly on the same input and diff per slot.

No CPU baseline, no known scale - the collaborator pointed out both are invalid
at stage 2. What is valid is which slots change across repeats on the same input.
A slot that is deterministic but near zero is not the bug; a slot that jumps
around is where the instability lives.
"""
import os, numpy as np, pytest

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"), reason="set PYFIDESLIB_BENCH_PARAMS=1")

def _engine(device, **ov):
    from test_bootstrap_noise_level import bench_params
    p = dict(bench_params()); p.update(ov)
    return pf.Engine(device, **p)

@BENCH
def test_stage2_stability(device):
    engine = _engine(device)
    b0 = np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)

    N_RUNS = 6
    runs = []
    for i in range(N_RUNS):
        ct = engine.encrypt(b0)
        out = engine.bootstrap_stage(ct, 2)
        vals = np.real(np.asarray(engine.decrypt(out)))[:b0.size]
        runs.append(vals)
        print(f"\n[stability] run {i}: slot 0 = {vals[0]:.6g}  median = {np.median(np.abs(vals)):.6g}")

    runs = np.array(runs)

    # Per-slot spread across runs
    spread = np.std(runs, axis=0)
    mean_abs = np.mean(np.abs(runs), axis=0)

    # Coefficient of variation: spread relative to magnitude
    cv = spread / (np.abs(mean_abs) + 1e-30)

    print(f"\n[stability] per-slot stability across {N_RUNS} runs:")
    print(f"    median |val|: {np.median(np.abs(mean_abs)):.6g}")
    print(f"    median spread: {np.median(spread):.6g}")
    print(f"    median CV: {np.median(cv):.6g}")

    # Top 20 most unstable slots by absolute spread
    order_spread = np.argsort(-spread)
    print(f"\n    top 20 by absolute spread:")
    for rank, idx in enumerate(order_spread[:20]):
        vals_str = " ".join(f"{runs[r][idx]:.4g}" for r in range(N_RUNS))
        print(f"      {rank+1:3d}. slot {idx:6d}: spread {spread[idx]:.4g}  |mean| {abs(mean_abs[idx]):.4g}  CV {cv[idx]:.4g}  vals=[{vals_str}]")

    # Top 20 by CV (relative instability)
    order_cv = np.argsort(-cv)
    print(f"\n    top 20 by coefficient of variation:")
    for rank, idx in enumerate(order_cv[:20]):
        vals_str = " ".join(f"{runs[r][idx]:.4g}" for r in range(N_RUNS))
        print(f"      {rank+1:3d}. slot {idx:6d}: CV {cv[idx]:.4g}  spread {spread[idx]:.4g}  |mean| {abs(mean_abs[idx]):.4g}  vals=[{vals_str}]")

    # Known slots
    known = [0, 9891, 19782, 20170, 16687, 29673]
    print(f"\n    known slots:")
    for s in known:
        if s < runs.shape[1]:
            vals_str = " ".join(f"{runs[r][s]:.6g}" for r in range(N_RUNS))
            print(f"      slot {s:6d}: spread {spread[s]:.4g}  CV {cv[s]:.4g}  vals=[{vals_str}]")

    # How many slots have CV > 1 (more variable than their mean)?
    unstable = np.where(cv > 1.0)[0]
    print(f"\n    {unstable.size} slots with CV > 1.0")
    if 0 < unstable.size <= 50:
        print(f"    indices: {sorted(unstable.tolist())}")

    # How many slots have spread > median spread * 10?
    threshold = np.median(spread) * 10
    jumpy = np.where(spread > threshold)[0]
    print(f"    {jumpy.size} slots with spread > 10x median spread ({threshold:.4g})")
    if 0 < jumpy.size <= 50:
        print(f"    indices: {sorted(jumpy.tolist())}")
