"""Check stability of each bootstrap stage independently.

The stage-2 non-determinism finding raises the question: is it stage 2 specifically,
or does it start earlier? Run each stage 4 times and check per-slot stability.
"""
import os, numpy as np, pytest

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"), reason="set PYFIDESLIB_BENCH_PARAMS=1")

def _engine(device, **ov):
    from test_bootstrap_noise_level import bench_params
    p = dict(bench_params()); p.update(ov)
    return pf.Engine(device, **p)

@BENCH
def test_each_stage_stability(device):
    engine = _engine(device)
    b0 = np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)
    N_RUNS = 4

    for stage, name in enumerate(("ModRaise", "CoeffsToSlots", "EvalMod", "SlotsToCoeffs"), start=1):
        runs = []
        for i in range(N_RUNS):
            ct = engine.encrypt(b0)
            out = engine.bootstrap_stage(ct, stage)
            vals = np.real(np.asarray(engine.decrypt(out)))[:b0.size]
            runs.append(vals)

        runs = np.array(runs)
        spread = np.std(runs, axis=0)
        mean_abs = np.mean(np.abs(runs), axis=0)
        cv = spread / (np.abs(mean_abs) + 1e-30)
        unstable = np.sum(cv > 0.1)

        slot0_vals = " ".join(f"{runs[r][0]:.6g}" for r in range(N_RUNS))
        print(f"\n[stage {stage} {name:13s}] unstable(>0.1 CV): {unstable:5d}/{runs.shape[1]}  "
              f"median spread {np.median(spread):.4g}  slot0=[{slot0_vals}]")
