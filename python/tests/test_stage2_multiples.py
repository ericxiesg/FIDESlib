"""Quick check: does the 0/9891/19782 small-slot pattern extend to other multiples of 9891?"""
import os, numpy as np, pytest

pf = pytest.importorskip("pyfideslib")
BENCH = pytest.mark.skipif(not os.environ.get("PYFIDESLIB_BENCH_PARAMS"), reason="set PYFIDESLIB_BENCH_PARAMS=1")

def _engine(device, **ov):
    from test_bootstrap_noise_level import bench_params
    p = dict(bench_params()); p.update(ov)
    return pf.Engine(device, **p)

@BENCH
def test_stage2_multiples_of_9891(device):
    engine = _engine(device)
    b0 = np.random.default_rng(23).uniform(0.0814, 0.9812, engine.slots)
    out = engine.bootstrap_stage(engine.encrypt(b0), 2)
    values = np.abs(np.real(np.asarray(engine.decrypt(out)))[:b0.size])
    median = np.median(values)

    print(f"\n[multiples] median {median:.6g}, checking multiples of 9891:")
    for k in range(8):
        s = (k * 9891) % values.size
        r = values[s] / max(median, 1e-30)
        label = "ANOMALOUS" if r < 0.1 else ("small" if r < 0.5 else "normal")
        print(f"  {k}×9891 mod {values.size} = slot {s:6d}: |val| {values[s]:12.6g}  ratio {r:8.3f}  {label}")

    # Also check 20170 and nearby
    print(f"\n  20170 region:")
    for s in [20170, 20170-388, 20170+388, 10279, 388, 29673, 3095]:
        if 0 <= s < values.size:
            r = values[s] / max(median, 1e-30)
            print(f"  slot {s:6d}: |val| {values[s]:12.6g}  ratio {r:8.3f}")

    # Check if the ~20 zero slots share a pattern
    zeros = np.where(values < 1e-6)[0]
    print(f"\n  {zeros.size} slots below 1e-6:")
    if zeros.size > 0:
        print(f"  indices: {sorted(zeros.tolist())[:50]}")
        # Check if any are multiples of 9891
        hits = [z for z in zeros if z % 9891 == 0]
        print(f"  multiples of 9891 among zeros: {hits}")
        # Check gaps
        if zeros.size > 1:
            zs = sorted(zeros.tolist())
            gaps = [zs[i+1]-zs[i] for i in range(len(zs)-1)]
            from collections import Counter
            print(f"  gap distribution: {Counter(gaps).most_common(5)}")
