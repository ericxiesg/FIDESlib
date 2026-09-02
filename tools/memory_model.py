#!/usr/bin/env python3
"""Device-memory model for FIDESlib CKKS bootstrapping (64-bit limbs).

Usage: memory_model.py --logN 16 --limbs 30 --dnum 3 --slots 32768 --lb 3 3 [--after-boot 10]

Reproduces the formulas in docs/level_truncated_keys.md and estimates the key/plaintext residency with
and without level truncation. Key counts follow OpenFHE's BSGS parametrisation
(numRotations = 2^(layersCollapse+1)-1 per layer, b = ceil(sqrt(numRotations)), g = ceil(numRotations/b)).
"""
import argparse, math

def bsgs_layers(log_slots, lb):
    # OpenFHE: layersCollapse = ceil(log_slots/lb); the last layer collapses the remainder.
    lc = math.ceil(log_slots / lb)
    rem = log_slots - lc * (lb - 1)
    out = []
    for i in range(lb):
        layers = lc if i < lb - 1 else rem
        n = (1 << (layers + 1)) - 1
        b = math.ceil(math.sqrt(n)); g = math.ceil(n / b)
        out.append((n, b, g))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--logN', type=int, default=16)
    ap.add_argument('--limbs', type=int, default=30, help='L+1 (number of Q primes)')
    ap.add_argument('--dnum', type=int, default=3)
    ap.add_argument('--slots', type=int, default=None)
    ap.add_argument('--lb', type=int, nargs=2, default=[3, 3])
    ap.add_argument('--mod-depth', type=int, default=9, help='approx mod-reduction depth (OpenFHE: 8-13 depending on key dist)')
    ap.add_argument('--word', type=int, default=8)
    ap.add_argument('--margin', type=int, default=1)
    a = ap.parse_args()

    N = 1 << a.logN
    slots = a.slots or N // 2
    Lp1 = a.limbs; L = Lp1 - 1
    alpha = math.ceil(Lp1 / a.dnum); K = alpha
    limb = N * a.word

    def key_bytes(m):  # m = max level (-1 -> full)
        if m < 0: m = L
        return 2 * math.ceil((m + 1) / alpha) * (m + 1 + K) * limb

    log_slots = int(math.log2(slots))
    cts = bsgs_layers(log_slots, a.lb[0]); stc = bsgs_layers(log_slots, a.lb[1])
    # FIDESlib AFFINE_LT: per layer keys = rotIn (g) + rotOut[1] (1); + accumulated offset key per transform
    n_cts = sum(g + 1 for (_, _, g) in cts) + 1
    n_stc = sum(g + 1 for (_, _, g) in stc) + 1
    n_acc = int(math.log2((N // 2) // slots)) if slots < N // 2 else 0
    n_keys = n_cts + n_stc + n_acc + 1  # + conjugate
    boot_depth = a.lb[0] + a.lb[1] + a.mod_depth
    stc_top = L - boot_depth + a.lb[1] + a.margin

    full = n_keys * key_bytes(-1)
    trunc = (n_cts + n_acc + 1) * key_bytes(-1) + n_stc * key_bytes(stc_top)
    pt_cts = sum(n for (n, _, _) in cts) * (L) * limb          # ~ at top levels
    pt_stc = sum(n for (n, _, _) in stc) * (stc_top + 1) * limb

    MiB = 1 << 20
    print(f"N=2^{a.logN} L+1={Lp1} dnum={a.dnum} alpha=K={K} limb={limb/MiB:.2f} MiB")
    print(f"complete key: {key_bytes(-1)/MiB:.1f} MiB; StC key truncated to level {stc_top}: {key_bytes(stc_top)/MiB:.1f} MiB")
    print(f"keys: CtS {n_cts}, StC {n_stc}, accumulate {n_acc}, conj 1 -> {n_keys} (index overlap not modelled)")
    print(f"rotation keys complete : {full/MiB/1024:.2f} GiB")
    print(f"rotation keys truncated: {trunc/MiB/1024:.2f} GiB  (-{100*(1-trunc/full):.0f} %)")
    print(f"CtS plaintexts ~{pt_cts/MiB/1024:.2f} GiB, StC plaintexts ~{pt_stc/MiB/1024:.2f} GiB (upper bounds)")

if __name__ == '__main__':
    main()
