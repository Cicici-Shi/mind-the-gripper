"""P11 (collision counterfactual) re-reported on the FIXED 20 task-3 paired keys.

Why (Astra, astra_block3_interpretation.md, last paragraph): the frozen prereg_block3_a2.py bootstraps the count
Delta3 = sum_{task-3 keys}(succ(UR5e_nc) - succ(UR5e_cg)) with one draw matrix over ALL 200 (task, episode) keys
(口径 K1/K4). In a resample the number of drawn task-3 keys is Binomial(200, 1/10), not 20, so the count can exceed
20 (upper bound +27 "rescued episodes" out of 20). That interval is valid for its own null test but cannot be shown
as an interval for "how many of the 20 task-3 episodes are rescued".

This script does NOT modify the frozen script. It imports its loaders (load_oft, load_pi05, Boot, mcnemar_exact)
read-only, rebuilds the two P11 cells per model, reproduces the frozen Delta3 bootstrap (sanity check), shows the
distribution of drawn task-3 keys, and then reports on the fixed 20 paired keys:

  M0  counts: UR5e_cg x/20, UR5e_nc y/20, the paired 2x2 (a = both succeed, b = nc only, c = cg only, d = neither),
      Clopper-Pearson 95% for each cell's SR.
  M1  exact McNemar (conditional binomial on the b + c discordant pairs), two-sided p, same as the frozen K5;
      plus the Clopper-Pearson 95% interval for b / (b + c) (share of discordant pairs that favour nc).
  M2  PRIMARY. Exact, conservative 95% interval for the paired difference Delta = (b - c) / 20:
      simultaneous Bonferroni Clopper-Pearson, i.e. 97.5% CP intervals for pi_b = P(nc only) and pi_c = P(cg only),
      Delta in [L_b - U_c, U_b - L_c], clipped to [-1, 1]. Coverage >= 95% by construction (no asymptotics).
      Reported as pp and as episodes out of 20 (x 20).
  M3  Newcombe (1998) method 10: hybrid Wilson-score interval for a paired difference of proportions
      (phi set to 0 when a margin is empty). Approximate but well calibrated near 0/1.
  M4  within-task bootstrap: resample the 20 task-3 keys with replacement (n fixed at 20), B = 10000, seed 0,
      percentile. Shown for comparison only: when c = 0 it is degenerate/anti-conservative at the boundary.

Usage (cluster, login node or CPU partition; pure numpy, no MuJoCo, no GPU):
  python p11_fixed20_ci.py --frozen $PROJ_ROOT/embody_v3/analysis/prereg_block3_a2.py > results_p11_fixed20.txt
"""
import argparse
import hashlib
import importlib.util
import math
import os

import numpy as np

H = os.path.expanduser
N3 = 20


def load_frozen(path):
    spec = importlib.util.spec_from_file_location("prereg_block3_a2_frozen", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # module body only defines things; main() is guarded by __name__
    return mod


# ---------------------------------------------------------------- exact binomial helpers (no scipy)
def binom_cdf(k, n, p):
    if k < 0:
        return 0.0
    if k >= n:
        return 1.0
    return sum(math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(k + 1))


def _bisect(f, lo, hi, it=200):
    for _ in range(it):
        mid = (lo + hi) / 2
        if f(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def clopper_pearson(x, n, conf=0.95):
    a = 1 - conf
    lo = 0.0 if x == 0 else _bisect(lambda p: 1 - binom_cdf(x - 1, n, p) >= a / 2, 0.0, 1.0)
    hi = 1.0 if x == n else _bisect(lambda p: binom_cdf(x, n, p) <= a / 2, 0.0, 1.0)
    return lo, hi


def wilson(x, n, z=1.959963984540054):
    p = x / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, c - h), min(1.0, c + h)


def newcombe10(a, b, c, d):
    n = a + b + c + d
    p1, p2 = (a + b) / n, (a + c) / n  # p1 = SR(nc), p2 = SR(cg)
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = (a + b) * (c + d) * (a + c) * (b + d)
    phi = (a * d - b * c) / math.sqrt(den) if den > 0 else 0.0
    th = p1 - p2
    lo = th - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = th + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return max(-1.0, lo), min(1.0, hi), phi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen", default="$PROJ_ROOT/embody_v3/analysis/prereg_block3_a2.py")
    ap.add_argument("--runs", default="$PROJ_ROOT/runs")
    ap.add_argument("--pi05", default="$PROJ_ROOT/pi05/v2")
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    fz = H(a.frozen)
    F = load_frozen(fz)
    sha = hashlib.sha256(open(fz, "rb").read()).hexdigest()
    pr = print
    pr("p11_fixed20_ci.py: P11 on the fixed 20 task-3 paired keys (UR5e_nc vs UR5e_cg); frozen script imported read-only")
    pr(f"frozen loaders: {fz}  sha256 {sha}")
    pr(f"runs={a.runs} pi05={a.pi05}  B={a.B}  seed={F.SEED}  n=200 design keys (tasks 0-9 x episodes 0-19)")
    pr("cells: UR5e_cg = OFT runs/ur5e_cg, pi0.5 v2/ur5e_cg_e*.jsonl; UR5e_nc = runs/ur5e_nc, v2/ur5e_nc_e*.jsonl")

    neps = 20
    keys = [(t, e) for t in range(10) for e in range(neps)]
    rows = []
    for model in ("OFT", "pi0.5"):
        data = {}
        for cell, d in (("UR5e_cg", "ur5e_cg"), ("UR5e_nc", "ur5e_nc")):
            if model == "OFT":
                data[cell] = F.load_oft(os.path.join(H(a.runs), d), neps, False, False, None)
            else:
                data[cell] = F.load_pi05(H(a.pi05), d, neps)[0]
        Y = {c: np.array([rec[k]["succ"] if k in rec else np.nan for k in keys]) for c, rec in data.items()}
        pr(f"\n======== {model}")
        pr(f"  keys loaded: UR5e_cg {len(data['UR5e_cg'])}/200, UR5e_nc {len(data['UR5e_nc'])}/200")

        # (A) reproduce the frozen estimand and show why its upper bound exceeds 20
        boot = F.Boot(keys, a.B)
        terms, filt, kind = F.SPATIAL_EST["Delta3"]
        r = boot.est(Y, terms, filt, kind)
        t3 = np.array([k[0] == 3 for k in keys])
        m = t3 & ~np.isnan(Y["UR5e_cg"]) & ~np.isnan(Y["UR5e_nc"])
        n_drawn = boot.W @ m.astype(float)
        q = np.percentile(n_drawn, [2.5, 50, 97.5])
        pr("  (A) frozen estimand reproduced (prereg_block3_a2.Boot over all 200 keys, count kind):")
        pr(f"      Delta3 = {r['point']:+.0f}  bootstrap 95% [{r['ci95'][0]:+.1f}, {r['ci95'][1]:+.1f}]  n={r['n']}")
        pr(f"      drawn task-3 keys per resample: median {q[1]:.0f}, 2.5-97.5% [{q[0]:.0f}, {q[2]:.0f}],"
           f" min {n_drawn.min():.0f}, max {n_drawn.max():.0f}; P(drawn > 20) = {(n_drawn > N3).mean():.3f}")
        pr("      -> the interval mixes the task-3 sample size with the rescue effect; not an interval for x/20.")

        # (B) fixed 20 paired keys
        k3 = sorted(k for k in keys if k[0] == 3 and k in data["UR5e_cg"] and k in data["UR5e_nc"])
        n = len(k3)
        cg = np.array([data["UR5e_cg"][k]["succ"] for k in k3])
        nc = np.array([data["UR5e_nc"][k]["succ"] for k in k3])
        A = int(((nc == 1) & (cg == 1)).sum())
        Bv = int(((nc == 1) & (cg == 0)).sum())
        C = int(((nc == 0) & (cg == 1)).sum())
        D = int(((nc == 0) & (cg == 0)).sum())
        delta = Bv - C
        pr(f"  (B) fixed task-3 paired keys: n = {n}" + ("" if n == N3 else "  WARNING n != 20"))
        cpc, cpn = clopper_pearson(int(cg.sum()), n), clopper_pearson(int(nc.sum()), n)
        pr(f"  M0  UR5e_cg {int(cg.sum())}/{n} (SR {100 * cg.mean():.0f}%, CP95 [{100 * cpc[0]:.1f}, {100 * cpc[1]:.1f}])"
           f"   UR5e_nc {int(nc.sum())}/{n} (SR {100 * nc.mean():.0f}%, CP95 [{100 * cpn[0]:.1f}, {100 * cpn[1]:.1f}])")
        pr(f"      paired 2x2: a both succeed {A}, b nc only (rescued) {Bv}, c cg only (lost) {C}, d neither {D}")
        pr(f"      Delta (fixed 20) = b - c = {delta:+d} episodes = {100 * delta / n:+.0f} pp")
        pm = F.mcnemar_exact(Bv, C)
        cpd = clopper_pearson(Bv, Bv + C) if Bv + C else (float("nan"), float("nan"))
        pr(f"  M1  exact McNemar two-sided p = {pm:.4g} (discordant {Bv} for nc / {C} for cg);"
           f" b/(b+c) = {Bv}/{Bv + C}, CP95 [{100 * cpd[0]:.1f}, {100 * cpd[1]:.1f}]%")
        lb, ub = clopper_pearson(Bv, n, 0.975)
        lc, uc = clopper_pearson(C, n, 0.975)
        e_lo, e_hi = max(-1.0, lb - uc), min(1.0, ub - lc)
        pr(f"  M2  PRIMARY exact conservative 95% (Bonferroni CP, 97.5% each): pi_b [{lb:.4f}, {ub:.4f}], pi_c [{lc:.4f}, {uc:.4f}]")
        pr(f"      Delta 95% [{100 * e_lo:+.1f}, {100 * e_hi:+.1f}] pp  = [{n * e_lo:+.1f}, {n * e_hi:+.1f}] of {n} episodes")
        nlo, nhi, phi = newcombe10(A, Bv, C, D)
        pr(f"  M3  Newcombe method 10 (phi = {phi:.3f}): Delta 95% [{100 * nlo:+.1f}, {100 * nhi:+.1f}] pp"
           f" = [{n * nlo:+.1f}, {n * nhi:+.1f}] of {n}")
        rng = np.random.default_rng(F.SEED)
        dd = nc - cg
        idx = rng.integers(0, n, size=(a.B, n))
        bs = dd[idx].sum(1)
        bl, bh = np.percentile(bs, [2.5, 97.5])
        pr(f"  M4  within-task bootstrap (n fixed {n}, B {a.B}, seed {F.SEED}): Delta 95% [{bl:+.1f}, {bh:+.1f}] of {n}"
           f"  (degenerate at the boundary; comparison only)")
        rows.append((model, int(cg.sum()), int(nc.sum()), n, Bv, C, delta, r["ci95"], (n * e_lo, n * e_hi),
                     (n * nlo, n * nhi), (bl, bh), pm))

    pr("\n======== side by side (episodes out of the 20 task-3 keys)")
    pr("  model  cg    nc    b/c    Delta  frozen all-key boot 95%  | exact Bonf-CP 95% (primary)  Newcombe-10 95%   task-3 boot 95%  McNemar p")
    for (mdl, c0, c1, n, b, c, dl, f, e, nw, bt, pm) in rows:
        pr(f"  {mdl:<6} {c0:>2}/{n}  {c1:>2}/{n}  {b}/{c}  {dl:+3d}    [{f[0]:+5.1f}, {f[1]:+5.1f}]           |"
           f" [{e[0]:+5.1f}, {e[1]:+5.1f}]               [{nw[0]:+5.1f}, {nw[1]:+5.1f}]    [{bt[0]:+5.1f}, {bt[1]:+5.1f}]   {pm:.3g}")
    pr("  Reading: report the rescue as x/20 with the exact interval (M2). The P11 verdict line uses the point estimate")
    pr("  (Delta3 >= 10 confirm) and is unaffected; the frozen bootstrap CI/p stay as the pre-registered record, but the")
    pr("  frozen upper bound > 20 is an artefact of the varying task-3 sample size and must not be shown as a count of 20.")


if __name__ == "__main__":
    main()
