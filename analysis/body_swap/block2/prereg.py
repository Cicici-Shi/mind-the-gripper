"""Pre-registered tests (proposal §analysis):
  P1  APP-weak (recolored arm, run 'rec'): d = SR(rec) - SR(SRC), paired over (task, ep).
      90% CI inside +-10 pp -> equivalent; 95% CI entirely outside +-10 pp -> different; else inconclusive.
      Also reported for APP-strong (Panda arm hidden, 'apps') as exploratory.
  P2  per arm: SR(GRP) - SR(ARM), 95% CI lower bound > 0 -> gripper swap hurts less than arm swap; else not separated.
  Interactions (IIWA, UR5e) = (p11 - p10) - (p01 - p00): one family, Holm-adjusted two-sided bootstrap p.
Paired cluster bootstrap over (task, ep) keys, all cells resampled jointly, B = 10000.

Usage: python prereg.py [RUNS_DIR]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load_dumps  # noqa: E402

B, MARGIN = 10000, 10.0


def cells(runs, names):
    d = {n: load_dumps(os.path.join(runs, n)) for n in names}
    keys = sorted(set.intersection(*(set(v) for v in d.values())))
    return keys, {n: np.array([d[n][k][1] for k in keys], float) for n in names}


def boot(keys, f, seed=0):
    rng = np.random.default_rng(seed)
    n = len(keys)
    return np.array([f(rng.integers(0, n, n)) for _ in range(B)])


def main(runs):
    print("== P1 equivalence (margin +-10 pp)")
    for cond, label in [("rec", "APP-weak recolor (P1)"), ("apps", "APP-strong hidden (exploratory)")]:
        keys, c = cells(runs, ["src", cond])
        diff = c[cond] - c["src"]
        est = 100 * diff.mean()
        bs = 100 * boot(keys, lambda i: diff[i].mean())
        lo90, hi90 = np.percentile(bs, [5, 95])
        lo95, hi95 = np.percentile(bs, [2.5, 97.5])
        if -MARGIN < lo90 and hi90 < MARGIN:
            verdict = "EQUIVALENT within +-10 pp"
        elif hi95 < -MARGIN or lo95 > MARGIN:
            verdict = "DIFFERENT beyond 10 pp"
        else:
            verdict = "INCONCLUSIVE"
        print(f"{label:34s} n={len(keys):3d} d={est:+5.1f} pp  90% CI [{lo90:+5.1f}, {hi90:+5.1f}]  95% CI [{lo95:+5.1f}, {hi95:+5.1f}]  -> {verdict}")

    print("\n== P2 gripper vs arm: SR(GRP) - SR(ARM)")
    for arm in ["iiwa", "ur5e"]:
        keys, c = cells(runs, ["grp", arm])
        diff = c["grp"] - c[arm]
        bs = 100 * boot(keys, lambda i: diff[i].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "gripper hurts LESS than arm" if lo > 0 else ("not separated (gripper hurts MORE: upper bound < 0)" if hi < 0 else "not separated")
        print(f"GRP - {arm.upper():5s} n={len(keys):3d} {100*diff.mean():+6.1f} pp  95% CI [{lo:+6.1f}, {hi:+6.1f}]  -> {verdict}")

    print("\n== Interaction family (Holm, two-sided bootstrap p)")
    res = []
    for arm in ["iiwa", "ur5e"]:
        keys, c = cells(runs, ["src", "apps", arm, arm + "hid"])
        inter = (c[arm + "hid"] - c[arm]) - (c["apps"] - c["src"])
        bs = 100 * boot(keys, lambda i: inter[i].mean(), seed=1)
        p = min(1.0, 2 * min((bs <= 0).mean(), (bs >= 0).mean()))
        res.append((arm, len(keys), 100 * inter.mean(), *np.percentile(bs, [2.5, 97.5]), p))
    order = sorted(range(len(res)), key=lambda j: res[j][-1])
    adj, running = {}, 0.0
    for rank, j in enumerate(order):
        running = max(running, min(1.0, (len(res) - rank) * res[j][-1]))
        adj[j] = running
    for j, (arm, n, est, lo, hi, p) in enumerate(res):
        print(f"{arm.upper():5s} n={n:3d} interaction {est:+5.1f} pp  95% CI [{lo:+5.1f}, {hi:+5.1f}]  p={p:.4f}  Holm p={adj[j]:.4f}  -> {'significant' if adj[j] < 0.05 else 'not significant'} at 0.05")


if __name__ == "__main__":
    main(os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "$PROJ_ROOT/runs"))
