"""RQ2 2x2 contrasts with a joint paired bootstrap over episode keys (task, ep).

Cells: p00 = SRC (Panda visible), p01 = APP-strong (Panda hidden), p10 = arm visible,
p11 = arm hidden. All four cells are resampled together on the keys present in all of them.
Usage: python contrasts.py [--runs $PROJ_ROOT/runs] [--src-log PATH]
"""
import argparse
import os

import numpy as np

from summarize import load_dumps, load_log


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
    ap.add_argument("--src-log", default=None)
    ap.add_argument("--B", type=int, default=10000)
    a = ap.parse_args()
    src = load_log(a.src_log) if a.src_log else {}
    src.update(load_dumps(os.path.join(a.runs, "src")))
    apps = load_dumps(os.path.join(a.runs, "apps"))
    rng = np.random.default_rng(0)
    for arm in ["iiwa", "ur5e"]:
        p10, p11 = load_dumps(os.path.join(a.runs, arm)), load_dumps(os.path.join(a.runs, arm + "hid"))
        keys = sorted(set(src) & set(apps) & set(p10) & set(p11))
        if not keys:
            print(arm, "no common keys")
            continue
        X = np.array([[src[k][1], apps[k][1], p10[k][1], p11[k][1]] for k in keys], float)

        def stats(M):
            m = M.mean(0)
            return {
                "hide on Panda (p01-p00)": m[1] - m[0],
                "hide on arm (p11-p10)": m[3] - m[2],
                "arm swap, visible (p10-p00)": m[2] - m[0],
                "mean arm effect": ((m[2] - m[0]) + (m[3] - m[1])) / 2,
                "mean hide effect": ((m[1] - m[0]) + (m[3] - m[2])) / 2,
                "interaction": (m[3] - m[2]) - (m[1] - m[0]),
            }

        point = stats(X)
        boots = {k: [] for k in point}
        for _ in range(a.B):
            s = stats(X[rng.integers(0, len(keys), len(keys))])
            for k, v in s.items():
                boots[k].append(v)
        cells = " ".join(f"{100 * v:.1f}" for v in X.mean(0))
        print(f"\n== {arm.upper()}  n={len(keys)} common keys  cells p00 p01 p10 p11 = {cells} %")
        for k, v in point.items():
            lo, hi = np.percentile(boots[k], [2.5, 97.5])
            print(f"  {k:30s} {100 * v:+6.1f} pp  [{100 * lo:+6.1f}, {100 * hi:+6.1f}]")


if __name__ == "__main__":
    main()
