"""Paired success-rate summary across conditions.

Reads per-episode dumps ($PROJ_ROOT/runs/<cond>/t{task}_e{ep}_i{init}.npz) and, for the clean
September SRC run, its eval log (INIT_ID + Success lines). Episodes are keyed by
(task, ep); the same key means the same init state and, with the fixed pipeline, the same
reference scene. Paired difference vs SRC with a cluster bootstrap over keys.

Usage: python summarize.py [--runs $PROJ_ROOT/runs] [--src-log PATH]
"""
import argparse
import glob
import os
import re

import numpy as np


def load_dumps(d):
    out = {}
    for f in glob.glob(os.path.join(d, "t*_e*_i*.npz")):
        z = np.load(f, allow_pickle=True)
        out[(int(z["task"]), int(z["ep"]))] = (int(z["init"]), bool(z["success"]))
    return out


def load_log(path):
    out, cur = {}, None
    for line in open(path, errors="ignore"):
        m = re.search(r"INIT_ID: task=(\d+) episode=(\d+) init=(\d+)", line)
        if m:
            cur = (int(m.group(1)), int(m.group(2)))
            out[cur] = [int(m.group(3)), None]
            continue
        m3 = re.search(r"Success: (True|False)", line)
        if m3 and cur:
            out.setdefault(cur, [None, None])[1] = m3.group(1) == "True"
            cur = None
    return {k: tuple(v) for k, v in out.items() if v[1] is not None}


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"),) * 2
    p = k / n
    c = (p + z * z / (2 * n)) / (1 + z * z / n)
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return c - h, c + h


def paired(a, b, B=10000, seed=0):
    keys = sorted(set(a) & set(b))
    if not keys:
        return None
    x = np.array([a[k][1] for k in keys], float)
    y = np.array([b[k][1] for k in keys], float)
    d = x - y
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(keys), size=(B, len(keys)))
    boots = d[idx].mean(1)
    return dict(n=len(keys), diff=d.mean(), lo=np.percentile(boots, 2.5), hi=np.percentile(boots, 97.5),
                a_only=int(((x == 1) & (y == 0)).sum()), b_only=int(((x == 0) & (y == 1)).sum()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
    ap.add_argument("--src-log", default=None)
    a = ap.parse_args()
    conds = {os.path.basename(p): load_dumps(p) for p in sorted(glob.glob(os.path.join(a.runs, "*"))) if os.path.isdir(p)}
    if a.src_log:
        conds["src_sep"] = load_log(a.src_log)
    src = dict(conds.get("src_sep", {}))
    src.update(conds.get("src", {}))  # new SRC episodes override/extend
    conds["SRC(all)"] = src
    print(f"{'condition':12s} {'n':>4s} {'succ':>5s} {'SR%':>6s} {'95% Wilson':>14s} | vs SRC paired: n  diff(pp) [95% boot]  cond-only/src-only")
    for name, c in conds.items():
        k, n = sum(v[1] for v in c.values()), len(c)
        lo, hi = wilson(k, n)
        line = f"{name:12s} {n:4d} {k:5d} {100*k/max(n,1):6.1f} [{100*lo:5.1f},{100*hi:5.1f}]"
        if name not in ("SRC(all)", "src", "src_sep"):
            p = paired(c, src)
            if p:
                line += f" | {p['n']:3d} {100*p['diff']:+6.1f} [{100*p['lo']:+5.1f},{100*p['hi']:+5.1f}]  {p['a_only']}/{p['b_only']}"
        print(line)
    # init-id consistency check: same key must mean same init state
    for name, c in conds.items():
        bad = [k for k in c if k in src and c[k][0] is not None and src[k][0] is not None and c[k][0] != src[k][0]]
        if bad:
            print(f"WARNING {name}: init mismatch on {len(bad)} keys, e.g. {bad[:3]}")


if __name__ == "__main__":
    main()
