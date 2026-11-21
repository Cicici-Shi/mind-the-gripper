"""phasegate readout (EXPLORATION tier: not pre-registered, small n, used only to pick what to confirm).

Reads out/runs/<block>/<mode>_<window>/t*_e*_i*.npz and prints, per mode:
  success rate per window cell, the paired difference to the "none" anchor on the common keys (key = task, init)
  with a task-clustered bootstrap 95% interval, the share of the "all" drop that each single window reproduces,
  and how long each window was open (endogenous window lengths).
Usage: python analysis/pg_readout.py out/runs/<block> [--out file]
"""
import argparse
import glob
import os
import sys

import numpy as np

WINDOWS = ("none", "approach", "grasp", "transport", "all", "post")  # post: phasegate-confirm (rows only if the cell exists)
MODES = ("look", "erase")


def load(cell_dir, mode, window):
    recs = {}
    for f in sorted(glob.glob(os.path.join(cell_dir, "t*_e*_i*.npz"))):
        z = np.load(f, allow_pickle=True)
        if str(z["pg_mode"]) != mode or str(z["pg_window"]) != window:
            raise SystemExit(f"{f}: dump says {z['pg_mode']}/{z['pg_window']}, directory says {mode}/{window}")
        g = np.asarray(z["pg_gate"], bool)
        recs[(int(z["task"]), int(z["init"]))] = {
            "y": bool(z["success"]), "steps": len(g) - 1, "gated": int(g.sum()),
            "t_close": int(z["pg_t_close"]), "t_lift": int(z["pg_t_lift"]),
            }
    return recs


def boot(keys, d, B=5000, seed=0):
    """Task-clustered bootstrap of the mean paired difference (percentile 95%)."""
    tasks = sorted({k[0] for k in keys})
    by = {t: np.asarray([d[k] for k in keys if k[0] == t], float) for t in tasks}
    rng = np.random.default_rng(seed)
    out = np.empty(B)
    for b in range(B):
        pick = rng.choice(len(tasks), len(tasks))
        out[b] = np.concatenate([by[tasks[i]] for i in pick]).mean()
    return float(np.percentile(out, 2.5)), float(np.percentile(out, 97.5))


def report(block_dir):
    L = [f"phasegate readout  block={os.path.basename(os.path.normpath(block_dir))}",
         "EXPLORATION tier: not pre-registered, used only to choose what to confirm on fresh keys.", ""]
    for mode in MODES:
        cells = {w: load(os.path.join(block_dir, f"{mode}_{w}"), mode, w) for w in WINDOWS
                 if os.path.isdir(os.path.join(block_dir, f"{mode}_{w}"))}
        cells = {w: r for w, r in cells.items() if r}
        if not cells:
            continue
        L.append(f"== mode {mode} ==")
        L.append(f"{'window':10s} {'n':>4s} {'SR':>6s} {'vs none (paired, pts)':>30s} {'share of all':>13s} "
                 f"{'gated steps':>12s} {'t_close':>8s} {'t_lift':>7s} {'no close':>9s} {'no lift':>8s}")
        base = cells.get("none")
        d_all = None
        if base and "all" in cells:
            ck = sorted(set(base) & set(cells["all"]))
            if ck:
                d_all = 100 * np.mean([cells["all"][k]["y"] - base[k]["y"] for k in ck])
        for w in WINDOWS:
            r = cells.get(w)
            if not r:
                continue
            ks = sorted(r)
            sr = 100 * np.mean([r[k]["y"] for k in ks])
            diff, share = "", ""
            if base and w != "none":
                ck = sorted(set(base) & set(r))
                if ck:
                    dd = {k: 100.0 * (r[k]["y"] - base[k]["y"]) for k in ck}
                    lo, hi = boot(ck, dd)
                    diff = f"{np.mean(list(dd.values())):+6.1f} [{lo:+6.1f},{hi:+6.1f}] n={len(ck)}"
                    if d_all is not None and abs(d_all) >= 5 and w != "all":
                        share = f"{np.mean(list(dd.values())) / d_all:5.2f}"
            tc = [r[k]["t_close"] for k in ks if r[k]["t_close"] >= 0]
            tl = [r[k]["t_lift"] for k in ks if r[k]["t_lift"] >= 0]
            L.append(f"{w:10s} {len(ks):4d} {sr:6.1f} {diff:>30s} {share:>13s} "
                     f"{np.median([r[k]['gated'] for k in ks]):12.0f} "
                     f"{(np.median(tc) if tc else float('nan')):8.0f} {(np.median(tl) if tl else float('nan')):7.0f} "
                     f"{100 * (1 - len(tc) / len(ks)):8.0f}% {100 * (1 - len(tl) / len(ks)):7.0f}%")
        L.append("")
    L.append("Reading notes: windows are defined by the episode's own events, so an intervention can lengthen its own "
             "window (compare 'gated steps', 'no close', 'no lift' with the none row). 'share of all' is shown only when "
             "the all-window difference is at least 5 points; shares need not add to 1.")
    return "\n".join(L)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("block_dir")
    ap.add_argument("--out")
    a = ap.parse_args()
    txt = report(a.block_dir)
    print(txt)
    if a.out:
        open(a.out, "w").write(txt + "\n")
    sys.exit(0)
