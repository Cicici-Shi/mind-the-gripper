"""Summarize the action-divergence probe (embody/probe.py output).

Per condition: mean position / rotation divergence of the 8-step action chunk from the Panda
reference, gripper-sign flip rate, split by phase (gripper open = approach, closed = carry),
with a bootstrap CI over episodes (frames within an episode are not independent).
'Panda' (same body re-rendered after transfer) is the noise floor.

Usage: python probe_summary.py PROBE_JSONL
"""
import json
import sys
from collections import defaultdict

import numpy as np

CONDS = ["repeat", "Panda", "Panda-hid", "IIWA", "IIWA-3rd-only", "IIWA-wrist-only", "IIWA-hid",
         "UR5e", "UR5e-3rd-only", "UR5e-wrist-only", "UR5e-hid"]


def boot_ci(groups, B=5000, seed=0):
    keys = list(groups)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(B):
        pick = rng.choice(len(keys), len(keys))
        vals = np.concatenate([groups[keys[i]] for i in pick])
        means.append(vals.mean())
    return np.percentile(means, [2.5, 97.5])


def main(path):
    rows = [json.loads(l) for l in open(path)]
    print(f"frames={len(rows)} episodes={len({(r['task'], r['ep']) for r in rows})}")
    rc = np.array([r["replay_check"] for r in rows])
    print(f"replay_check (re-rendered Panda vs logged action, max abs): median {np.median(rc):.4f}  p95 {np.percentile(rc, 95):.4f}")
    for phase, sel in [("all", lambda r: True), ("open", lambda r: not r["gripper_closed"]), ("closed", lambda r: r["gripper_closed"])]:
        sub = [r for r in rows if sel(r)]
        if not sub:
            continue
        print(f"\n[{phase}] n={len(sub)}")
        print(f"{'condition':18s} {'dpos':>7s} {'95% CI':>17s} {'drot':>7s} {'flip%':>6s}")
        for c in CONDS:
            if c not in sub[0]:
                continue
            g = defaultdict(list)
            for r in sub:
                g[(r["task"], r["ep"])].append(r[c]["dpos"])
            g = {k: np.array(v) for k, v in g.items()}
            lo, hi = boot_ci(g)
            dpos = np.mean([r[c]["dpos"] for r in sub])
            drot = np.mean([r[c]["drot"] for r in sub])
            flip = 100 * np.mean([r[c]["grip_flip"] for r in sub])
            print(f"{c:18s} {dpos:7.4f} [{lo:7.4f},{hi:7.4f}] {drot:7.4f} {flip:6.2f}")


if __name__ == "__main__":
    main(sys.argv[1])
