"""Action-divergence probe split by task phase (SRC successful episodes only).

Each probe frame (task, ep, step) is labelled from the SRC per-step dump:
  approach   grip site > 7.5 cm from the target bowl, gripper open, bowl not yet lifted
  pre-grasp  within 7.5 cm, gripper command still open, before the first close
  grasp      from the first close command until the bowl is lifted > 3 cm
  transport  bowl lifted, before the gripper command opens again
  place      after the first open command that follows the lift
dpos is the probe's position divergence of the action chunk; rel = dpos / |a_ref pos| of that frame.
CI: bootstrap over episodes.

Usage: python probe_phase.py PROBE_JSONL SRC_DUMP_DIR
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stages import qpos_addr, task_objects  # noqa: E402

PHASES = ["approach", "pre-grasp", "grasp", "transport", "place"]
CONDS = ["Panda", "Panda-hid", "IIWA", "IIWA-3rd-only", "IIWA-wrist-only", "IIWA-hid",
         "UR5e", "UR5e-3rd-only", "UR5e-wrist-only", "UR5e-hid"]


def phase_labels(z, target):
    names = [str(n) for n in z["joint_names"]]
    ad = qpos_addr(names)
    a0 = ad[target + "_joint0"][0]
    bowl = z["qpos"][:, a0:a0 + 3]
    eef = z["eef_pos"]
    close = z["action"][:, -1] > 0          # LIBERO gripper: +1 close, -1 open
    lifted = (bowl[:, 2] - bowl[0, 2]) > 0.03
    T = len(eef)
    t_close = int(np.argmax(close)) if close.any() else T
    t_lift = int(np.argmax(lifted)) if lifted.any() else T
    after = np.where(~close[t_lift:])[0] if t_lift < T else []
    t_rel = t_lift + int(after[0]) if len(after) else T
    near = np.linalg.norm(eef - bowl, axis=1) < 0.075
    lab = []
    for t in range(T):
        if t >= t_rel:
            lab.append("place")
        elif t >= t_lift:
            lab.append("transport")
        elif t >= t_close:
            lab.append("grasp")
        elif near[t]:
            lab.append("pre-grasp")
        else:
            lab.append("approach")
    return lab


def boot(groups, B=4000, seed=0):
    keys = list(groups)
    rng = np.random.default_rng(seed)
    m = [np.concatenate([groups[keys[i]] for i in rng.choice(len(keys), len(keys))]).mean() for _ in range(B)]
    return np.percentile(m, [2.5, 97.5])


def main(probe, src):
    objs = task_objects()
    dumps = {}
    for f in glob.glob(os.path.join(src, "t*_e*_i*.npz")):
        z = np.load(f, allow_pickle=True)
        if bool(z["success"]):
            dumps[(int(z["task"]), int(z["ep"]))] = phase_labels(z, objs[int(z["task"])][0])
    rows = []
    for l in open(probe):
        r = json.loads(l)
        lab = dumps.get((r["task"], r["ep"]))
        if lab is None or r["step"] >= len(lab):
            continue
        r["phase"] = lab[r["step"]]
        rows.append(r)
    print(f"frames={len(rows)} episodes={len({(r['task'], r['ep']) for r in rows})}")
    print("frames per phase:", {p: sum(r["phase"] == p for r in rows) for p in PHASES})
    for metric in ["dpos", "rel"]:
        print(f"\n== {metric} (mean [95% CI over episodes])")
        print(f"{'condition':16s}" + "".join(f"{p:>24s}" for p in PHASES))
        for c in CONDS:
            line = f"{c:16s}"
            for p in PHASES:
                g = defaultdict(list)
                for r in rows:
                    if r["phase"] != p:
                        continue
                    v = r[c]["dpos"]
                    if metric == "rel":
                        v = v / max(r["a_ref_pos_norm"], 1e-6)
                    g[(r["task"], r["ep"])].append(v)
                if not g:
                    line += f"{'-':>24s}"
                    continue
                g = {k: np.array(v) for k, v in g.items()}
                mean = np.concatenate(list(g.values())).mean()
                lo, hi = boot(g)
                line += f"{mean:9.3f} [{lo:5.3f},{hi:5.3f}]"
            print(line)
    print("\n== gripper flip % (sign of predicted gripper differs from Panda input)")
    for c in CONDS:
        print(f"{c:16s}" + "".join(f"{100*np.mean([r[c]['grip_flip'] for r in rows if r['phase']==p] or [0]):24.2f}" for p in PHASES))


if __name__ == "__main__":
    main(*sys.argv[1:3])
