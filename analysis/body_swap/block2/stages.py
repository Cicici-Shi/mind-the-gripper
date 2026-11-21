"""Stage metrics and failure labels from per-step dumps (numpy only; runs on the login node).

Stages for "pick up the <target> black bowl and place it on the plate":
  reach     eef (grip site) within 7.5 cm of the bowl origin at some step (all successful
            episodes so far come within 7.2 cm; the bowl origin is below the rim it is grasped by)
  lift      target bowl raised > 3 cm above its start
  transport target bowl, after being lifted and moved > 5 cm, within 6 cm (xy) of the plate
  success   task success flag
Flags: wrong_object (the other bowl lifted > 3 cm), drop (target lifted, later back below
1 cm while not over the plate). A failure is labelled by the first stage it misses.

Usage: python stages.py RUNS_DIR [cond ...]
"""
import glob
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

BDDL_DIR = None
_PERTURBED = re.compile(r"_view_|_add_|_level|_light_|_table_\d|_tb_\d|_language_|_noise_| copy")


def task_objects():
    cfg = open(os.path.expanduser("~/.libero/config.yaml")).read()
    bddl = re.search(r"bddl_files:\s*(\S+)", cfg).group(1)
    d = os.path.join(bddl, "libero_spatial")
    names = sorted(f for f in os.listdir(d) if f.endswith(".bddl") and not _PERTURBED.search(f))
    out = []
    for f in names:
        txt = open(os.path.join(d, f)).read()
        target, plate = re.search(r"\(:obj_of_interest(.*?)\)", txt, re.S).group(1).split()[:2]
        out.append((target, plate))
    return out


def qpos_addr(joint_names):
    addr, a = {}, 0
    for n in joint_names:
        n = str(n)
        size = 7 if (n.endswith("_joint0") and not n.startswith(("robot0", "gripper0"))) else 1
        addr[n] = (a, size)
        a += size
    return addr


def episode_stages(z, target, plate):
    names = [str(n) for n in z["joint_names"]]
    ad = qpos_addr(names)
    q = z["qpos"]
    if q.shape[1] != sum(s for _, s in ad.values()):
        return None
    pos = lambda obj: q[:, ad[obj + "_joint0"][0]: ad[obj + "_joint0"][0] + 3]
    bowl, pl = pos(target), pos(plate)
    other = [n[:-len("_joint0")] for n in names if "black_bowl" in n and not n.startswith(target)]
    eef = z["eef_pos"]
    rise = bowl[:, 2] - bowl[0, 2]
    lifted = rise > 0.03
    over = np.linalg.norm(bowl[:, :2] - pl[:, :2], axis=1) < 0.06
    # carried: off its start spot by > 5 cm in xy after having been lifted (the plate can sit lower
    # than the start, e.g. bowl on the cabinet, so height above start is not required at the plate)
    moved = np.linalg.norm(bowl[:, :2] - bowl[0, :2], axis=1) > 0.05
    carried = np.maximum.accumulate(lifted) & moved
    s = {
        "reach": bool((np.linalg.norm(eef - bowl, axis=1) < 0.075).any()),
        "lift": bool(lifted.any()),
        "transport": bool((carried & over).any()),
        "success": bool(z["success"]),
        "wrong_object": bool(any(((pos(o)[:, 2] - pos(o)[0, 2]) > 0.03).any() for o in other)),
    }
    drop = False
    if lifted.any():
        t0 = int(np.argmax(lifted))
        drop = bool(((rise[t0:] < 0.01) & ~over[t0:]).any())
    s["drop"] = drop
    if s["success"]:
        s["label"] = "success"
    elif s["wrong_object"]:
        s["label"] = "wrong_object"
    elif not s["reach"]:
        s["label"] = "no_reach"
    elif not s["lift"]:
        s["label"] = "no_grasp"
    elif s["drop"]:
        s["label"] = "drop"
    elif not s["transport"]:
        s["label"] = "no_transport"
    else:
        s["label"] = "no_place"
    return s


def main(runs, conds):
    objs = task_objects()
    conds = conds or sorted(os.path.basename(p) for p in glob.glob(os.path.join(runs, "*")) if os.path.isdir(p))
    print(f"{'condition':12s} {'n':>4s} {'reach':>6s} {'lift':>6s} {'transp':>6s} {'succ':>6s} | failure labels")
    for c in conds:
        rows = []
        for f in sorted(glob.glob(os.path.join(runs, c, "*.npz"))):
            z = np.load(f, allow_pickle=True)
            t = int(z["task"])
            st = episode_stages(z, *objs[t])
            if st:
                rows.append(st)
        if not rows:
            continue
        n = len(rows)
        pct = lambda k: 100 * sum(r[k] for r in rows) / n
        labels = Counter(r["label"] for r in rows if r["label"] != "success")
        print(f"{c:12s} {n:4d} {pct('reach'):6.1f} {pct('lift'):6.1f} {pct('transport'):6.1f} {pct('success'):6.1f} | {dict(labels)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
