"""Scene check for libero_object (protocol_block3.md section 6): per (task, episode), the first-step qpos of every
non-robot joint (objects, basket) in a condition's dump against the SRC dump of the same key, plus the static
body placements (ref_body_pos). pi0.5 jsonl records carry the same as "objq" / "fix".
Usage: python tools/fixcheck_object.py SRC_DIR COND_DIR [COND_DIR ...]   (OFT dump dirs or pi0.5 jsonl files)
"""
import glob, json, os, sys
import numpy as np

def load(p):
    out = {}
    if os.path.isdir(p):
        for f in glob.glob(os.path.join(p, "t*_e*_i*.npz")):
            z = np.load(f, allow_pickle=True)
            names = [str(x) for x in z["joint_names"]]
            q0 = z["qpos"][0]
            from_names = {}
            # qpos layout: walk joints in order with their widths (free 7, hinge/slide 1)
            adr = 0
            widths = []
            for n in names:
                widths.append(n)
            objq = {}
            nq = len(q0)
            # free joints are the non-robot joints; robot joints are 1-wide and come first in LIBERO models
            robot = [n for n in names if n.startswith(("robot0", "gripper0"))]
            k = len(robot)
            rest = [n for n in names if not n.startswith(("robot0", "gripper0"))]
            w = (nq - k) / max(len(rest), 1)
            assert names[:k] == robot and w == 7, ("unexpected qpos layout", f, w)
            for i, n in enumerate(rest):
                objq[n] = q0[k + 7 * i: k + 7 * i + 7]
            out[(int(z["task"]), int(z["ep"]))] = (objq, z["ref_body_pos"])
    else:
        for l in open(p):
            r = json.loads(l)
            if r.get("error"):
                continue
            out[(r["task"], r["episode"])] = ({k: np.array(v) for k, v in r["objq"].items()}, None)
    return out

src = load(sys.argv[1])
for c in sys.argv[2:]:
    C = load(c)
    worst, n = 0.0, 0
    for k, (oq, fb) in C.items():
        if k not in src:
            continue
        so, sf = src[k]
        common = set(oq) & set(so)
        d = max(float(np.abs(oq[j][:3] - so[j][:3]).max()) * 1000 for j in common) if common else float("nan")
        if fb is not None and sf is not None:
            d = max(d, float(np.abs(fb - sf).max()) * 1000)
        worst, n = max(worst, d), n + 1
    print(f"{c}: keys {n}, worst object/fixture position diff vs SRC {worst:.4f} mm")
