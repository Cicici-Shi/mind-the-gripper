"""Block-2 smoke acceptance for TWIN_RENDER conditions (protocol_block2.md, section Smoke acceptance).

Reads per-episode dumps of each twin condition, compares with its native counterpart on the same keys.
Usage: python twin_check.py [RUNS_DIR]
"""
import glob
import os
import sys

import numpy as np

R = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "$PROJ_ROOT/runs")
PAIRS = {"sham_twin": "sham", "iiwa_twin": "iiwa", "ur5e_twin": "ur5e", "grpkp100_twin": "grpkp100", "ur5e_rev": "sham"}


def load(cond):
    out = {}
    for f in glob.glob(os.path.join(R, cond, "*.npz")):
        z = np.load(f, allow_pickle=True)
        out[(int(z["task"]), int(z["ep"]))] = z
    return out


for cond, nat in PAIRS.items():
    tw = load(cond)
    if not tw:
        continue
    na = load(nat)
    src = load("src")
    pe = np.concatenate([z["twin_pos_err_mm"] for z in tw.values()])
    re_ = np.concatenate([z["twin_rot_err_deg"] for z in tw.values()])
    t0 = np.stack([z["twin_t0"] for z in tw.values()])  # agent_max, agent_mean, wrist_max, wrist_mean
    stale = sum(int(z["twin_stale"]) for z in tw.values())
    miss = int(((pe > 2) | (re_ > 1)).sum())
    # protocol gate counts policy-query frames only: OFT queries every 8 steps (paint index 0 = step 0)
    qpe = np.concatenate([z["twin_pos_err_mm"][::8] for z in tw.values()])
    qre = np.concatenate([z["twin_rot_err_deg"][::8] for z in tw.values()])
    qmiss = int(((qpe > 2) | (qre > 1)).sum())
    post = [x for z in tw.values() for x in z["twin_posture_deg"]]
    keys = sorted(set(tw) & set(na))
    agree = sum(bool(tw[k]["success"]) == bool(na[k]["success"]) for k in keys)
    s_tw = sum(bool(tw[k]["success"]) for k in keys)
    s_na = sum(bool(na[k]["success"]) for k in keys)
    scene = max(float(np.abs(z["ref_body_pos"] - src[k]["ref_body_pos"]).max()) * 1000 for k, z in tw.items() if k in src)
    print(f"== {cond}  (n={len(tw)}, vs {nat} on {len(keys)} keys)")
    print(f"  t0 agent max {t0[:,0].max():.0f} mean {t0[:,1].max():.4f} | wrist max {t0[:,2].max():.0f} mean {t0[:,3].max():.4f}")
    print(f"  IK pos mm median {np.median(pe):.4f} p99 {np.percentile(pe,99):.4f} max {pe.max():.4f} | rot deg max {re_.max():.4f} | misses {miss} of {len(pe)} | query-frame misses {qmiss} of {len(qpe)} ({100*qmiss/max(len(qpe),1):.2f}%, gate < 1%)")
    print(f"  stale frames {stale}" + (f" | posture p95 {np.percentile(post,95):.2f} deg max {max(post):.2f}" if post else ""))
    print(f"  success twin {s_tw}/{len(keys)} vs native {s_na}/{len(keys)}, agreement {agree}/{len(keys)} | scene vs SRC max {scene:.4f} mm")
