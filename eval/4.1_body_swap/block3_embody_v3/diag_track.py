"""CPU diagnostic (i): per-step OSC tracking error from existing 200-episode dumps (no simulation).

For each step k of each episode: commanded delta = action[:3] * 0.05 m and action[3:6] * 0.5 rad
(OSC_POSE output_max; world-frame axis-angle as robosuite applies it), actual delta = eef_pos[k+1] -
eef_pos[k] and the world-frame rotation R_{k+1} R_k^T (eef_quat is the hand body; hand_in_site is
constant, so the world-frame increment equals the grip-site increment). Reports per condition:
position / orientation error |actual - commanded| (median, p90), gain = <actual, cmd> / |cmd|^2 for
steps with a sizeable command, the orientation error split by episode phase (step bins) and by the
size of the commanded rotation, and the per-axis split of the orientation error (world x, y, z).
Usage: python tools/diag_track.py [cond ...]
"""
import glob
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

R = os.path.expanduser("$PROJ_ROOT/runs")
conds = sys.argv[1:] or ["src", "sham", "rec", "iiwa", "ur5e", "grpkp100", "grp", "apps"]
print(f"{'cond':10s} {'n_steps':>8s} | pos err mm med/p90 | rot err deg med/p90 | pos gain | rot gain | "
      "rot err p90 by step bin [0-40,40-80,80-120,120+] | rot err p90 by |cmd rot| [<2,2-5,>5 deg] | rot err med world |x|,|y|,|z| deg")
for c in conds:
    P, Rr, gp, gr, step, cmdr, axes = [], [], [], [], [], [], []
    for f in sorted(glob.glob(os.path.join(R, c, "t*_e*_i*.npz"))):
        z = np.load(f, allow_pickle=True)
        a, p, q = z["action"].astype(float), z["eef_pos"].astype(float), z["eef_quat"].astype(float)
        if len(a) < 2:
            continue
        cp = a[:-1, :3] * 0.05
        cr = a[:-1, 3:6] * 0.5
        ap_ = p[1:] - p[:-1]
        rot = Rotation.from_quat(q[1:]) * Rotation.from_quat(q[:-1]).inv()
        ar = rot.as_rotvec()
        P.append(np.linalg.norm(ap_ - cp, axis=1) * 1000)
        e = ar - cr
        Rr.append(np.degrees(np.linalg.norm(e, axis=1)))
        axes.append(np.degrees(np.abs(e)))
        big = np.linalg.norm(cp, axis=1) > 0.005
        gp.append((np.sum(ap_ * cp, axis=1) / np.maximum(np.sum(cp * cp, axis=1), 1e-12))[big])
        bigr = np.linalg.norm(cr, axis=1) > np.radians(2)
        gr.append((np.sum(ar * cr, axis=1) / np.maximum(np.sum(cr * cr, axis=1), 1e-12))[bigr])
        step.append(np.arange(len(cp)))
        cmdr.append(np.degrees(np.linalg.norm(cr, axis=1)))
    if not P:
        continue
    P, Rr, gp, gr, step, cmdr, axes = map(np.concatenate, (P, Rr, gp, gr, step, cmdr, axes))
    bins = [Rr[(step >= lo) & (step < hi)] for lo, hi in ((0, 40), (40, 80), (80, 120), (120, 10**6))]
    cb = [Rr[(cmdr >= lo) & (cmdr < hi)] for lo, hi in ((0, 2), (2, 5), (5, 1e9))]
    f90 = lambda x: f"{np.percentile(x, 90):.2f}" if len(x) else "-"  # noqa: E731
    print(f"{c:10s} {len(P):8d} | {np.median(P):6.2f} / {np.percentile(P, 90):6.2f} | {np.median(Rr):6.2f} / {np.percentile(Rr, 90):6.2f} | "
          f"{np.median(gp):.3f} | {np.median(gr):.3f} | {', '.join(f90(b) for b in bins)} | {', '.join(f90(b) for b in cb)} | "
          f"{', '.join(f'{x:.2f}' for x in np.median(axes, axis=0))}")
