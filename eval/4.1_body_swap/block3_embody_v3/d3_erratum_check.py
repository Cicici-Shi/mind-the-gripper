"""Amendment 2 ruling 3 (b), D3 erratum (GPT Astra 6 ruling, protocol_toy1_astra_decisions.md last paragraph).
Corrected identity, signed lateral VECTORS in the physical hand frame (mm):
    root_off_lat  ==  PandaPad_lat(q1, q2)  -  R85Pad_lat(fj(f))
PandaPad_lat: Panda pad-centre (mean of finger1/2_pad_collision geoms) in the hand frame, by forward kinematics of
    the robosuite Panda + PandaGripper MJCF at finger joints (q1, q2) = dump gripper_qpos[k] (the same pad geoms
    and hand body that embody/body_env.py::_twin_paint_sunk uses).
R85Pad_lat: gripper_calib.npz r85_pad_hand interpolated at fj = fj0 + (1 - f)(fjN - fj0), f = clip((q1 - q2) / (2 *
    0.039), 0, 1) (body_env PANDA_OPEN and _open_fraction_panda for a Panda physical gripper).
Check: max |root_off_lat - (PandaPad_lat - R85Pad_lat)| <= 0.5 mm over query frames (OFT twin_root_off_mm[::8], frame
k <-> gripper_qpos[k], the last frame has no row and is skipped). The old D3 (|lateral| - |q1 + q2| / 2, moduli) is
printed alongside, unchanged. Thresholds 0.5 mm and 6.4 mm unchanged. Does not touch prereg_block3_a2.py.
Usage: python tools/d3_erratum_check.py [--runs $PROJ_ROOT/runs] [--cell g_r] [--calib embody/gripper_calib.npz]
"""
import argparse, glob, os, re
import numpy as np
import mujoco
from robosuite.models.robots import Panda
from robosuite.models.grippers import PandaGripper

ap = argparse.ArgumentParser()
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--cell", default="g_r")
ap.add_argument("--calib", default=os.path.expanduser("$PROJ_ROOT/embody_v3/embody/gripper_calib.npz"))
a = ap.parse_args()

rob = Panda(); rob.add_gripper(PandaGripper(idn=0))
m = rob.get_model(mode="mujoco"); d = mujoco.MjData(m)
hand = [i for i in range(m.nbody) if m.body(i).name.endswith("right_hand")][0]
jadr = [m.jnt_qposadr[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"gripper0_finger_joint{i}")] for i in (1, 2)]
pads = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"gripper0_finger{i}_pad_collision") for i in (1, 2)]
print(f"hand body {m.body(hand).name}; pads {[m.geom(p).name for p in pads]}")


def panda_pad(q1, q2):
    d.qpos[:] = 0
    d.qpos[jadr[0]], d.qpos[jadr[1]] = q1, q2
    mujoco.mj_kinematics(m, d)
    R = d.xmat[hand].reshape(3, 3)
    return R.T @ (np.mean(d.geom_xpos[pads], 0) - d.xpos[hand]) * 1000


for q in [(0.039, -0.039), (0.002, -0.002), (0.0374, -0.0190)]:
    print(f"FK pad centre in hand frame at q={q}: {np.round(panda_pad(*q), 4)} mm")
c = dict(np.load(a.calib))
fjt, r85 = c["r85_fj"], c["r85_pad_hand"] * 1000
print(f"calib panda_pad_hand (free air) {np.round(c['panda_pad_hand'] * 1000, 4)} mm; r85 lateral y range {r85[:, 1].min():.3f}..{r85[:, 1].max():.3f} mm")

res_new, res_old, per_ep, pre_new = [], [], [], []
for f in sorted(glob.glob(os.path.join(a.runs, a.cell, "t*_e*_i*.npz"))):
    z = np.load(f, allow_pickle=True)
    if "twin_root_off_mm" not in z.files:
        continue
    ro = np.asarray(z["twin_root_off_mm"], float)
    gq = np.asarray(z["gripper_qpos"], float)
    fg = int(z["contact_first_grip_obj"]) if "contact_first_grip_obj" in z.files else -1
    ks = np.arange(len(ro))[::8]
    ks = ks[ks < len(gq)]
    worst = 0.0
    for k in ks:
        q1, q2 = gq[k]
        fr = float(np.clip((q1 - q2) / (2 * 0.039), 0.0, 1.0))
        fj = fjt[0] + (1.0 - fr) * (fjt[-1] - fjt[0])
        pred = panda_pad(q1, q2)[:2] - np.array([np.interp(fj, fjt, r85[:, i]) for i in (0, 1)])
        rn = float(np.linalg.norm(ro[k, :2] - pred))
        ro_ = abs(float(np.hypot(ro[k, 0], ro[k, 1])) - abs(q1 + q2) * 500)
        res_new.append(rn); res_old.append(ro_); worst = max(worst, rn)
        if fg < 0 or k <= fg:
            pre_new.append(rn)
    m_ = re.search(r"t(\d+)_e(\d+)", os.path.basename(f))
    per_ep.append((worst, int(m_.group(1)), int(m_.group(2))))
rn, ro_ = np.array(res_new), np.array(res_old)
print(f"\n{a.cell}: {len(per_ep)} episodes, {len(rn)} query frames")
print(f"OLD D3 (moduli, unchanged): max {ro_.max():.3f} mm, frames > 0.5 mm {int((ro_ > 0.5).sum())}  -> {'停下查' if ro_.max() > 0.5 else 'pass'}")
print(f"ERRATUM (signed lateral vectors): max {rn.max():.3f} mm, p99 {np.percentile(rn, 99):.3f}, median {np.median(rn):.4f}, "
      f"frames > 0.5 mm {int((rn > 0.5).sum())}; before first free-object contact max {max(pre_new):.3f} mm  -> "
      f"{'停下查（实现有别的问题）' if rn.max() > 0.5 else '通过，警报解除'}")
print("worst episodes:", [(round(w, 3), t, e) for w, t, e in sorted(per_ep, reverse=True)[:5]])
