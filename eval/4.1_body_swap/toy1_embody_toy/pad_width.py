"""TOY1 (decision (c)): actual Robotiq kp100 pad width vs the Panda-equivalent opening given by GRP_MAP=kp100 / kp20.
For each finger_joint value of the kp100 free-air table (gripper_calib.npz r85_fj, r85_q), set the Robotiq linkage,
measure the distance between the two fingerpad geoms (centre to centre, mm) and the inner gap (minus pad thickness is
not modelled: centre distance reported); the Panda pads at the mapped Panda-equivalent joints (0.039 f, -0.039 f) are
measured the same way. Prints a table and the max |Robotiq - Panda| under each map."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from embody import robots  # noqa
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
s = OriginalSpatialSuite(); t = s.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
c = dict(np.load(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embody", "gripper_calib.npz")))
def pads(body, names):
    e = ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    e.reset(); return e, [e.sim.model.geom_name2id("gripper0_" + n) for n in names]
er, gr = pads("PandaR85", ("left_fingerpad_collision", "right_fingerpad_collision"))
ep, gp = pads("Panda", ("finger1_pad_collision", "finger2_pad_collision"))
rr, rp = er.robots[0], ep.robots[0]
def sep(e, g, idx, q):
    e.sim.data.qpos[idx] = q; e.sim.forward()
    return float(np.linalg.norm(e.sim.data.geom_xpos[g[0]] - e.sim.data.geom_xpos[g[1]])) * 1000
closed100 = float(c["r85_fj"][-1])
rows = []
for fj, q6 in zip(c["r85_fj"], c["r85_q"]):
    w_r = sep(er, gr, rr._ref_gripper_joint_pos_indexes, q6)
    out = [fj, w_r]
    for closed in (0.50, closed100):
        f = float(np.clip(1 - fj / closed, 0, 1))
        out.append(sep(ep, gp, rp._ref_gripper_joint_pos_indexes, [0.039 * f, -0.039 * f]))
    rows.append(out)
rows = np.array(rows)
print("finger_joint  robotiq_pad_sep_mm  panda_sep_kp20map_mm  panda_sep_kp100map_mm")
for r in rows[::3].tolist() + [rows[-1].tolist()]:
    print("%.3f  %.1f  %.1f  %.1f" % tuple(r))
w = rows[:, 1]
fr = (w - w.min()) / (w.max() - w.min())
f100 = np.clip(1 - rows[:, 0] / closed100, 0, 1)
f20 = np.clip(1 - rows[:, 0] / 0.50, 0, 1)
print(f"Robotiq pad separation range {w.min():.1f}..{w.max():.1f} mm; Panda range at f=0..1: {rows[:,3].min():.1f}..{rows[:,3].max():.1f} mm")
print(f"opening fraction by pad width vs map: max |f_width - f_kp100| = {np.abs(fr - f100).max():.3f}, max |f_width - f_kp20| = {np.abs(fr - f20).max():.3f}")
print(f"at fj=0.35: f_width={np.interp(0.35, rows[:,0], fr):.3f} f_kp100={1-0.35/closed100:.3f} f_kp20={max(0,1-0.35/0.5):.3f}")
