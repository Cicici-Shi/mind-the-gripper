"""Free-air gripper calibration for embody_v3 (writes embody/gripper_calib.npz).

r85_fj, r85_q : Robotiq85 six linkage joints as a function of finger_joint, from 50 commands between
                full open and full closed, each settled 300 substeps (arm and scene held), kp 100 (G's kp).
                Used by the G_R twin (Panda physics, Robotiq pixels).
rt_open, rt_closed : Rethink [l, r] slide joints after settling at command open / closed (600 substeps).
rt_pad_z, panda_pad_z, r85_pad_z : finger-pad centre z in the grip-site frame (m), same method for all
                three (mean of the two pad collision geoms at the open state) so the Panda/Robotiq values
                can be checked against pad_offset.py (-3.6 / -10.0 mm).
Usage (compute node): python tools/calib_grippers.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embody import align, robots  # noqa: F401,E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

suite = OriginalSpatialSuite()
task = suite.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)


def make(body):
    e = ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False, has_offscreen_renderer=False,
                   use_camera_obs=False)
    e.seed(0)
    e.reset()
    return e


def settle(e, cmd, n, kp=None):
    sim, m, d = e.sim, e.sim.model, e.sim.data
    robot = e.robots[0]
    act = [i for i in range(m.nu) if (m.actuator_id2name(i) or "").startswith("gripper0")]
    if kp:
        for i in act:
            m.actuator_gainprm[i, 0] = float(kp)
            m.actuator_biasprm[i, 1] = -float(kp)
    lo, hi = m.actuator_ctrlrange[act, 0], m.actuator_ctrlrange[act, 1]
    target = (lo + hi) / 2 + (hi - lo) / 2 * np.asarray(cmd, dtype=float)
    keep = d.qpos.copy()
    gidx = np.array(robot._ref_gripper_joint_pos_indexes)
    gvel = np.array(robot._ref_gripper_joint_vel_indexes)
    nz = np.setdiff1d(np.arange(m.nv), gvel)
    for _ in range(n):
        d.ctrl[:] = 0.0
        d.ctrl[act] = target
        sim.step()
        g = d.qpos[gidx].copy()
        d.qpos[:] = keep
        d.qpos[gidx] = g
        d.qvel[nz] = 0.0
    sim.forward()
    return d.qpos[gidx].copy()


def pad_z(e, pads):
    sim, m, d = e.sim, e.sim.model, e.sim.data
    p_s, R_s = align.site_pose(sim, e.robots[0].eef_site_id)
    zs = []
    for g in range(m.ngeom):
        n = m.geom_id2name(g) or ""
        if any(n.endswith(p) for p in pads):
            zs.append((R_s.T @ (d.geom_xpos[g] - p_s))[2])
    assert len(zs) == 2, (pads, zs)
    return float(np.mean(zs))


out = {}
# Robotiq 85 table
e = make("PandaR85")
fj, qs = [], []
for c in np.linspace(-1, 1, 50):
    q = settle(e, [c], 300, kp=100)
    fj.append(q[0])
    qs.append(q)
fj, qs = np.array(fj), np.array(qs)
order = np.argsort(fj, kind="stable")
fj, qs = fj[order], qs[order]
keep = np.concatenate([[True], np.diff(fj) > 1e-6])
out["r85_fj"], out["r85_q"] = fj[keep], qs[keep]
settle(e, [-1], 600, kp=100)
m_, d_ = e.sim.model, e.sim.data
p_s, R_s = align.site_pose(e.sim, e.robots[0].eef_site_id)
for g in range(m_.ngeom):
    n = m_.geom_id2name(g) or ""
    if n.startswith("gripper0") and "collision" in n:
        print("  R85 geom", n, "in-site mm", np.round(R_s.T @ (d_.geom_xpos[g] - p_s) * 1000, 1))
out["r85_pad_z"] = -0.0100  # pad_offset.py, 2026-10-26 (block-2 constant)
print("R85 finger_joint range", fj.min(), fj.max(), "n", int(keep.sum()), "pad_z mm", out["r85_pad_z"] * 1000)
e.close()

# Panda (method check against pad_offset.py)
e = make("Panda")
settle(e, [1, -1], 600)  # Panda: [-1,1] closes, so [1,-1] opens
out["panda_pad_z"] = pad_z(e, ("finger1_pad_collision", "finger2_pad_collision"))
print("Panda pad_z mm", out["panda_pad_z"] * 1000)
e.close()

# Rethink
e = make("PandaRT")
print("Rethink gripper joints", e.robots[0].gripper_joints)
out["rt_open"] = settle(e, [-1, 1], 600)
out["rt_pad_z"] = pad_z(e, ("l_fingerpad_g0", "r_fingerpad_g0"))
out["rt_closed"] = settle(e, [1, -1], 600)
print("Rethink open", out["rt_open"], "closed", out["rt_closed"], "pad_z mm", out["rt_pad_z"] * 1000)
gap = {}
for tag, cmd in (("open", [-1, 1]), ("closed", [1, -1])):
    settle(e, cmd, 600)
    m, d = e.sim.model, e.sim.data
    p = [d.geom_xpos[m.geom_name2id(f"gripper0_{s}_fingerpad_g0")] for s in ("l", "r")]
    gap[tag] = float(np.linalg.norm(p[0] - p[1]))
print("Rethink pad-centre gap mm", {k: round(v * 1000, 2) for k, v in gap.items()})
e.close()

path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embody", "gripper_calib.npz")
np.savez(path, **out)
print("wrote", path)
