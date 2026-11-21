"""Block-3 calibration additions (appends keys to embody/gripper_calib.npz).
- r85 pad centre per table row: midpoint of the two fingerpad collision geoms, in the HAND frame (z) and the
  site frame; Panda pad centre in the hand frame. Gives the sunk-twin offset and how much it varies with opening.
- RT2: free-air open / closed slide positions, inner pad gap (centre gap minus pad thickness), finger overlap.
"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.argv = sys.argv[:1]
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib_grippers.py")).read().split("out = {}")[0]
exec(src)
P = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "embody", "gripper_calib.npz")
out = dict(np.load(P))

def pad_centre(e, names):
    m, d = e.sim.model, e.sim.data
    gs = [m.geom_name2id("gripper0_" + n) for n in names]
    return np.mean([d.geom_xpos[g] for g in gs], axis=0), gs

def in_hand(e, p):
    r = e.robots[0]
    b = e.sim.model.body_name2id(r.robot_model.eef_name)
    R = e.sim.data.xmat[b].reshape(3, 3); x = e.sim.data.xpos[b]
    return R.T @ (p - x)

e = make("Panda"); settle(e, [1, -1], 600)
pc, _ = pad_centre(e, ("finger1_pad_collision", "finger2_pad_collision"))
out["panda_pad_hand"] = in_hand(e, pc); print("Panda pad centre in hand frame mm", np.round(out["panda_pad_hand"] * 1000, 3))
e.close()
e = make("PandaR85"); r = e.robots[0]; gidx = np.array(r._ref_gripper_joint_pos_indexes)
ph = []
for q in out["r85_q"]:
    e.sim.data.qpos[gidx] = q; e.sim.forward()
    pc, _ = pad_centre(e, ("left_fingerpad_collision", "right_fingerpad_collision"))
    ph.append(in_hand(e, pc))
ph = np.array(ph); out["r85_pad_hand"] = ph
print("R85 pad centre in hand frame z mm: open %.2f, fj=0.5 %.2f, closed %.2f; x,y max |.| %.3f %.3f" % (
    ph[0, 2] * 1000, np.interp(0.5, out["r85_fj"], ph[:, 2]) * 1000, ph[-1, 2] * 1000, np.abs(ph[:, 0]).max() * 1000, np.abs(ph[:, 1]).max() * 1000))
print("sunk offset needed (R85 pad z - Panda pad z) mm: open %.2f closed(fj .5) %.2f closed(kp100 plateau) %.2f" % (
    (ph[0, 2] - out["panda_pad_hand"][2]) * 1000, (np.interp(0.5, out["r85_fj"], ph[:, 2]) - out["panda_pad_hand"][2]) * 1000, (ph[-1, 2] - out["panda_pad_hand"][2]) * 1000))
e.close()
e = make("PandaRT2")
m = e.sim.model
print("RT2 joint ranges", [m.jnt_range[m.joint_name2id(n)] for n in ("gripper0_l_finger_joint", "gripper0_r_finger_joint")],
      "ctrlranges", [m.actuator_ctrlrange[m.actuator_name2id(n)] for n in ("gripper0_gripper_l_finger_joint", "gripper0_gripper_r_finger_joint")],
      "kp", [m.actuator_gainprm[m.actuator_name2id(n), 0] for n in ("gripper0_gripper_l_finger_joint", "gripper0_gripper_r_finger_joint")],
      "force", [m.actuator_forcerange[m.actuator_name2id(n)] for n in ("gripper0_gripper_l_finger_joint", "gripper0_gripper_r_finger_joint")])
for tag, cmd in (("open", [-1, 1]), ("closed", [1, -1])):
    q = settle(e, cmd, 800); out["rt2_" + tag] = q
    d = e.sim.data
    pl, pr = [d.geom_xpos[m.geom_name2id(f"gripper0_{s}_fingerpad_g0")] for s in ("l", "r")]
    half = m.geom_size[m.geom_name2id("gripper0_l_fingerpad_g0")][1]
    gap = np.linalg.norm(pl - pr) - 2 * half
    out[f"rt2_inner_gap_{tag}"] = gap
    print(f"RT2 {tag}: q {np.round(q, 5)}, pad centre gap {np.linalg.norm(pl - pr) * 1000:.2f} mm, inner gap {gap * 1000:.2f} mm")
out["rt2_pad_z"] = out["rt_pad_z"]
e.close()
e = make("PandaRT")
q = settle(e, [-1, 1], 800); d = e.sim.data
pl, pr = [d.geom_xpos[e.sim.model.geom_name2id(f"gripper0_{s}_fingerpad_g0")] for s in ("l", "r")]
print("stock RT open pad centre gap %.2f mm" % (np.linalg.norm(pl - pr) * 1000))
e.close()
np.savez(P, **out); print("wrote", P)
