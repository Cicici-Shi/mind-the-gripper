"""GRP interface check: Panda hand vs Robotiq85 on the Panda arm, after grip-site alignment.
Reports finger-closing axis in world for both, wrist camera pose in the grip-site frame,
and saves agentview / wrist images for visual inspection."""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_dev"))
from embody import align, robots  # noqa
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image

OUT = os.path.expanduser("$PROJ_ROOT/embody_dev/grp_check"); os.makedirs(OUT, exist_ok=True)
suite = OriginalSpatialSuite(); tid = 0
task = suite.get_task(tid); init = suite.get_task_init_states(tid)[0]
b = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
mk = lambda r: OffScreenRenderEnv(bddl_file_name=b, camera_heights=256, camera_widths=256, robots=[r])

def pad_axis(e):
    m, d = e.sim.model, e.sim.data
    names = [m.geom_id2name(g) or "" for g in range(m.ngeom)]
    L = [g for g, n in enumerate(names) if n.startswith("gripper0") and "pad" in n and ("left" in n or "finger1" in n) and "collision" in n]
    R = [g for g, n in enumerate(names) if n.startswith("gripper0") and "pad" in n and ("right" in n or "finger2" in n) and "collision" in n]
    pl, pr = d.geom_xpos[L].mean(0), d.geom_xpos[R].mean(0)
    v = pr - pl
    return v / np.linalg.norm(v), float(np.linalg.norm(v)) * 1000, [names[g] for g in L + R]

ref = mk("Panda"); ref.seed(0); ref.reset(); obs = ref.set_init_state(init)
for _ in range(10): obs, *_ = ref.step([0, 0, 0, 0, 0, 0, -1])
f = align.frames(ref)
ax_p, w_p, n_p = pad_axis(ref)
print("Panda pads", n_p, "width_mm", round(w_p, 1), "axis", ax_p.round(3))
print("Panda cam in site", f["wrist_cam_pos_in_site"].round(4), "hand_in_site\n", f["hand_in_site"].round(3))
Image.fromarray(obs["agentview_image"][::-1]).save(f"{OUT}/panda_agent.png")
Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(f"{OUT}/panda_wrist.png")

tg = mk("PandaR85"); tg.seed(0); tg.reset()
m = tg.sim.model
print("R85 gripper joints:", [tg.sim.model.joint_id2name(j) for j in range(m.njnt) if (m.joint_id2name(j) or "").startswith("gripper0")])
print("R85 gripper qpos idx", tg.robots[0]._ref_gripper_joint_pos_indexes)
q = align.ik_site(tg, f["site_pos"], f["site_mat"], tg.robots[0].init_qpos)
align.apply_static_body_pose(tg, align.static_body_pose(ref))
objs = align.object_state_by_name(ref)
align.apply_object_state(tg, objs)
tg.sim.data.qpos[tg.robots[0]._ref_joint_pos_indexes] = q
tg.sim.forward()
ft = align.frames(tg)
ax_r, w_r, n_r = pad_axis(tg)
print("R85 pads", n_r, "width_mm", round(w_r, 1), "axis", ax_r.round(3))
print("angle between finger axes (deg):", round(float(np.degrees(np.arccos(abs(np.clip(ax_p @ ax_r, -1, 1))))), 2))
print("site pos err mm", round(float(np.linalg.norm(ft["site_pos"] - f["site_pos"]) * 1000), 4), "rot err deg", round(align.rot_err_deg(ft["site_mat"], f["site_mat"]), 4))
print("R85 cam in site", ft["wrist_cam_pos_in_site"].round(4), "hand_in_site\n", ft["hand_in_site"].round(3))
tg.env._update_observables(force=True); o = tg.env._get_observations()
print("R85 gripper_qpos obs", o["robot0_gripper_qpos"].round(3), "eef_quat", o["robot0_eef_quat"].round(3), "panda eef_quat", obs["robot0_eef_quat"].round(3))
Image.fromarray(o["agentview_image"][::-1]).save(f"{OUT}/r85_agent.png")
Image.fromarray(o["robot0_eye_in_hand_image"][::-1]).save(f"{OUT}/r85_wrist.png")
