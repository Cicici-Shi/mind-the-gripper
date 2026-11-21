import os, sys
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_toy"))
import numpy as np
from embody import robots, align  # noqa
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
s = OriginalSpatialSuite(); t = s.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
for body, pads in (("Panda", ("finger1_pad_collision", "finger2_pad_collision")), ("PandaR85", ("left_fingerpad_collision", "right_fingerpad_collision"))):
    e = ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    e.reset(); sim = e.sim; m, d = sim.model, sim.data
    p, R = align.site_pose(sim, e.robots[0].eef_site_id)
    a = d.geom_xpos[m.geom_name2id("gripper0_" + pads[0])]; b = d.geom_xpos[m.geom_name2id("gripper0_" + pads[1])]
    ax = (a - b) / np.linalg.norm(a - b)
    hb = m.body_name2id(e.robots[0].robot_model.eef_name); Rh = d.xmat[hb].reshape(3, 3)
    base = [i for i in range(m.nbody) if m.body_parentid[i] == 0 and (m.body_id2name(i) or "").startswith("robot0")]
    print(body, "pad axis . site x/y/z", np.round(R.T @ ax, 3), "site-in-hand", np.round(Rh.T @ R, 3).tolist(), "base", [m.body_id2name(i) for i in base], "Rbase", np.round(d.xmat[base[0]].reshape(3,3),3).tolist(), "pad sep mm", round(float(np.linalg.norm(a-b))*1000,1))
