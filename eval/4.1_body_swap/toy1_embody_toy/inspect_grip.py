import os, sys
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_toy"))
from embody import robots  # noqa
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
s = OriginalSpatialSuite(); t = s.get_task(0)
bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
for body in ("Panda", "PandaR85"):
    e = ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    m = e.sim.model
    print("==", body, "eef", e.robots[0].robot_model.eef_name)
    for b in range(m.nbody):
        n = m.body_id2name(b) or ""
        if n.startswith(("gripper0", "robot0_right_hand", "robot0_link7")):
            gs = [g for g in range(m.ngeom) if m.geom_bodyid[g] == b]
            print(n, "parent", m.body_id2name(m.body_parentid[b]), "jnt", m.body_jntnum[b], "pos", m.body_pos[b].round(4),
                  [(m.geom_id2name(g), int(m.geom_group[g]), round(float(m.geom_rgba[g][3]),2), int(m.geom_contype[g])) for g in gs])
    print("cams", [m.camera_id2name(i) + ":" + str(m.body_id2name(m.cam_bodyid[i])) for i in range(m.ncam)])
