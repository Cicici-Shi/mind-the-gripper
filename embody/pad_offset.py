import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_dev"))
from embody import align, robots  # noqa
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
suite = OriginalSpatialSuite(); task = suite.get_task(0)
b = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
for r in ["Panda", "PandaR85"]:
    e = OffScreenRenderEnv(bddl_file_name=b, camera_heights=64, camera_widths=64, robots=[r]); e.seed(0); e.reset()
    m, d = e.sim.model, e.sim.data
    p_s, R_s = align.site_pose(e.sim, e.robots[0].eef_site_id)
    for g in range(m.ngeom):
        n = m.geom_id2name(g) or ""
        if n.startswith("gripper0") and ("pad" in n or "tip" in n or "finger" in n) and "collision" in n:
            rel = R_s.T @ (d.geom_xpos[g] - p_s)
            print(r, f"{n:45s} in-site(mm) {np.round(rel*1000,1)}  size {np.round(m.geom_size[g]*1000,1)}")
    e.close()
