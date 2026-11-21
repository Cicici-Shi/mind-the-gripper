"""Closed-gripper finger-pad gap: Panda vs Rethink (free air, pads inner surfaces), and bowl rim thickness."""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embody import robots  # noqa
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
sys.argv = sys.argv[:1]
exec(open("tools/calib_grippers.py").read().split("out = {}")[0])
for body, pads, cmd in (("Panda", ("finger1_pad_collision", "finger2_pad_collision"), [-1, 1]), ("PandaRT", ("l_fingerpad_g0", "r_fingerpad_g0"), [1, -1])):
    e = make(body); settle(e, cmd, 800); m, d = e.sim.model, e.sim.data
    gs = [g for g in range(m.ngeom) if any((m.geom_id2name(g) or "").endswith(p) for p in pads)]
    c = [d.geom_xpos[g] for g in gs]; R = d.geom_xmat[gs[0]].reshape(3, 3)
    print(body, "closed: pad centre gap mm %.2f, pad half-sizes mm %s, type %s" % (np.linalg.norm(c[0] - c[1]) * 1000, np.round(m.geom_size[gs[0]] * 1000, 2), m.geom_type[gs[0]]))
    e.close()
