"""embody_next (2026-11-06): free-air calibration and smoke of the Robotiq 140 body (PandaR140), no policy.
For each R140_KP in --kps and each (suite, task 0, ep 0 init) scene: BodyEnv set_init_state (R140_CLOSED=0.7 dummy so the
proprio map does not assert), then on the raw robosuite env: arm action 0, gripper open (-1) for 120 steps, closed (+1)
for 300 steps, open again for 200. Per step: the six gripper joints, pad centres in the grip-site frame, pad distance,
gripper-scene contacts, QACC warnings. Prints the open / closed finger_joint plateaus and the open pad-centre z."""
import json, os, sys
import numpy as np
code = os.path.expanduser(sys.argv[1]); out = sys.argv[2]
sys.path.insert(0, code)
import mujoco
from embody import align
from embody.body_env import BodyEnv
from embody.original import SUITES
from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv
os.environ.setdefault("R140_CLOSED", "0.7")
rows = []
for suite_name in ("libero_spatial", "libero_goal"):
    suite = SUITES[suite_name]()
    for kp in ("20", "100"):
        os.environ["R140_KP"] = kp
        task = suite.get_task(0)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        np.random.seed(7)
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False); ref.seed(0)
        env = BodyEnv(ref, bddl, "PandaR140", 256, 10, log=print, render=False)
        env.reset()
        init = int(np.random.RandomState(5478).permutation(50)[0])
        env.set_init_state(suite.get_task_init_states(0)[init])
        raw = env.env
        m, d = raw.sim.model, raw.sim.data
        r = raw.robots[0]
        gnames = list(r.gripper_joints); gi = list(r._ref_gripper_joint_pos_indexes)
        assert gnames[0] == "gripper0_finger_joint", gnames
        pads = [m.geom_name2id("gripper0_left_fingerpad_collision"), m.geom_name2id("gripper0_right_fingerpad_collision")]
        W = int(mujoco.mjtWarning.mjWARN_BADQACC)
        rec = {"suite": suite_name, "kp": kp, "joints": gnames, "q": [], "pad_z_mm": [], "pad_dist_mm": [], "contacts": 0, "qacc": 0}
        w0 = int(d._data.warning[W].number)
        for phase, g, n in (("open", -1, 120), ("close", 1, 300), ("reopen", -1, 200)):
            for _ in range(n):
                raw.step([0, 0, 0, 0, 0, 0, g])
                p_s, R_s = align.site_pose(raw.sim, r.eef_site_id)
                pc = [R_s.T @ (d.geom_xpos[p] - p_s) for p in pads]
                rec["q"].append(d.qpos[gi].tolist())
                rec["pad_z_mm"].append(float((pc[0][2] + pc[1][2]) / 2 * 1000))
                rec["pad_dist_mm"].append(float(np.linalg.norm(d.geom_xpos[pads[0]] - d.geom_xpos[pads[1]]) * 1000))
                for i in range(d.ncon):
                    c = d.contact[i]
                    b1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""; b2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
                    if b1.startswith("gripper0") != b2.startswith("gripper0") and not (b1 + b2).count("robot0"):
                        rec["contacts"] += 1; break
        rec["qacc"] = int(d._data.warning[W].number) - w0
        q = np.asarray(rec["q"])
        rec["open_fj"] = float(q[110, 0]); rec["closed_fj"] = float(q[410, 0]); rec["reopen_fj"] = float(q[-1, 0])
        rec["open_pad_z_mm"] = rec["pad_z_mm"][110]; rec["closed_pad_z_mm"] = rec["pad_z_mm"][410]
        rec["open_pad_dist_mm"] = rec["pad_dist_mm"][110]; rec["closed_pad_dist_mm"] = rec["pad_dist_mm"][410]
        rec["fj_range"] = [float(q[:, 0].min()), float(q[:, 0].max())]
        rec["closed_fj_drift_last50"] = float(np.ptp(q[370:420, 0]))
        rec["init_q"] = q[0].tolist()
        print(json.dumps({k: v for k, v in rec.items() if k not in ("q", "pad_z_mm", "pad_dist_mm")}), flush=True)
        rows.append(rec)
        env.close()
json.dump(rows, open(out, "w"))
