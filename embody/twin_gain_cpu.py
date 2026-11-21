"""Choose the TWIN_RENDER nullspace gain without rendering (CPU node).

Replays SRC actions on a physical body (BodyEnv, render=False) and, for several gains, keeps a
separate warm-started Panda twin IK chain. Reports pose error, misses (pos > 2 mm or rot > 1 deg)
and, on the Panda body, posture gap |q_twin - q_physical| (max over joints, degrees).
Usage: python -m embody.twin_gain_cpu SRC_DUMP_DIR BODY N_EPISODES
"""
import glob
import os
import sys

import numpy as np

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from . import align
from .body_env import BodyEnv
from .original import OriginalSpatialSuite
from .replay import set_ref_scene

GAINS = [0.0, 0.05, 0.1, 0.2, 0.5]


def main(src_dir, body, n_eps):
    suite = OriginalSpatialSuite()
    files = []
    for f in sorted(glob.glob(os.path.join(src_dir, "t*_e*_i*.npz"))):
        z = np.load(f, allow_pickle=True)
        if bool(z["success"]) and int(z["ep"]) % 7 == 0:
            files.append(f)
    files = files[: int(n_eps)]
    stats = {g: {"pos": [], "rot": [], "post": []} for g in GAINS}
    for f in files:
        d = np.load(f, allow_pickle=True)
        tid, init = int(d["task"]), int(d["init"])
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        ref.seed(0)
        env = BodyEnv(ref, bddl, body, 256, 10, log=lambda m: None, render=False)
        twin = ControlEnv(bddl_file_name=bddl, robots=["Panda"], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        twin.seed(0)
        env.reset()
        twin.reset()
        set_ref_scene(env.ref, d)
        obs = env.set_init_state(suite.get_task_init_states(tid)[init])
        q_start = env.ref.sim.data.qpos[env.ref.robots[0]._ref_joint_pos_indexes].copy()
        qs = {g: q_start.copy() for g in GAINS}
        for a in d["action"]:
            p, R = align.site_pose(env.env.sim, env.env.robots[0].eef_site_id)
            for g in GAINS:
                q, ep, er = align.ik_site_ns(twin, p, R, qs[g], q_start, gain=g)
                qs[g] = q
                stats[g]["pos"].append(ep * 1000)
                stats[g]["rot"].append(np.degrees(er))
                if body == "Panda":
                    q_phys = env.env.sim.data.qpos[env.env.robots[0]._ref_joint_pos_indexes]
                    stats[g]["post"].append(float(np.degrees(np.max(np.abs(q - q_phys)))))
            obs, _, done, _ = env.step(a.tolist())
            if done:
                break
        env.close()
        twin.close()
        print(f"done {os.path.basename(f)}", flush=True)
    for g in GAINS:
        s = stats[g]
        pos, rot = np.array(s["pos"]), np.array(s["rot"])
        miss = int(((pos > 2) | (rot > 1)).sum())
        line = (f"{body} gain={g:<5} steps={len(pos)} pos_mm median={np.median(pos):.5f} p99={np.percentile(pos, 99):.4f} "
                f"max={pos.max():.4f} rot_deg max={rot.max():.4f} misses={miss}")
        if s["post"]:
            line += f" posture_deg p50={np.median(s['post']):.2f} p95={np.percentile(s['post'], 95):.2f} max={max(s['post']):.2f}"
        print(line, flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:4])
