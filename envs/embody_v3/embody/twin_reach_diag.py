"""Why can the Panda twin not match some UR5e grip-site poses? (CPU, no rendering)

For ur5e_twin dumps, rebuild the physical UR5e grip-site pose at steps whose twin IK missed
(> 2 mm), then try on a Panda: (a) full pose IK from 50 random in-range seeds, (b) position-only IK.
Reports reachability and which Panda joints sit at a limit in the best full-pose solution, and the
UR5e gripper's yaw/tilt relative to the Panda's SRC distribution.
Usage: python -m embody.twin_reach_diag RUNS_DIR N_FRAMES
"""
import glob
import os
import sys

import mujoco
import numpy as np
import robosuite.utils.transform_utils as T

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from . import align, robots  # noqa: F401  registers MountedIIWA / MountedUR5e
from .original import OriginalSpatialSuite


def pos_only_ik(env, p_target, q0, iters=600, damping=1e-3):
    robot = env.robots[0]
    sim = env.sim
    m, d = align._raw(sim)
    qi = np.array(robot._ref_joint_pos_indexes)
    di = np.array(robot._ref_joint_vel_indexes)
    jid = [sim.model.joint_name2id(j) for j in robot.robot_joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    saved = d.qpos.copy()
    q = np.array(q0, float)
    jacp = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qi] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, _ = align.site_pose(sim, robot.eef_site_id)
        e = p_target - p
        if np.linalg.norm(e) < 1e-5:
            break
        mujoco.mj_jacSite(m, d, jacp, None, robot.eef_site_id)
        J = jacp[:, di]
        q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + damping * np.eye(3), e), lo, hi)
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return q, float(np.linalg.norm(e))


def main(runs, n_frames):
    n_frames = int(n_frames)
    suite = OriginalSpatialSuite()
    rng = np.random.default_rng(0)
    rows = []
    for f in sorted(glob.glob(os.path.join(runs, "ur5e_twin", "*.npz"))):
        z = np.load(f, allow_pickle=True)
        pe = z["twin_pos_err_mm"]
        bad = [i for i in np.nonzero(pe > 2)[0] if i < len(z["qpos"])]
        if not bad:
            continue
        tid = int(z["task"])
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        ur = ControlEnv(bddl_file_name=bddl, robots=["UR5e"], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        pa = ControlEnv(bddl_file_name=bddl, robots=["Panda"], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        ur.reset(); pa.reset()
        names = [str(x) for x in z["joint_names"]]
        uq = [names.index(f"robot0_{j}") if f"robot0_{j}" in names else None for j in []]  # placeholder
        arm_idx = [i for i, n in enumerate(names) if n.startswith("robot0_")]
        r = ur.robots[0]
        jid = [pa.sim.model.joint_name2id(j) for j in pa.robots[0].robot_joints]
        lo, hi = pa.sim.model.jnt_range[jid, 0], pa.sim.model.jnt_range[jid, 1]
        for i in bad[:: max(1, len(bad) // 6)]:
            ur.sim.data.qpos[r._ref_joint_pos_indexes] = z["qpos"][i][arm_idx[: len(r._ref_joint_pos_indexes)]]
            ur.sim.forward()
            p, R = align.site_pose(ur.sim, r.eef_site_id)
            best = (1e9, None)
            for _ in range(50):
                q0 = rng.uniform(lo, hi)
                q = align.ik_site(pa, p, R, q0, iters=600)
                pa.sim.data.qpos[pa.robots[0]._ref_joint_pos_indexes] = q
                pa.sim.forward()
                pp, RR = align.site_pose(pa.sim, pa.robots[0].eef_site_id)
                err = np.linalg.norm(pp - p) * 1000 + align.rot_err_deg(RR, R)
                if err < best[0]:
                    best = (err, q, np.linalg.norm(pp - p) * 1000, align.rot_err_deg(RR, R))
            qpos_only, e_pos = pos_only_ik(pa, p, best[1])
            at_lim = [k + 1 for k in range(7) if min(best[1][k] - lo[k], hi[k] - best[1][k]) < np.radians(1)]
            z_axis = R[:, 2]
            yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
            tilt = np.degrees(np.arccos(np.clip(-z_axis[2], -1, 1)))
            rows.append((os.path.basename(f), i, best[2], best[3], e_pos * 1000, at_lim, yaw, tilt, p))
            print(f"{os.path.basename(f)} step {i}: full-pose best pos {best[2]:.1f} mm rot {best[3]:.2f} deg | "
                  f"position-only {e_pos*1000:.2f} mm | Panda joints at limit {at_lim} | UR5e grip yaw {yaw:.0f} tilt {tilt:.0f} | p {np.round(p,3)}", flush=True)
            if len(rows) >= n_frames:
                return
        ur.close(); pa.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
