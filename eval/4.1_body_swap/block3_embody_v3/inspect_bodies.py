"""Static interface checks for block-3 bodies (CPU, no rendering).

For each body: joint/actuator counts, torque limits, OSC dims, hand_in_site and wrist camera pose in
the grip-site frame against Panda (plan section 3.2 items 2, 3, 6). For PandaL3 also forward-kinematics
equivalence with Panda at q3 = lock angle on 200 random joint vectors.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embody import align, robots  # noqa: E402
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


ref = make("Panda")
fr = align.frames(ref)
for body in ["Panda", "IIWA", "UR5e", "PandaL3", "Kinova3", "PandaRT", "PandaR85"]:
    e = make(body)
    r = e.robots[0]
    m = e.sim.model
    f = align.frames(e)
    tl = m.actuator_ctrlrange[r._ref_joint_actuator_indexes]
    print(f"== {body}: joints {len(r.robot_joints)} {r.robot_joints}")
    print(f"   init_qpos {np.round(r.init_qpos, 4)} | arm actuators {len(r._ref_joint_actuator_indexes)} "
          f"torque limits {tl[:, 1].tolist()} | OSC control_dim {r.controller.control_dim} "
          f"joint_dim {r.controller.joint_dim} kp {np.round(r.controller.kp, 1).tolist()}")
    print(f"   hand_in_site vs Panda {align.rot_err_deg(f['hand_in_site'], fr['hand_in_site']):.4f} deg | "
          f"wrist cam in site: pos diff {np.linalg.norm(f['wrist_cam_pos_in_site'] - fr['wrist_cam_pos_in_site']) * 1000:.3f} mm "
          f"rot diff {align.rot_err_deg(f['wrist_cam_mat_in_site'], fr['wrist_cam_mat_in_site']):.3f} deg | "
          f"cam fovy {m.cam_fovy[m.camera_name2id('robot0_eye_in_hand')]:.1f} | base pos diff "
          f"{np.linalg.norm(f['base_pos'] - fr['base_pos']) * 1000:.2f} mm | gripper joints {r.gripper_joints}")
    if body == "PandaL3":
        assert "robot0_joint3" not in r.robot_joints
        rng = np.random.RandomState(0)
        jr = ref.robots[0]
        jids = [ref.sim.model.joint_name2id(j) for j in jr.robot_joints]
        lo, hi = ref.sim.model.jnt_range[jids, 0], ref.sim.model.jnt_range[jids, 1]
        worst_p, worst_r = 0.0, 0.0
        for _ in range(200):
            q7 = lo + (hi - lo) * rng.rand(7)
            q7[2] = robots.PANDA_L3_ANGLE
            ref.sim.data.qpos[jr._ref_joint_pos_indexes] = q7
            ref.sim.forward()
            e.sim.data.qpos[r._ref_joint_pos_indexes] = np.delete(q7, 2)
            e.sim.forward()
            pa, Ra = align.site_pose(ref.sim, jr.eef_site_id)
            pb, Rb = align.site_pose(e.sim, r.eef_site_id)
            worst_p = max(worst_p, np.linalg.norm(pa - pb) * 1000)
            worst_r = max(worst_r, align.rot_err_deg(Ra, Rb))
        print(f"   FK equivalence with Panda at q3={robots.PANDA_L3_ANGLE:.6f}: max pos {worst_p:.6f} mm, max rot {worst_r:.6f} deg")
        # dynamics check: link masses/inertia unchanged
        mb = [ref.sim.model.body_mass[ref.sim.model.body_name2id(f"robot0_link{i}")] for i in range(8)]
        ma = [m.body_mass[m.body_name2id(f"robot0_link{i}")] for i in range(8)]
        print(f"   link masses equal: {np.allclose(ma, mb)}")
        ref.reset()
    e.close()
