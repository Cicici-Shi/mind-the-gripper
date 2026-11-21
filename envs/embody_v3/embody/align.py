"""Frames, IK alignment and state transfer between bodies.

Conventions verified in robosuite 1.4.1 SingleArm:
  robot0_eef_pos  = world position of the gripper's grip_site
  robot0_eef_quat = world orientation of robot_model.eef_name (arm hand body), xyzw
  OSC controls grip_site pose.
"""
import mujoco
import numpy as np
import robosuite.utils.transform_utils as T


def _raw(sim):
    return sim.model._model, sim.data._data


def site_pose(sim, site_id):
    return sim.data.site_xpos[site_id].copy(), sim.data.site_xmat[site_id].reshape(3, 3).copy()


def body_mat(sim, name):
    return sim.data.get_body_xmat(name).reshape(3, 3).copy()


def rot_err_deg(Ra, Rb):
    c = (np.trace(Ra.T @ Rb) - 1) / 2
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def frames(env):
    """Everything needed to compare interfaces across bodies, in world frame."""
    robot = env.robots[0]
    sim = env.sim
    p_site, R_site = site_pose(sim, robot.eef_site_id)
    R_hand = body_mat(sim, robot.robot_model.eef_name)
    cam = sim.model.camera_name2id("robot0_eye_in_hand")
    cam_pos = sim.data.cam_xpos[cam].copy()
    cam_mat = sim.data.cam_xmat[cam].reshape(3, 3).copy()
    return {
        "wrist_cam_pos_in_site": R_site.T @ (cam_pos - p_site),
        "wrist_cam_mat_in_site": R_site.T @ cam_mat,
        "base_pos": sim.data.get_body_xpos(robot.robot_model.root_body).copy(),
        "base_mat": body_mat(sim, robot.robot_model.root_body),
        "site_pos": p_site,
        "site_mat": R_site,
        "hand_mat": R_hand,
        # fixed offset between the orientation the policy sees and the one OSC controls
        "hand_in_site": R_site.T @ R_hand,
        "arm_qpos": sim.data.qpos[robot._ref_joint_pos_indexes].copy(),
        "gripper_qpos": sim.data.qpos[robot._ref_gripper_joint_pos_indexes].copy(),
    }


def ik_site(env, p_target, R_target, q0, iters=400, damping=1e-3, tol_pos=1e-5, tol_rot=1e-4):
    """Damped least squares on grip_site pose, arm joints only, joint limits clipped.
    Deterministic: always starts from q0 (the robot's default init_qpos)."""
    robot = env.robots[0]
    sim = env.sim
    m, d = _raw(sim)
    qpos_idx = np.array(robot._ref_joint_pos_indexes)
    dof_idx = np.array(robot._ref_joint_vel_indexes)
    jnt_ids = [sim.model.joint_name2id(j) for j in robot.robot_joints]
    lo, hi = m.jnt_range[jnt_ids, 0], m.jnt_range[jnt_ids, 1]
    limited = m.jnt_limited[jnt_ids].astype(bool)
    saved = d.qpos.copy()
    q = np.array(q0, dtype=float)
    jacp = np.zeros((3, m.nv))
    jacr = np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qpos_idx] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        p, R = site_pose(sim, robot.eef_site_id)
        e_p = p_target - p
        e_r = T.quat2axisangle(T.mat2quat(R_target @ R.T))
        if np.linalg.norm(e_p) < tol_pos and np.linalg.norm(e_r) < tol_rot:
            break
        mujoco.mj_jacSite(m, d, jacp, jacr, robot.eef_site_id)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        e = np.concatenate([e_p, e_r])
        dq = J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), e)
        q = q + dq
        q[limited] = np.clip(q[limited], lo[limited], hi[limited])
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return q


def quat_sign(q, q_ref):
    """+1 or -1 so that sign * q lies in the same hemisphere as q_ref.
    UR5e's hand body quaternion comes out as -q of Panda's for the same rotation,
    which becomes a 2*pi jump after axis-angle conversion. Fixed per episode:
    MuJoCo body quaternions are continuous along a trajectory."""
    return 1.0 if float(np.dot(q, q_ref)) >= 0 else -1.0


def object_state_by_name(env):
    """Non-robot joints (objects, fixtures) as {joint_name: (qpos, qvel)}."""
    robot = env.robots[0]
    own = set(robot.robot_joints) | set(robot.gripper_joints)
    m = env.sim.model
    out = {}
    for j in range(m.njnt):
        name = m.joint_id2name(j)
        if name is None or name in own:
            continue
        qa, va = m.get_joint_qpos_addr(name), m.get_joint_qvel_addr(name)
        qs = slice(*qa) if isinstance(qa, tuple) else slice(qa, qa + 1)
        vs = slice(*va) if isinstance(va, tuple) else slice(va, va + 1)
        out[name] = (env.sim.data.qpos[qs].copy(), env.sim.data.qvel[vs].copy())
    return out


def static_body_pose(env):
    """World pose of bodies that have no joints: the fixtures (cabinet, stove, ...).

    LIBERO places these at reset by writing model.body_pos/body_quat, drawing from the
    env's RNG. A body with a different joint count consumes that RNG differently, so the
    cabinet landed 6.9 mm and the stove 6.1 mm away under UR5e. They must be copied over
    or the arm effect is measured against a slightly different scene.
    """
    m = env.sim.model
    jointed = set()
    for j in range(m.njnt):
        jointed.add(int(m.jnt_bodyid[j]))
    robot = env.robots[0]
    own = ("robot0", "gripper0", "mount0")
    out = {}
    for b in range(m.nbody):
        name = m.body_id2name(b)
        if name is None or b in jointed or name.startswith(own) or name in ("world",):
            continue
        # only root-level placements matter; children follow their parent
        out[name] = (m.body_pos[b].copy(), m.body_quat[b].copy())
    del robot
    return out


def apply_static_body_pose(env, poses):
    """Write fixture placements into the target model. Static bodies, so this fully
    determines their world pose after mj_forward."""
    m = env.sim.model
    missing, moved_mm = [], 0.0
    for name, (pos, quat) in poses.items():
        try:
            b = m.body_name2id(name)
        except (ValueError, KeyError):
            missing.append(name)
            continue
        moved_mm = max(moved_mm, float(np.linalg.norm(m.body_pos[b] - pos) * 1000))
        m.body_pos[b] = pos
        m.body_quat[b] = quat
    return missing, moved_mm


def apply_object_state(env, objs):
    m = env.sim.model
    missing = []
    for name, (qp, qv) in objs.items():
        try:
            qa, va = m.get_joint_qpos_addr(name), m.get_joint_qvel_addr(name)
        except (ValueError, KeyError):
            missing.append(name)
            continue
        qs = slice(*qa) if isinstance(qa, tuple) else slice(qa, qa + 1)
        vs = slice(*va) if isinstance(va, tuple) else slice(va, va + 1)
        env.sim.data.qpos[qs] = qp
        env.sim.data.qvel[vs] = qv
    return missing


def load_state_transfer(env, objs, arm_qpos, gripper_qpos, fixtures=None):
    """Cross-body replacement for env.set_init_state(flat_state):
    fixtures placed by body name, objects copied by joint name, robot joints set
    explicitly, zero robot velocity."""
    robot = env.robots[0]
    if fixtures:
        apply_static_body_pose(env, fixtures)
    missing = apply_object_state(env, objs)
    env.sim.data.qpos[robot._ref_joint_pos_indexes] = arm_qpos
    env.sim.data.qvel[robot._ref_joint_vel_indexes] = 0.0
    if gripper_qpos is not None:  # None: target has a different gripper, handled by the caller
        env.sim.data.qpos[robot._ref_gripper_joint_pos_indexes] = gripper_qpos
    env.sim.forward()
    # OSC's nullspace term pulls toward controller.initial_joint, which is the model's
    # default init_qpos. Without this the new arms drift ~40 mm in 10 idle steps.
    robot.controller.update(force=True)
    robot.controller.update_initial_joints(np.asarray(arm_qpos))
    env.check_success()
    env._post_process()
    env._update_observables(force=True)
    return env.env._get_observations(), missing


def ik_site_ns(env, p_target, R_target, q0, q_rest, gain=0.1, iters=400, damping=1e-3,
               tol_pos=1e-5, tol_rot=1e-4):
    """Grip-site IK for the TWIN_RENDER posture only; ik_site (used by the body transfer) is unchanged.

    Per control step: solve the pose from the previous solution (warm start), take ONE nullspace
    step of size `gain` toward q_rest (exact pseudo-inverse projector, so the pose is not disturbed
    to first order), then polish the pose again. `gain` is the fraction of the posture gap closed
    per step, the kinematic analogue of OSC's nullspace pull toward the episode-start joints.
    Returns (q, pos_err_m, rot_err_rad) after the polish.
    """
    robot = env.robots[0]
    sim = env.sim
    m, d = _raw(sim)
    qpos_idx = np.array(robot._ref_joint_pos_indexes)
    dof_idx = np.array(robot._ref_joint_vel_indexes)
    jnt_ids = [sim.model.joint_name2id(j) for j in robot.robot_joints]
    lo, hi = m.jnt_range[jnt_ids, 0], m.jnt_range[jnt_ids, 1]
    limited = m.jnt_limited[jnt_ids].astype(bool)
    kw = dict(iters=iters, damping=damping, tol_pos=tol_pos, tol_rot=tol_rot)
    q = ik_site(env, p_target, R_target, q0, **kw)
    if gain > 0:
        saved = d.qpos.copy()
        d.qpos[qpos_idx] = q
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        mujoco.mj_jacSite(m, d, jacp, jacr, robot.eef_site_id)
        J = np.vstack([jacp[:, dof_idx], jacr[:, dof_idx]])
        q = q + gain * (np.eye(len(q)) - np.linalg.pinv(J) @ J) @ (np.asarray(q_rest, dtype=float) - q)
        q[limited] = np.clip(q[limited], lo[limited], hi[limited])
        d.qpos[:] = saved
        mujoco.mj_forward(m, d)
        q = ik_site(env, p_target, R_target, q, **kw)
    saved = d.qpos.copy()
    d.qpos[qpos_idx] = q
    mujoco.mj_kinematics(m, d)
    p, R = site_pose(sim, robot.eef_site_id)
    e_p = p_target - p
    e_r = T.quat2axisangle(T.mat2quat(R_target @ R.T))
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))
