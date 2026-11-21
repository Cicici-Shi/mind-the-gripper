"""G1 probe: can IIWA / UR5e be mounted in LIBERO, aligned to Panda, and does the
new state-transfer pipeline reproduce Panda's inputs (SRC-sham)?

Per task (original libero_spatial tasks, init state 0), after fixes on 2026-09-15:
  1. Panda via the original pipeline (env.set_init_state), 10 wait steps = reference first observation.
  2. Panda / IIWA / UR5e via the new pipeline from that settled state (objects by name,
     IK to Panda's grip_site pose, controller refs updated, quaternion sign matched).
  3. First-observation residuals, drift over 10 idle steps, frame offsets, wrist camera extrinsics,
     proprio and image differences, and an action probe (9 directions).
Writes out/probe_task{t}.json and out/probe_task{t}.png.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embody import align, robots  # noqa: E402  (robots registers classes)

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
import robosuite.utils.transform_utils as T  # noqa: E402

DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0])
WAIT = 10
RES = 256


def make_env(task, robot):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES, robots=[robot])
    env.seed(0)
    env.reset()
    return env


def proprio(obs, sign=1.0):
    q = obs["robot0_eef_quat"] * sign
    return np.concatenate([obs["robot0_eef_pos"], T.quat2axisangle(q), obs["robot0_gripper_qpos"]])


def images(obs):
    return obs["agentview_image"][::-1, ::-1], obs["robot0_eye_in_hand_image"][::-1, ::-1]


def settle(env, obs):
    for _ in range(WAIT):
        obs, _, _, _ = env.step(DUMMY)
    return obs


def summarize(f):
    return {k: np.asarray(v).round(5).tolist() for k, v in f.items()}


def residual(fa, fb):
    return {
        "site_pos_mm": float(np.linalg.norm(fa["site_pos"] - fb["site_pos"]) * 1000),
        "site_rot_deg": align.rot_err_deg(fa["site_mat"], fb["site_mat"]),
        "hand_rot_deg": align.rot_err_deg(fa["hand_mat"], fb["hand_mat"]),
        "base_pos_mm": float(np.linalg.norm(fa["base_pos"] - fb["base_pos"]) * 1000),
        "base_rot_deg": align.rot_err_deg(fa["base_mat"], fb["base_mat"]),
        "wrist_cam_pos_mm": float(np.linalg.norm(fa["wrist_cam_pos_in_site"] - fb["wrist_cam_pos_in_site"]) * 1000),
        "wrist_cam_rot_deg": align.rot_err_deg(fa["wrist_cam_mat_in_site"], fb["wrist_cam_mat_in_site"]),
    }


def action_probe(env, objs, q, g):
    """Site displacement after 10 steps of a single-axis action, from the aligned settled state."""
    out = {}
    axes = {"+x": 0, "-x": 0, "+y": 1, "-y": 1, "+z": 2, "-z": 2, "+rx": 3, "+ry": 4, "+rz": 5}
    for name, ax in axes.items():
        env.reset()
        obs, _ = align.load_state_transfer(env, objs, q, g)
        settle(env, obs)
        f0 = align.frames(env)
        a = DUMMY.copy()
        a[ax] = -0.5 if name.startswith("-") else 0.5
        for _ in range(10):
            env.step(a)
        f1 = align.frames(env)
        dp = (f1["site_pos"] - f0["site_pos"]) * 1000
        dr = T.quat2axisangle(T.mat2quat(f1["site_mat"] @ f0["site_mat"].T))
        out[name] = {"dpos_mm": dp.round(2).tolist(), "drot_deg": np.degrees(dr).round(2).tolist()}
    return out


def run(task_id, suite, outdir):
    task = suite.get_task(task_id)
    init = suite.get_task_init_states(task_id)[0]
    rec = {"task_id": task_id, "task": task.name, "bodies": {}}

    # 1. original pipeline
    env = make_env(task, "Panda")
    obs = env.set_init_state(init)
    objs = align.object_state_by_name(env)
    f_init = align.frames(env)
    ref_pre = {"proprio": proprio(obs), "imgs": images(obs)}
    obs = settle(env, obs)
    f_ref = align.frames(env)
    ref_post = {"proprio": proprio(obs), "imgs": images(obs)}
    objs_ref_post = align.object_state_by_name(env)
    rec["reference"] = {"frames_after_settle": summarize(f_ref), "eef_quat_xyzw": obs["robot0_eef_quat"].round(5).tolist()}
    env.close()

    rows = [np.concatenate(ref_post["imgs"], axis=1)]
    q_ref = ref_post_quat = obs["robot0_eef_quat"].copy()
    for arm in ["Panda", "IIWA", "UR5e"]:
        # Transfer the Panda state *after* the 10 wait steps, so the policy's first
        # observation is compared directly and body-specific settling never happens.
        env = make_env(task, arm)
        q0 = env.robots[0].init_qpos
        q = align.ik_site(env, f_ref["site_pos"], f_ref["site_mat"], q0)
        obs, missing = align.load_state_transfer(env, objs_ref_post, q, f_ref["gripper_qpos"])
        sign = align.quat_sign(obs["robot0_eef_quat"], q_ref)
        f_t0 = align.frames(env)
        t0 = {"proprio": proprio(obs, sign), "imgs": images(obs)}
        obs = settle(env, obs)
        f_hold = align.frames(env)
        b = {
            "ik_qpos": np.round(q, 5).tolist(),
            "quat_sign": sign,
            "missing_object_joints": missing,
            "residual_first_obs": residual(f_t0, f_ref),
            "drift_over_10_idle_steps": residual(f_hold, f_t0),
            "hand_in_site_offset_vs_panda_deg": align.rot_err_deg(f_t0["hand_in_site"], f_ref["hand_in_site"]),
            "proprio_diff_first_obs": (t0["proprio"] - ref_post["proprio"]).round(6).tolist(),
            "agentview_mean_abs_diff": float(np.abs(t0["imgs"][0].astype(int) - ref_post["imgs"][0].astype(int)).mean()),
            "wrist_mean_abs_diff": float(np.abs(t0["imgs"][1].astype(int) - ref_post["imgs"][1].astype(int)).mean()),
            "wrist_max_abs_diff": int(np.abs(t0["imgs"][1].astype(int) - ref_post["imgs"][1].astype(int)).max()),
            "action_probe": action_probe(env, objs_ref_post, q, f_ref["gripper_qpos"]),
        }
        rec["bodies"][arm] = b
        rows.append(np.concatenate(t0["imgs"], axis=1))
        env.close()

    Image.fromarray(np.concatenate(rows, axis=0)).save(os.path.join(outdir, f"probe_task{task_id}.png"))
    with open(os.path.join(outdir, f"probe_task{task_id}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=2, help="number of original tasks to probe")
    ap.add_argument("--out", default="out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    from embody.original import OriginalSpatialSuite
    suite = OriginalSpatialSuite()
    print("original tasks:", suite.n_tasks, [t.name for t in suite.tasks], flush=True)
    for t in range(min(args.tasks, suite.n_tasks)):
        rec = run(t, suite, args.out)
        for arm, b in rec["bodies"].items():
            print(t, arm, "first obs:", json.dumps({k: round(v, 4) for k, v in b["residual_first_obs"].items()}),
                  "drift10:", round(b["drift_over_10_idle_steps"]["site_pos_mm"], 3),
                  "proprio:", b["proprio_diff_first_obs"],
                  "img:", round(b["agentview_mean_abs_diff"], 2), round(b["wrist_mean_abs_diff"], 2), flush=True)


if __name__ == "__main__":
    main()
