"""RQ3 layer 1: replay SRC's successful action sequences on another body.

For every successful SRC episode dumped by run_libero_eval (DUMP_DIR), rebuild the same
initial scene on the target body through BodyEnv (Panda reference runs the wait steps,
the settled scene and grip-site pose are transferred), then execute the recorded actions
open loop at the same control rate. R=1 if the task's success check fires within the
recorded action budget. Physics only: no rendering is involved in success checking.

REPLAY_MODE=closed (layer 2): each step sends the recorded action plus a proportional
correction toward the reference end-effector pose of that step (position and orientation, in
the OSC_POSE delta units), so drift from open-loop execution is fed back. Gripper commands stay
as recorded. Failure here means the body could not follow Panda's hand path, not that it drifted.

embody_v3 adds diagnostics only (physics unchanged): orientation tracking error, arm torque
saturation (|torque| >= 0.99 of the actuator limit, sampled at every control substep through a wrapper
on robot.control), and stability (non-finite qpos, max arm joint speed, MuJoCo warning counters).

Usage: python -m embody.replay SRC_DUMP_DIR BODY OUT_JSONL
"""
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from libero.libero import get_libero_path
from libero.libero.envs.env_wrapper import ControlEnv

from .body_env import BodyEnv
from .original import OriginalSpatialSuite

NUM_STEPS_WAIT = 10
CLOSED = os.environ.get("REPLAY_MODE") == "closed"
POS_SCALE, ROT_SCALE = 0.05, 0.5  # OSC_POSE output_max: one unit of action = 5 cm / 0.5 rad
K_POS, K_ROT = float(os.environ.get("K_POS", "1.0")), float(os.environ.get("K_ROT", "1.0"))


def correct(a, obs, p_ref, q_ref):
    """Recorded action plus feedback toward the reference pose of the current step."""
    a = np.array(a, dtype=float)
    a[:3] += K_POS * (p_ref - obs["robot0_eef_pos"]) / POS_SCALE
    r_err = Rotation.from_quat(q_ref) * Rotation.from_quat(obs["robot0_eef_quat"]).inv()
    a[3:6] += K_ROT * r_err.as_rotvec() / ROT_SCALE
    a[:6] = np.clip(a[:6], -1, 1)
    return a


class _SatProbe:
    """Records clipped arm torques at every robot.control call (one per simulation substep)."""

    def __init__(self, env):
        self.env = env
        self.rows = []
        r = env.robots[0]
        self.lim = env.sim.model.actuator_ctrlrange[r._ref_joint_actuator_indexes, 1].copy()
        orig = r.control

        def ctl(action, policy_step=False):
            orig(action, policy_step=policy_step)
            self.rows.append(np.abs(np.asarray(r.torques, dtype=float)) >= 0.99 * self.lim)

        r.control = ctl

    def summary(self):
        if not self.rows:
            return {}
        a = np.asarray(self.rows)
        return {"sat_substep_any": float(a.any(axis=1).mean()), "sat_joint_frac": float(a.mean()),
                "sat_per_joint": [round(float(x), 4) for x in a.mean(axis=0)]}


def set_ref_scene(ref, d):
    """Write the dumped reference-scene body poses (fixtures) into the reference model, so the
    replay starts from exactly the scene the SRC episode had. Fixture placement otherwise comes
    from the global np.random stream, whose position depends on which episodes ran before."""
    m = ref.sim.model
    for b, name in enumerate(str(x) for x in d["ref_body_names"]):
        if not name:
            continue
        try:
            bid = m.body_name2id(name)
        except (ValueError, KeyError):
            continue
        m.body_pos[bid] = d["ref_body_pos"][b]
        m.body_quat[bid] = d["ref_body_quat"][b]


def main(src_dir, body, out_path):
    suite = OriginalSpatialSuite()
    done_keys = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done_keys.add((r["task"], r["init"]))
    files = sorted(glob.glob(os.path.join(src_dir, "t*_e*_i*.npz")))
    by_task = {}
    for f in files:
        d = np.load(f, allow_pickle=True)
        if bool(d["success"]):
            by_task.setdefault(int(d["task"]), []).append(f)
    out = open(out_path, "a")
    only = os.environ.get("TASKS")
    for tid in sorted(by_task):
        if only and str(tid) not in only.split(","):
            continue
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
        ref.seed(0)
        env = BodyEnv(ref, bddl, body, 256, NUM_STEPS_WAIT, log=lambda m: print(m, flush=True), render=False)
        for f in by_task[tid]:
            d = np.load(f, allow_pickle=True)
            init = int(d["init"])
            if (tid, init) in done_keys:
                continue
            actions, ref_eef, ref_q = d["action"], d["eef_pos"], d["eef_quat"]
            env.reset()
            set_ref_scene(env.ref, d)  # the SRC episode's own fixture placement, not a fresh RNG draw
            obs = env.set_init_state(inits[init])
            probe = _SatProbe(env.env)
            from .contacts import make as _make_contacts
            clog = _make_contacts(env)  # online arm-environment contacts (+ UR5e_nc forearm penetration)
            sim = env.env.sim
            r0 = env.env.robots[0]
            success, n, err, rerr, vmax, finite = False, 0, [], [], 0.0, True
            for a, p_ref, q_r in zip(actions, ref_eef, ref_q):
                err.append(float(np.linalg.norm(obs["robot0_eef_pos"] - p_ref)) * 1000)
                rerr.append(float(np.degrees((Rotation.from_quat(q_r) * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).magnitude())))
                if CLOSED:
                    a = correct(a, obs, p_ref, q_r)
                obs, _, done, _ = env.step(a.tolist())
                clog.update(a)
                n += 1
                finite = finite and bool(np.isfinite(sim.data.qpos).all())
                vmax = max(vmax, float(np.abs(sim.data.qvel[r0._ref_joint_vel_indexes]).max()))
                if done:
                    success = True
                    break
            rec = {"task": tid, "init": init, "ep": int(d["ep"]), "body": body, "mode": "closed" if CLOSED else "open", "replay_success": success,
                   "steps": n, "src_steps": int(len(actions)),
                   "track_err_mm_mean": float(np.mean(err)), "track_err_mm_max": float(np.max(err)),
                   "track_rot_deg_mean": float(np.mean(rerr)), "track_rot_deg_max": float(np.max(rerr)),
                   "align_pos_mm": env.align_info["pos_mm"], "align_rot_deg": env.align_info["rot_deg"],
                   "fixture_mm": env.align_info["fixture_mm"], "finite": finite, "qvel_arm_max": vmax,
                   "mj_warnings": int(sum(int(w.number) for w in sim.data._data.warning)), **probe.summary(), "contacts": clog.summary()}
            out.write(json.dumps(rec) + "\n")
            out.flush()
            print(rec, flush=True)
        env.close()


if __name__ == "__main__":
    main(*sys.argv[1:4])
