"""Block-3 smoke test for one new condition: replay SRC dumps (no policy) through embody_v3 BodyEnv.

Per episode: first-frame alignment (grip-site pos/rot vs the Panda reference, fixture_mm, wrist camera
and hand_in_site in the site frame), closed-loop replay of the SRC actions (same correction as
embody.replay REPLAY_MODE=closed) with success, stability (non-finite qpos, max arm joint speed, MuJoCo
warnings), twin IK misses on policy-query frames (every 8th paint, > 2 mm or > 1 deg), twin step-0
image difference against the reference Panda, and for G_R the twin-vs-physical wrist camera pose.
Screenshots (policy view, agentview | wrist, rotated 180 deg as the policies see them): step 0 and the
first step after the recorded gripper command closes (or step 60), for the first episode only.

Usage: python tools/smoke.py --name G_R --body Panda --tasks 0,1,...,9 --ep 0 --out DIR [--render 1]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
from scipy.spatial.transform import Rotation

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embody import align  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from embody.replay import correct, set_ref_scene  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--body", required=True)
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--ep", default="0")
ap.add_argument("--out", required=True)
ap.add_argument("--render", type=int, default=1)
ap.add_argument("--shots", default=os.path.expanduser("~/tmp/v3_smoke"))
ap.add_argument("--nshots", type=int, default=1, help="episodes with screenshots (step 0 and grasp)")
ap.add_argument("--save-t0", default="", help="dir: save the step-0 policy images per episode (npz)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
os.makedirs(args.shots, exist_ok=True)
suite = OriginalSpatialSuite()
SRC = os.path.expanduser("$PROJ_ROOT/runs/src")
shots_done = 0


def shot(obs, path):
    from PIL import Image
    a = np.asarray(obs["agentview_image"])[::-1, ::-1]
    w = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
    Image.fromarray(np.concatenate([a, w], axis=1).astype(np.uint8)).save(path)


for tid in [int(t) for t in args.tasks.split(",")]:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if args.render:
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    else:
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    logs = []
    env = BodyEnv(ref, bddl, args.body, 256, 10, log=logs.append, render=bool(args.render))
    for ep in [int(e) for e in args.ep.split(",")]:
        f = glob.glob(os.path.join(SRC, f"t{tid}_e{ep}_i*.npz"))[0]
        d = np.load(f, allow_pickle=True)
        env.reset()
        set_ref_scene(env.ref, d)
        logs.clear()
        obs = env.set_init_state(inits[int(d["init"])])
        static = set(align.static_body_pose(env.ref).keys())
        names = [str(x) for x in d["ref_body_names"]]
        tm = env.env.sim.model
        fix = max(float(np.abs(tm.body_pos[tm.body_name2id(n)] - d["ref_body_pos"][i]).max()) * 1000
                  for i, n in enumerate(names) if n in static)
        sim = env.env.sim
        r0 = env.env.robots[0]
        cam_err = []
        take = None
        if args.save_t0:
            os.makedirs(args.save_t0, exist_ok=True)
            np.savez_compressed(os.path.join(args.save_t0, f"t{tid}_e{ep}.npz"), agent=obs["agentview_image"],
                                wrist=obs["robot0_eye_in_hand_image"])
        tagp = "" if shots_done == 0 else f"_t{tid}e{ep}"
        if args.render and shots_done < args.nshots:
            shot(obs, os.path.join(args.shots, f"{args.name}{tagp}_1_step0.png"))
            closes = np.nonzero(d["action"][:, -1] > 0)[0]
            take = int(closes[0]) + 3 if len(closes) else 60
        success, n, err, rerr, vmax, finite = False, 0, [], [], 0.0, True
        from embody.contacts import make as _mk
        clog = _mk(env)
        for k, (a, p_ref, q_r) in enumerate(zip(d["action"], d["eef_pos"], d["eef_quat"])):
            err.append(float(np.linalg.norm(obs["robot0_eef_pos"] - p_ref)) * 1000)
            rerr.append(float(np.degrees((Rotation.from_quat(q_r) * Rotation.from_quat(obs["robot0_eef_quat"]).inv()).magnitude())))
            ac = correct(a, obs, p_ref, q_r)
            obs, _, done, _ = env.step(ac.tolist())
            clog.update(ac)
            n += 1
            finite = finite and bool(np.isfinite(sim.data.qpos).all())
            vmax = max(vmax, float(np.abs(sim.data.qvel[r0._ref_joint_vel_indexes]).max()))
            if env.twin is not None and getattr(env, "twin_seed_phys", False):
                ct = env.twin.sim.model.camera_name2id("robot0_eye_in_hand")
                cp = sim.model.camera_name2id("robot0_eye_in_hand")
                cam_err.append(float(np.linalg.norm(env.twin.sim.data.cam_xpos[ct] - sim.data.cam_xpos[cp]) * 1000))
            if take is not None and k + 1 == take:
                shot(obs, os.path.join(args.shots, f"{args.name}{tagp}_2_step{take}.png"))
            if done:
                success = True
                break
        if take is not None:
            if n < take:
                shot(obs, os.path.join(args.shots, f"{args.name}{tagp}_2_step{n}_end.png"))
            shots_done += 1
        rec = {"name": args.name, "body": args.body, "task": tid, "ep": ep, "init": int(d["init"]),
               "replay_success": success, "steps": n, "src_steps": int(len(d["action"])),
               "align": env.align_info, "fixture_vs_src_mm": fix,
               "track_err_mm_mean": float(np.mean(err)), "track_rot_deg_mean": float(np.mean(rerr)),
               "finite": finite, "qvel_arm_max": vmax,
               "mj_warnings": int(sum(int(w.number) for w in sim.data._data.warning)),
               "contacts": clog.summary(), "log": [l for l in logs if "BODY_ALIGN" in l or "TWIN_T0" in l]}
        ts = getattr(env, "twin_stats", None)
        if ts:
            pe, re_ = np.asarray(ts["pos_err_mm"]), np.asarray(ts["rot_err_deg"])
            rec["twin"] = {"n": int(len(pe)), "nq": int(len(pe[::8])),
                           "qmiss": int(((pe[::8] > 2) | (re_[::8] > 1)).sum()),
                           "miss": int(((pe > 2) | (re_ > 1)).sum()), "pos_max": float(pe.max()),
                           "rot_max": float(re_.max()), "stale": int(ts["stale"]), "t0": ts["t0"],
                           "retries": int(ts.get("retries", 0)),
                           "over_1mm_q": int((pe[::8] > 1).sum()), "pos_q_max": float(pe[::8].max()),
                           "posture_p95": float(np.percentile(ts["posture_deg"], 95)) if ts["posture_deg"] else None,
                           "cam_err_mm_max": float(max(cam_err)) if cam_err else None}
        with open(os.path.join(args.out, f"{args.name}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("task", "ep", "replay_success", "fixture_vs_src_mm", "finite")}), rec["align"], rec.get("twin"), flush=True)
    env.close()
