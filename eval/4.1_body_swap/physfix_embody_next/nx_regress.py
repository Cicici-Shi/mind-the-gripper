"""embody_next copy of embody_toy/tools/regress_img.py (2026-11-06): adds --suite (libero_spatial | libero_object, the
suite classes are taken from the --code dir's embody.original), never reads or prints success / done, and does not stop
at done (the dump's action list already ends where the eval stopped). Otherwise unchanged:

TOY1 copy of embody_v3/tools/regress.py (2026-11-03): also hashes the policy images (agentview, wrist) of every
step (index 0 = after set_init_state) and stores them, plus BodyEnv.toy_stats when present, so two code dirs can be
compared pixel by pixel (tools/regress_img_cmp.py). --save-img N keeps the raw images of the first N steps.

Replay dumped episodes of an existing condition with a given embody code dir (no policy).

Mimics run_libero_eval: one reference env per task (OffScreenRenderEnv when --render 1, else a
physics-only ControlEnv), episodes of the task in order, BodyEnv.reset (RNG window) before each.
After reset the dumped reference fixtures are written into the reference model (as replay.py and
_contacts.py do), then set_init_state, then the dumped actions step by step.

Per episode it writes one jsonl line and an npz with the replayed qpos trajectory:
  steps, qpos_max_diff vs the dumped qpos (same array layout), first step
  where it exceeds 1e-9, fixture_mm (target static bodies vs dumped reference, by name), natural_fixture_mm
  (reference fixtures drawn by the RNG before overwriting, vs dump), align info, twin stats when present.

Usage: python tools/regress.py --code DIR --cond COND --body BODY|SRC --tasks 0,1 --eps 0-4 --out OUTDIR [--render 1]
Environment variables of the original run (HIDE_ARM, RECOLOR_ARM, GRP_KP, TWIN_RENDER, ...) must be set by the caller.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--cond", required=True)
ap.add_argument("--body", required=True)
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--eps", default="0-4")
ap.add_argument("--out", required=True)
ap.add_argument("--render", type=int, default=0)
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--tag", default="")
ap.add_argument("--suite", default="libero_spatial", choices=["libero_spatial", "libero_object"])
ap.add_argument("--save-img", type=int, default=0)
ap.add_argument("--contacts", type=int, default=0, help="2026-10-31: attach embody.contacts.ContactLog during the replay "
                "(as the eval does) to show it does not change the physics; its summary goes into the jsonl")
args = ap.parse_args()

sys.path.insert(0, os.path.expanduser(args.code))
import embody.original as _orig  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402

import embody  # noqa: E402
assert os.path.dirname(os.path.abspath(embody.__file__)).startswith(os.path.abspath(os.path.expanduser(args.code))), embody.__file__

import random  # noqa: E402
np.random.seed(7)  # run_libero_eval: set_seed_everywhere(7) once, then all tasks in order
random.seed(7)
a, b = (int(x) for x in args.eps.split("-"))
EPS = list(range(a, b + 1))
os.makedirs(args.out, exist_ok=True)
suite = {"libero_spatial": _orig.OriginalSpatialSuite, "libero_object": _orig.OriginalObjectSuite}[args.suite]()
DUMMY = [0, 0, 0, 0, 0, 0, -1]
logs = []


def fixtures_vs_dump(model, z, names_filter=None):
    names = [str(x) for x in z["ref_body_names"]]
    worst = 0.0
    for i, n in enumerate(names):
        if not n or n == "world" or n.startswith(("robot0", "gripper0", "mount0")):
            continue
        try:
            bid = model.body_name2id(n)
        except (ValueError, KeyError):
            continue
        if names_filter is not None and n not in names_filter:
            continue
        worst = max(worst, float(np.abs(model.body_pos[bid] - z["ref_body_pos"][i]).max()) * 1000)
    return worst


for tid in [int(t) for t in args.tasks.split(",")]:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if args.render:
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    else:
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    if args.body != "SRC":
        from embody.body_env import BodyEnv
        env = BodyEnv(ref, bddl, args.body, 256, 10, log=lambda m: logs.append(m), render=bool(args.render))
    else:
        env = ref
    for ep in EPS:
        fs = glob.glob(os.path.join(args.runs, args.cond, f"t{tid}_e{ep}_i*.npz"))
        if not fs:
            print("missing dump", args.cond, tid, ep, flush=True)
            continue
        z = np.load(fs[0], allow_pickle=True)
        init = int(z["init"])
        env.reset()
        rm = (env.ref if env is not ref else ref).sim.model
        natural = fixtures_vs_dump(rm, z)
        names = [str(x) for x in z["ref_body_names"]]
        for i, n in enumerate(names):
            if not n:
                continue
            try:
                bid = rm.body_name2id(n)
            except (ValueError, KeyError):
                continue
            rm.body_pos[bid] = z["ref_body_pos"][i]
            rm.body_quat[bid] = z["ref_body_quat"][i]
        logs.clear()
        obs = env.set_init_state(inits[init])
        if env is ref:
            for _ in range(10):
                obs, _, _, _ = env.step(DUMMY)
            fix_mm = 0.0
            static = None
        else:
            from embody import align
            static = set(align.static_body_pose(env.ref).keys())
            fix_mm = fixtures_vs_dump(env.env.sim.model, z, static)
        sim = (env.env if env is not ref else ref).sim
        qd = z["qpos"]
        import hashlib
        def _h(o):
            return [hashlib.md5(np.ascontiguousarray(o[k]).tobytes()).hexdigest() for k in ("agentview_image", "robot0_eye_in_hand_image")] if args.render else []
        hashes, imgs = [_h(obs)], []
        if args.render and args.save_img:
            imgs.append(np.stack([obs["agentview_image"], obs["robot0_eye_in_hand_image"]]))
        traj = []
        n = 0
        clog = None
        for k, act in enumerate(z["action"]):
            traj.append(sim.data.qpos.copy())
            if args.contacts and clog is None:
                from embody import contacts as _contacts
                clog = _contacts.make(env)
            obs, _, _, _ = env.step(act.tolist())
            hashes.append(_h(obs))
            if args.render and len(imgs) < args.save_img:
                imgs.append(np.stack([obs["agentview_image"], obs["robot0_eye_in_hand_image"]]))
            if clog is not None:
                clog.update(act)
            n += 1
        traj = np.asarray(traj)
        m = min(len(traj), len(qd))
        dq = np.abs(traj[:m] - qd[:m]).max(axis=1) if m else np.zeros(0)
        bad = np.nonzero(dq > 1e-9)[0]
        rec = {"cond": args.cond, "code": args.code, "tag": args.tag, "task": tid, "ep": ep, "init": init,
               "steps": n,
               "dump_steps": int(len(z["action"])), "qpos_max_diff": float(dq.max()) if m else None,
               "first_diverge": int(bad[0]) if len(bad) else None,
               "fixture_mm": fix_mm, "natural_fixture_mm": natural,
               "align": getattr(env, "align_info", None), "log": [l for l in logs if "BODY_ALIGN" in l or "TWIN" in l][:3]}
        if clog is not None:
            rec["contacts"] = clog.summary()
        ts = getattr(env, "twin_stats", None)
        save = {"qpos": traj, "img_md5": np.asarray(hashes)}
        if imgs:
            save["imgs"] = np.asarray(imgs, dtype=np.uint8)
        toy = getattr(env, "toy_stats", None) if env is not ref else None
        if toy and toy.get("px"):
            save["toy_px"] = np.asarray(toy["px"])
        if ts:
            pe, re_ = np.asarray(ts["pos_err_mm"]), np.asarray(ts["rot_err_deg"])
            rec["twin"] = {"n": int(len(pe)), "pos_max": float(pe.max()), "rot_max": float(re_.max()),
                           "qmiss": int(((pe[::8] > 2) | (re_[::8] > 1)).sum()), "nq": int(len(pe[::8])),
                           "miss": int(((pe > 2) | (re_ > 1)).sum()), "stale": int(ts["stale"]), "t0": ts["t0"]}
            save.update(twin_pos=pe, twin_rot=re_)
            if ts.get("root_off_mm"):
                save.update(twin_root_off=np.asarray(ts["root_off_mm"]))
            if "twin_pos_err_mm" in z:
                dp = z["twin_pos_err_mm"]
                mm = min(len(dp), len(pe))
                rec["twin"]["vs_dump_pos_max_diff"] = float(np.abs(dp[:mm] - pe[:mm].astype(np.float32)).max()) if mm else None
                rec["twin"]["dump_t0"] = z["twin_t0"].tolist()
        np.savez_compressed(os.path.join(args.out, f"t{tid}_e{ep}.npz"), **save)
        with open(os.path.join(args.out, "result.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps({k: rec[k] for k in ("task", "ep", "steps", "qpos_max_diff", "first_diverge", "fixture_mm", "natural_fixture_mm")}), flush=True)
    pass  # envs are not closed between tasks, as in run_libero_eval (closing + new EGL context segfaulted on node03)
