"""embody_next (2026-11-06): interface and physics check of BodyEnv on LIBERO goal / LIBERO-10 (no policy).

One process = one eval shard (EP_OFFSET = --offset, --n episodes per task, all tasks in order), replaying the RNG path
of run_libero_eval_g10.py: np.random.seed(7) / random.seed(7) once, per task a reference env (OffScreenRenderEnv with
--render 1, physics-only ControlEnv otherwise) with seed(0), BodyEnv (or the plain env for --body SRC), and per episode
env.reset() before anything else. Per key (task, ep):

  scene  init id (RandomState(5478 + task).permutation(50)[ep]) vs the SRC dump and scene_index; the reference's
         naturally drawn static bodies vs the SRC dump (natural_mm) and vs scene_index fixtures (natural_si_mm); then the
         dumped placements are written into the reference (as tools/regress.py) and set_init_state runs; target static
         bodies vs dump (fixture_mm); every non-robot joint of the target vs the SRC dump's first recorded qpos, by name
         (free joints: position mm, quaternion abs; hinge / slide: abs value, drawers and doors); robot root body world
         position vs the reference Panda's (base_mm); BodyEnv.align_info; robot-vs-scene contacts right after the
         transfer (init_pen_mm = deepest penetration, pairs).
  physics for each --modes entry: replay = the SRC dump's actions (OFT on native Panda), zero = --zero-steps x
         [0,0,0,0,0,0,-1] (hold pose, gripper open). Per control step: robot joint qpos, MuJoCo bad-QACC warnings and
         sim-time resets. Detector D of vla_zoo/tools/physics_breakdown_scan.py (imported unchanged, sha256 c1d960f7...):
         arm |dq| > pi/2 between steps or beyond the Panda limits by > 0.1 rad; Robotiq finger_joint outside [-0.5, 1.2].
         Exact definition only for Panda-arm bodies; for other arms the same rule with the arm's own joint ranges (D_own),
         and for slide-joint grippers (Panda, Rethink) fingers beyond their own range by > 10 mm (B_slide) -- both are
         analogues, labelled as such. ContactLog (embody.contacts, read-only) gives arm-scene contact steps and pairs.
         The episode is NOT stopped at done and no success / done flag is recorded or printed.
  twin   (render + TWIN_RENDER=1) BodyEnv.twin_stats per painted frame: IK residual pos_err_mm / rot_err_deg, retries,
         camera error, stale frames, step-0 image identity.
The second and later modes of a key run inside a saved np.random window (reset, same dumped placements), so the
draws of the next keys are the eval's.

Usage: python tools/nx_check.py --code $PROJ_ROOT/embody_next --suite libero_goal --body PandaR85 --offset 0 --out DIR
       [--n 5] [--tasks 0,1] [--modes replay,zero] [--zero-steps 200] [--render 0]
Cell switches (GRP_KP, GRP_MAP, COPY_GRIP_CMD, TWIN_RENDER, ...) come from the environment, as in the eval.
"""
import argparse
import glob
import importlib.util
import json
import os
import random
import re
import sys
import time

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", required=True)
ap.add_argument("--suite", required=True)
ap.add_argument("--body", required=True)
ap.add_argument("--offset", type=int, required=True)
ap.add_argument("--n", type=int, default=5)
ap.add_argument("--tasks", default="")
ap.add_argument("--modes", default="replay,zero")
ap.add_argument("--zero-steps", type=int, default=200)
ap.add_argument("--render", type=int, default=0)
ap.add_argument("--src-dir", default="")
ap.add_argument("--scene-index", default="")
ap.add_argument("--out", required=True)
args = ap.parse_args()

HOME = os.path.expanduser("~")
G10 = f"{HOME}/proj/vla_zoo/runs/src_goal10"
# SRC dumps: goal / 10 = src_goal10 OFT baselines (scene_index_*.json alongside); spatial = models1 OFTi SRC dumps
# (no scene index; natural_si_mm is then not computed)
DEF_SRC = {"libero_goal": f"{G10}/oft_goal/src", "libero_10": f"{G10}/oft_10/src",
           "libero_spatial": f"{HOME}/proj/vla_zoo/runs/oft_sp/src", "libero_object": f"{HOME}/proj/vla_zoo/runs/oft_obj/obj_src"}
src_dir = args.src_dir or DEF_SRC[args.suite]
si_path = args.scene_index or f"{G10}/scene_index_{args.suite}.json"
SI = json.load(open(si_path)) if os.path.exists(si_path) else None

sys.path.insert(0, os.path.expanduser(args.code))
import embody  # noqa: E402
assert os.path.abspath(embody.__file__).startswith(os.path.abspath(os.path.expanduser(args.code))), embody.__file__
from embody.original import SUITES  # noqa: E402
from embody import contacts as C  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from libero.libero.envs.env_wrapper import ControlEnv  # noqa: E402
import mujoco  # noqa: E402

_spec = importlib.util.spec_from_file_location("pbs", f"{HOME}/proj/vla_zoo/tools/physics_breakdown_scan.py")
pbs = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(pbs)  # main() is behind __name__ == "__main__"; only detect / LIM are used

np.random.seed(7)  # run_libero_eval: set_seed_everywhere(7) once per process, then all tasks in order
random.seed(7)
DUMMY = [0, 0, 0, 0, 0, 0, -1]
BADQACC = int(mujoco.mjtWarning.mjWARN_BADQACC)
os.makedirs(args.out, exist_ok=True)
tag = f"{args.suite}_{args.body}_e{args.offset}"
fout = open(os.path.join(args.out, f"{tag}.jsonl"), "a")
suite = SUITES[args.suite]()
tasks = [int(t) for t in args.tasks.split(",")] if args.tasks else list(range(suite.n_tasks))
OWN = ("robot0", "gripper0", "mount0")
logs = []


def nonrobot_static_vs_dump(model, z, only=None):
    names = [str(x) for x in z["ref_body_names"]]
    worst = 0.0
    for i, n in enumerate(names):
        if not n or n == "world" or n.startswith(OWN) or (only is not None and n not in only):
            continue
        try:
            b = model.body_name2id(n)
        except (ValueError, KeyError):
            continue
        worst = max(worst, float(np.abs(model.body_pos[b] - z["ref_body_pos"][i]).max()) * 1000)
    return worst


def natural_vs_si(model, fx):
    worst, n = 0.0, 0
    for name, pos in fx.items():
        try:
            b = model.body_name2id(name)
        except (ValueError, KeyError):
            continue
        worst = max(worst, float(np.abs(model.body_pos[b] - np.asarray(pos)).max()) * 1000)
        n += 1
    return worst, n


def joint_slices(model, name):
    qa = model.get_joint_qpos_addr(name)
    return slice(*qa) if isinstance(qa, tuple) else slice(qa, qa + 1)


def objects_vs_dump(ref_model, tgt_sim, z):
    """non-robot joints: target now vs the SRC dump's first recorded qpos (layout = reference Panda model)."""
    q0 = np.asarray(z["qpos"][0], float)
    tm = tgt_sim.model
    free_mm, free_quat, art, art_names, n = 0.0, 0.0, 0.0, [], 0
    for j in range(ref_model.njnt):
        name = ref_model.joint_id2name(j)
        if not name or name.startswith(OWN):
            continue
        rs = joint_slices(ref_model, name)
        try:
            ts = joint_slices(tm, name)
        except (ValueError, KeyError):
            art_names.append("MISSING:" + name)
            continue
        a, b = tgt_sim.data.qpos[ts], q0[rs]
        n += 1
        if len(a) == 7:
            free_mm = max(free_mm, float(np.abs(a[:3] - b[:3]).max()) * 1000)
            free_quat = max(free_quat, float(min(np.abs(a[3:] - b[3:]).max(), np.abs(a[3:] + b[3:]).max())))
        else:
            d = float(np.abs(a - b).max())
            if d > art:
                art = d
            art_names.append(name)
    return {"n_joints": n, "free_mm": free_mm, "free_quat": free_quat, "articulated_abs": art,
            "articulated": [x for x in art_names if x.startswith("MISSING")] or len(art_names)}


def scene_contacts(sim):
    """robot (robot0 / gripper0 / mount0) vs non-robot geoms with dist < 0 right now."""
    m, d = sim.model, sim.data
    deep, pairs = 0.0, {}
    for i in range(d.ncon):
        c = d.contact[i]
        b1 = m.body_id2name(m.geom_bodyid[c.geom1]) or ""
        b2 = m.body_id2name(m.geom_bodyid[c.geom2]) or ""
        r1, r2 = b1.startswith(OWN), b2.startswith(OWN)
        if r1 == r2 or c.dist >= 0:
            continue
        rb, ob = (b1, b2) if r1 else (b2, b1)
        pen = -float(c.dist) * 1000
        deep = max(deep, pen)
        k = f"{rb}|{ob}"
        pairs[k] = round(max(pairs.get(k, 0.0), pen), 2)
    return deep, pairs


def arm_info(env_robot, model):
    """qpos indices of the arm joints, finger joints, and limits for detector D."""
    ai = list(env_robot._ref_joint_pos_indexes)
    panda_arm = [f"robot0_joint{i}" for i in range(1, 8)] == list(env_robot.robot_joints)
    if panda_arm:
        lim, exact = pbs.LIM, True
    else:
        ids = [model.joint_name2id(n) for n in env_robot.robot_joints]
        lim = np.array([model.jnt_range[i] if model.jnt_limited[i] else (-np.inf, np.inf) for i in ids])
        exact = False
    gi = list(env_robot._ref_gripper_joint_pos_indexes)
    gn = list(env_robot.gripper_joints)
    fi = None
    slide = []
    for k, n in zip(gi, gn):
        if n == "gripper0_finger_joint":
            fi = k
        jid = model.joint_name2id(n)
        if int(model.jnt_type[jid]) == 2:  # mjJNT_SLIDE
            slide.append((k, model.jnt_range[jid].copy()))
    return ai, lim, exact, fi, slide, gi


def detect(q, ai, lim, fi, slide):
    """detector D (pbs.detect) with this body's limits; slide-joint fingers by own range + 10 mm (analogue)."""
    qa = q[:, ai]
    A = B = None
    if len(q) > 1:
        dq = np.abs(np.diff(qa, axis=0)).max(axis=1)
        jump = np.nonzero(dq > np.pi / 2)[0]
        mj = round(float(np.degrees(dq.max())), 1)
    else:
        jump, mj = np.zeros(0, int), 0.0
    oob = np.nonzero(((qa < lim[:, 0] - 0.1) | (qa > lim[:, 1] + 0.1)).any(axis=1))[0]
    if len(jump) or len(oob):
        A = {"first_step": int(min(([jump[0] + 1] if len(jump) else []) + ([oob[0]] if len(oob) else []))),
             "max_jump_deg": mj, "beyond_limit": bool(len(oob))}
    loose = False
    if fi is not None:
        fq = q[:, fi]
        bad = np.nonzero((fq < -0.5) | (fq > 1.2))[0]
        if len(bad):
            B = {"first_step": int(bad[0]), "extreme": round(float(fq[np.abs(fq).argmax()]), 2)}
        loose = bool(((fq < -1.0) | (fq > 2.0)).any())
    S = None
    for k, (lo, hi) in slide:
        bad = np.nonzero((q[:, k] < lo - 0.01) | (q[:, k] > hi + 0.01))[0]
        if len(bad):
            S = {"first_step": int(bad[0]) if S is None else min(S["first_step"], int(bad[0]))}
    return A, B, S, loose, mj


def run_mode(env, ref, body_on, tid, ep, init, inits, z, mode, scene_first):
    logs.clear()
    t0 = time.time()
    obs = env.set_init_state(inits[init])
    if not body_on:
        for _ in range(10):
            obs, _, _, _ = env.step(DUMMY)
    tgt = env.env if body_on else ref
    sim = tgt.sim
    rec = {"suite": args.suite, "body": args.body, "task": tid, "ep": ep, "init": init, "mode": mode,
           "cell_env": {k: os.environ.get(k) for k in ("GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY",
                                                         "TWIN_CAMS", "TWIN_CAM", "TWIN_CAM_SYNC", "NO_FOREARM_CONTACT", "R140_KP")
                        if os.environ.get(k) is not None}}
    if scene_first is not None:
        rec.update(scene_first)
    rm = (env.ref if body_on else ref).sim.model
    if body_on:
        from embody import align
        static = set(align.static_body_pose(env.ref).keys())
        rec["fixture_mm"] = nonrobot_static_vs_dump(sim.model, z, static)
        rec["align"] = env.align_info
        rr, tr = env.ref.robots[0], env.env.robots[0]
        pb_ref = env.ref.sim.data.get_body_xpos(rr.robot_model.root_body)
        pb_tgt = sim.data.get_body_xpos(tr.robot_model.root_body)
        rec["base_mm"] = [round(float(x) * 1000, 3) for x in (pb_tgt - pb_ref)]
        rec["robot_class"] = type(tr.robot_model).__name__
    rec["objects"] = objects_vs_dump(rm, sim, z)
    rec["init_pen_mm"], rec["init_pairs"] = scene_contacts(sim)
    robot = tgt.robots[0]
    ai, lim, exact, fi, slide, gi = arm_info(robot, sim.model)
    rec["D_exact"] = exact
    acts = [a.tolist() for a in z["action"]] if mode == "replay" else [DUMMY] * args.zero_steps
    clog = C.make(env)  # read-only contact log, works on BodyEnv and on the plain env
    q = [sim.data.qpos.copy()]
    qacc, resets = 0, 0
    raw = sim.data._data
    w0 = int(raw.warning[BADQACC].number)
    gp = robot.gripper.naming_prefix
    movable = clog._obj_body
    grip_obj_steps, both_pads_steps = 0, 0
    pad_names = [n for n in ("left_fingerpad_collision", "right_fingerpad_collision", "finger1_pad_collision",
                             "finger2_pad_collision", "l_fingerpad_g0", "r_fingerpad_g0") if (gp + n) in sim.model._geom_name2id]
    pad_ids = {sim.model.geom_name2id(gp + n) for n in pad_names}
    for act in acts:
        tb = float(raw.time)
        env.step(act)
        if clog is not None:
            clog.update(act)
        raw = sim.data._data
        w = int(raw.warning[BADQACC].number)
        if w != w0:
            qacc += max(w - w0, 1)
            w0 = w
        if float(raw.time) < tb:
            resets += 1
        d_ = sim.data
        hit, pads_hit = False, set()
        for i in range(d_.ncon):
            c = d_.contact[i]
            for g1, g2 in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if (sim.model.body_id2name(sim.model.geom_bodyid[g1]) or "").startswith(gp) and int(sim.model.geom_bodyid[g2]) in movable:
                    hit = True
                    if g1 in pad_ids:
                        pads_hit.add(g1)
        grip_obj_steps += hit
        both_pads_steps += len(pads_hit) >= 2
        q.append(sim.data.qpos.copy())
    q = np.asarray(q)
    if not body_on and mode == "replay":  # SRC sanity: the replay must reproduce the dumped native trajectory
        qd = np.asarray(z["qpos"], float)
        m_ = min(len(qd), len(q))
        rec["src_qpos_maxdiff"] = float(np.abs(q[:m_] - qd[:m_]).max())
    A, B, S, loose, mj = detect(q, ai, lim, fi, slide)
    rec.update({"grip_obj_steps": int(grip_obj_steps), "both_pads_obj_steps": int(both_pads_steps), "pad_geoms": pad_names,
                "n_steps_run": len(acts), "D_arm": A, "D_finger": B, "D_slide": S, "D_loose": loose,
                "max_jump_deg": mj, "qacc": qacc, "time_resets": resets,
                "breakdown": bool(A or B or S or qacc or resets)})
    if clog is not None:
        s = clog.summary()
        rec["contacts"] = {k: s[k] for k in ("arm_env_contact_steps", "first_contact", "pairs") if k in s}
    ts = getattr(env, "twin_stats", None) if body_on else None
    if ts:
        pe, re_ = np.asarray(ts["pos_err_mm"]), np.asarray(ts["rot_err_deg"])
        rec["twin"] = {"n": int(len(pe)), "pos_p50": float(np.median(pe)), "pos_p99": float(np.percentile(pe, 99)),
                       "pos_max": float(pe.max()), "rot_max": float(re_.max()),
                       "gt1mm": int((pe > 1).sum()), "gt2mm": int((pe > 2).sum()),
                       "q_gt1mm": int((pe[::8] > 1).sum()), "nq": int(len(pe[::8])),
                       "retries": int(ts.get("retries", 0)), "stale": int(ts["stale"]), "t0": ts["t0"],
                       "cam_err_mm_max": float(np.max(ts.get("cam_err_mm", [0]))),
                       "cam_err_deg_max": float(np.max(ts.get("cam_err_deg", [0])))}
        np.savez_compressed(os.path.join(args.out, f"{tag}_t{tid}_e{ep}_{mode}_twin.npz"), pos=pe, rot=re_)
    rec["wall_s"] = round(time.time() - t0, 1)
    rec["log"] = [l for l in logs if "BODY_ALIGN" in l or "TWIN" in l][:3]
    np.savez_compressed(os.path.join(args.out, f"{tag}_t{tid}_e{ep}_{mode}.npz"),
                        q=q[:, ai + gi].astype(np.float32))
    fout.write(json.dumps(rec) + "\n")
    fout.flush()
    print(json.dumps({k: rec.get(k) for k in ("task", "ep", "mode", "breakdown", "qacc", "fixture_mm", "base_mm",
                                               "init_pen_mm")}), flush=True)


for tid in tasks:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    if args.render:
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    else:
        ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    body_on = args.body != "SRC"
    scene_cls = None
    if body_on:
        from embody.body_env import BodyEnv
        from embody.scenes import body_scene_class
        scene_cls = body_scene_class(args.body, bddl)
        env = BodyEnv(ref, bddl, args.body, 256, 10, log=lambda m: logs.append(m), render=bool(args.render))
        assert type(env.env.robots[0].robot_model).__name__ == scene_cls[0], (type(env.env.robots[0].robot_model).__name__, scene_cls)
    else:
        env = ref
    perm = np.random.RandomState(5478 + tid).permutation(len(inits))
    for ep in range(args.offset, args.offset + args.n):
        init = int(perm[ep])
        fs = glob.glob(os.path.join(src_dir, f"t{tid}_e{ep}_i*.npz"))
        assert len(fs) == 1, (tid, ep, fs)
        z = np.load(fs[0], allow_pickle=True)
        si = SI[f"{tid},{ep}"] if SI is not None else {"init": init, "fixtures": {}}
        env.reset()
        rm = (env.ref if body_on else ref).sim.model
        first = {"init_ok": int(z["init"]) == init and int(si["init"]) == init,
                 "natural_mm": nonrobot_static_vs_dump(rm, z),
                 "natural_si_mm": natural_vs_si(rm, si["fixtures"])[0] if SI is not None else None,
                 "scene": None if scene_cls is None else list(scene_cls)}

        def place():
            names = [str(x) for x in z["ref_body_names"]]
            for i, n in enumerate(names):
                if not n:
                    continue
                try:
                    b = rm.body_name2id(n)
                except (ValueError, KeyError):
                    continue
                rm.body_pos[b] = z["ref_body_pos"][i]
                rm.body_quat[b] = z["ref_body_quat"][i]

        modes = [m for m in args.modes.split(",") if m]
        for k, mode in enumerate(modes):
            if k == 0:
                place()
                run_mode(env, ref, body_on, tid, ep, init, inits, z, mode, first)
            else:
                st = np.random.get_state()
                env.reset()
                rm = (env.ref if body_on else ref).sim.model
                place()
                run_mode(env, ref, body_on, tid, ep, init, inits, z, mode, {"init_ok": first["init_ok"], "scene": first["scene"]})
                np.random.set_state(st)
print("DONE", tag, flush=True)
