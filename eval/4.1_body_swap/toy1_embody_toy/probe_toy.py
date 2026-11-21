"""TOY1 cached-frame probes (protocol_toy1.md 5.3 / 5.4 as amended by protocol_toy1_astra_decisions.md (f)),
2026-11-03. Frozen OFT, no rollout, no success rates read (episode outcome is not used for Robotiq frames).

Frames: Robotiq G states (physical PandaR85, GRP_KP=100, GRP_MAP=kp100): --runs/--cond dumps, 6 episodes per task in
ascending ep (any outcome), 5 chunk-boundary frames each (probe_common.select_frames: 2 approach + 3 pre-grasp),
episodes with < 3 pre-grasp boundaries skipped. The SAME state, physical wrist camera and proprio are used for every
appearance and condition (decision (f)).

Shift probe (5.3, --do shift), two appearances on each frame, wrist camera pinned to the physical camera:
  robotiq: twin PandaR85, joints copied (TWIN_MOUNT=copy), the twin Robotiq ROOT body is translated (arm fixed)
  panda  : twin Panda placed as G_T^fc (pads on the Robotiq pads, IK chain walked through the episode), the twin
           Panda gripper root (hand + fingers) is translated (arm fixed)
  conditions: 0, 0p (repeat), x-10, x-5, x+5, x+10 (along u_x = grip-site x = finger opening axis, world), y-5, y+5
  (u_y), cam-5, cam+5 (wrist camera translated along u_x; parallax, descriptive only per decision (f)).
Parts probe (5.4, --do parts) on the same frames, cameras at the physical pose: c0 G as is; c1 Panda twin (G_T^fc
image); c2 Robotiq all hidden (alpha 0); c3 Robotiq knuckles/fingers/pads hidden; c4 Robotiq base hidden; c5 = c3 +
Panda twin finger pixels (twin hand hidden, twin finger segmentation pixels copied over, no depth test); c6 = c4 +
Panda twin hand pixels (twin fingers hidden). Coverage counts: c5 twin-finger pixels landing on physical Robotiq base
pixels (c3 render), c6 twin-hand pixels landing on physical Robotiq finger pixels (c4 render). First --save-png frames
are saved as image grids.

Output jsonl rows (shift): appearance, task, ep, init, step, phase, u_x, u_y, R_base, actions{cond: 8x7 processed},
raw{cond: 8x7}, proprio{cond}, cam_err_mm{cond}, cam_err_deg{cond}, centroid_px{cond: [drow, dcol] vs "0"},
expected_px{cond}, grip_px{cond}, same_0_0p{pixels, actions}; rows (parts): task, ep, init, step, phase,
actions{c0..c6}, raw{...}, coverage{c5, c6}, px{ci: [agent_grip, wrist_grip]}.
Usage: python tools/probe_toy.py --runs $PROJ_ROOT/runs_toy --cond toy_g --out DIR [--do shift,parts] [--tasks ...]
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import probe_common as PC  # noqa: E402
from embody import align  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--cond", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--do", default="shift,parts")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--per-task", type=int, default=6)
ap.add_argument("--max-frames", type=int, default=0, help="smoke: stop after this many frames")
ap.add_argument("--save-png", type=int, default=10)
ap.add_argument("--frame-rule", default="astra", choices=["protocol", "preclose", "astra"])
ap.add_argument("--eps", default="6-11", help="episode range a-b (astra rule: grpkp100 ep 6-11 per task)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
DO = set(args.do.split(","))
G = {"BODY": "PandaR85", "GRP_KP": "100", "GRP_MAP": "kp100", "COPY_GRIP_CMD": "1"}
CFG = {"panda": dict(G, TWIN_RENDER="1", TWIN_CAM="phys"),
       "robotiq": dict(G, TWIN_RENDER="1", TWIN_BODY="PandaR85", TWIN_MOUNT="copy", TWIN_CAM="phys")}
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM", "TWIN_CAMS",
        "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM", "TWIN_NS_GAIN"}
CONDS = [("0", None, 0), ("0p", None, 0), ("x-10", "x", -10), ("x-5", "x", -5), ("x+5", "x", 5), ("x+10", "x", 10),
         ("y-5", "y", -5), ("y+5", "y", 5), ("cam-5", "cam", -5), ("cam+5", "cam", 5)]
CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))
suite = OriginalSpatialSuite()
objs = PC.task_objects()
fh_s = open(os.path.join(args.out, "shift.jsonl"), "a") if "shift" in DO else None
fh_p = open(os.path.join(args.out, "parts.jsonl"), "a") if "parts" in DO else None
png_dir = os.path.join(args.out, "png")
n_png = 0
n_frames = 0


def set_vars(v):
    for k in ALLV:
        os.environ.pop(k, None)
    os.environ.update(v)


def make_env(bddl, v):
    set_vars(v)
    st = np.random.get_state()
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    e = BodyEnv(ref, bddl, v["BODY"], 256, 10, log=lambda m: None)
    np.random.set_state(st)
    return e


def paint(env, qpos, probe=None):
    env.set_probe(**(probe or {}))
    o = PC.physical_obs(env, qpos)
    o = env._twin_paint(o)
    return env._fix(o)


def grip_centroid(env):
    """Twin gripper pixels in the raw wrist image: (count, [mean row, mean col], mask)."""
    gid = PC.seg_ids(env, env.twin.sim, "robot0_eye_in_hand")
    mask = np.isin(gid, PC.gripper_geoms(env.twin.sim.model))
    if not mask.any():
        return 0, None, mask
    r, c = np.nonzero(mask)
    return int(mask.sum()), [float(r.mean()), float(c.mean())], mask


def shift_frame(env, app, z, k, lang, u):
    rec = {"actions": {}, "raw": {}, "proprio": {}, "cam_err_mm": {}, "cam_err_deg": {}, "centroid_px": {},
           "expected_px": {}, "grip_px": {}, "mask_shift_px": {}, "expected_px_lib": {}}
    m0 = None
    imgs = {}
    c0 = None
    p_pad = None
    for name, ax, dmm in CONDS:
        vec = None if ax is None else (u[ax if ax != "cam" else "x"] * dmm / 1000.0)
        probe = {}
        if ax in ("x", "y"):
            probe = {"robot_shift": None, "grip_shift": vec}
        elif ax == "cam":
            probe = {"cam_shift": vec}
        o = paint(env, z["qpos"][k], probe)
        if app == "panda" and name == "0":  # freeze the twin arm of this frame for all conditions (IK solved once)
            env.probe_fix_q = np.array(env._twin_q, copy=True)
        raw, proc, state = PC.query(o, lang)
        rec["raw"][name], rec["actions"][name], rec["proprio"][name] = raw.tolist(), proc.tolist(), state.tolist()
        rec["cam_err_mm"][name] = env.twin_stats["cam_err_mm"][-1]
        rec["cam_err_deg"][name] = env.twin_stats["cam_err_deg"][-1]
        n, cen, msk = grip_centroid(env)
        rec["grip_px"][name] = n
        if name == "0":
            c0, m0 = cen, msk
            pads = (("finger1_pad_collision", "finger2_pad_collision") if app == "panda"
                    else ("left_fingerpad_collision", "right_fingerpad_collision"))
            p_pad = env._pad_centre(env.twin.sim, pads)
        rec["mask_shift_px"][name] = PC.mask_shift(m0, msk) if (m0 is not None and name != "0" and not name.startswith("cam")) else None
        rec["centroid_px"][name] = None if (cen is None or c0 is None) else [cen[0] - c0[0], cen[1] - c0[1]]
        if ax in ("x", "y") and p_pad is not None:
            pr = PC.project_raw(env.twin.sim, "robot0_eye_in_hand", np.stack([p_pad, p_pad + vec]))  # p_pad: unshifted
            rec["expected_px"][name] = (pr[1] - pr[0]).tolist()
            pl = PC.project(env.twin.sim, "robot0_eye_in_hand", np.stack([p_pad, p_pad + vec]))
            rec["expected_px_lib"][name] = (pl[1] - pl[0]).tolist()
        imgs[name] = {kk: np.array(o[kk], copy=True) for kk, _ in CAMS}
    rec["same_0_0p"] = {"pixels": all(np.array_equal(imgs["0"][kk], imgs["0p"][kk]) for kk, _ in CAMS),
                        "actions": rec["raw"]["0"] == rec["raw"]["0p"]}
    env.set_probe()
    env.probe_fix_q = None
    return rec, imgs


def parts_frame(env, z, k, lang, save):
    """env = the panda-appearance env (physical G + twin Panda in G_T^fc pose); state already painted at k."""
    sim_p, sim_t = env.env.sim, env.twin.sim
    pb, pf = env._grip_parts(sim_p.model)
    tb, tf = env._grip_parts(sim_t.model)
    o_phys = PC.physical_obs(env, z["qpos"][k])
    o_tw = env._fix(env._twin_paint(dict(o_phys)))
    o_phys = env._fix(o_phys)
    out = {"c0": o_phys, "c1": o_tw}
    cov = {}
    px = {}
    for name, hide in (("c2", pb + pf), ("c3", pf), ("c4", pb)):
        o = dict(o_phys)
        for kk, cam in CAMS:
            o[kk] = env._render(sim_p, cam, hide, depth=False, how="alpha")[0]
        out[name] = o
    for name, base, hide_t, show_t, phys_hidden, phys_target in (("c5", "c3", tb, tf, pf, pb), ("c6", "c4", tf, tb, pb, pf)):
        o = dict(out[base])
        cov[name] = {}
        for kk, cam in CAMS:
            _, _, gid_t = env._render(sim_t, cam, hide_t, depth=False, seg=True, how="alpha")
            _, _, gid_p = env._render(sim_p, cam, phys_hidden, depth=False, seg=True, how="alpha")
            tw_rgb = env._render(sim_t, cam, hide_t, depth=False, how="alpha")[0]
            m = np.isin(gid_t, show_t)
            img = np.array(o[kk], copy=True)
            img[m] = tw_rgb[m]
            o[kk] = img
            cov[name][cam] = {"twin_px": int(m.sum()), "on_phys_kept_px": int((m & np.isin(gid_p, phys_target)).sum())}
        out[name] = o
    for name in ("c0", "c1", "c2", "c3", "c4", "c5", "c6"):
        for kk, cam in CAMS:
            gid = PC.seg_ids(env, sim_t if name == "c1" else sim_p, cam) if name in ("c0", "c1") else None
            if gid is not None:
                px.setdefault(name, []).append(int(np.isin(gid, PC.gripper_geoms((sim_t if name == "c1" else sim_p).model)).sum()))
    rec = {"actions": {}, "raw": {}, "coverage": cov, "px": px}
    for name, o in out.items():
        raw, proc, _ = PC.query(o, lang)
        rec["actions"][name], rec["raw"][name] = proc.tolist(), raw.tolist()
    if save:
        from PIL import Image
        os.makedirs(png_dir, exist_ok=True)
        row = lambda kk: np.concatenate([np.asarray(out[n][kk])[::-1, ::-1] for n in ("c0", "c1", "c2", "c3", "c4", "c5", "c6")], axis=1)
        Image.fromarray(np.concatenate([row("agentview_image"), row("robot0_eye_in_hand_image")], 0).astype(np.uint8)).save(
            os.path.join(png_dir, f"parts_t{z['task']}_e{z['ep']}_s{k}.png"))
    return rec


print(f"PROBE_TOY runs={args.runs} cond={args.cond} do={sorted(DO)}", flush=True)
for tid in [int(t) for t in args.tasks.split(",")]:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    envs = {a: make_env(bddl, v) for a, v in CFG.items()}
    target = objs[tid][0]
    used = 0
    ea, eb = (int(x) for x in args.eps.split("-"))
    for ep, f in PC.episodes(args.runs, args.cond, args.per_task, False, [tid], eps=set(range(ea, eb + 1)))[tid]:
        if (args.frame_rule != "astra" and used >= args.per_task) or (args.max_frames and n_frames >= args.max_frames):
            break
        z = np.load(f, allow_pickle=True)
        frames, info = PC.select_frames(z, target, rule=args.frame_rule)
        if frames is None:
            print(json.dumps({"skip": [tid, ep], **info}), flush=True)
            with open(os.path.join(args.out, "missing.jsonl"), "a") as fm:
                fm.write(json.dumps({"task": tid, "ep": ep, **info}) + "\n")
            continue
        used += 1
        t0 = time.time()
        for a, e in envs.items():
            set_vars(CFG[a])
            PC.start_episode(e, z, inits)
        fk = {fr[0]: fr[1:] for fr in frames}
        last = max(fk)
        for k in range(0, last + 1):  # walk the panda twin IK chain (G_T^fc pose) through the episode
            if k not in fk:
                paint(envs["panda"], z["qpos"][k])
                continue
            phase = fk[k][0]
            extra = {"dist_cm": fk[k][1], "stratum": fk[k][2], "t_c": info.get("t_c")} if len(fk[k]) > 1 else {}
            p, R = align.site_pose(envs["robotiq"].env.sim, envs["robotiq"].env.robots[0].eef_site_id)
            u = {"x": R[:, 0].copy(), "y": R[:, 1].copy()}
            m_ = envs["robotiq"].env.sim.model
            Rb = envs["robotiq"].env.sim.data.xmat[m_.body_name2id("robot0_base")].reshape(3, 3)
            base = {"task": tid, "ep": ep, "init": int(z["init"]), "step": k, "phase": phase, "cond_src": args.cond, "frame_rule": args.frame_rule,
                    "u_x": u["x"].tolist(), "u_y": u["y"].tolist(), "R_base": Rb.tolist(), **extra}
            if "shift" in DO:
                for a in ("robotiq", "panda"):
                    rec, _ = shift_frame(envs[a], a, z, k, task.language, u)
                    fh_s.write(json.dumps({"appearance": a, **base, **rec}) + "\n")
                fh_s.flush()
            if "parts" in DO:
                paint(envs["panda"], z["qpos"][k])  # re-paint at k without offsets (chain state)
                rec = parts_frame(envs["panda"], z, k, task.language, n_png < args.save_png)
                n_png += 1
                fh_p.write(json.dumps({**base, **rec}) + "\n")
                fh_p.flush()
            n_frames += 1
        print(json.dumps({"task": tid, "ep": ep, "frames": [list(f_) for f_ in frames], "sec": round(time.time() - t0, 1), **info}), flush=True)
    if args.max_frames and n_frames >= args.max_frames:
        break
print("PROBE_TOY done", n_frames, flush=True)
