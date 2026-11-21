"""TOY1 smoke (no policy, 2026-11-03): replay the dumped actions of a published cell (default grpkp100 = Robotiq kp100,
physical body PandaR85) through embody_toy BodyEnv with the TOY_* variables set by the caller, so every condition
sees exactly the same physical trajectory, and save what the policy would see.

Per episode (jsonl): steps, replay success vs dump, qpos max diff vs dump, ms per step, twin stats (pad / hand
residual, stale), toy pixel counts (min / median / max per column), wrist-camera world position of the policy view
relative to the physical pipeline camera, and for TOY_PARTS a check that hidden geoms are really absent from the
segmentation and that the twin camera sits on the physical one.
Screenshots (policy orientation, agentview | wrist, 256 px each, plus a 2x wrist crop): step 0, 8 steps before the
first close command, and 3 steps after it.

Usage: python tools/smoke_toy.py --name NAME --body PandaR85 --cond grpkp100 --tasks 0,2 --eps 0 --out DIR --shots DIR
"""
import argparse, glob, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--body", default="PandaR85")
ap.add_argument("--cond", default="grpkp100")
ap.add_argument("--tasks", default="0,2")
ap.add_argument("--eps", default="0")
ap.add_argument("--out", required=True)
ap.add_argument("--shots", required=True)
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--max-steps", type=int, default=0, help="stop after this many steps (0 = whole dump)")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True); os.makedirs(args.shots, exist_ok=True)
suite = OriginalSpatialSuite()


def shot(obs, path):
    from PIL import Image
    a = np.asarray(obs["agentview_image"])[::-1, ::-1]
    w = np.asarray(obs["robot0_eye_in_hand_image"])[::-1, ::-1]
    wz = np.kron(w[64:192, 64:192], np.ones((2, 2, 1), dtype=np.uint8))  # centre crop of the wrist view, 2x
    Image.fromarray(np.concatenate([a, w, wz], axis=1).astype(np.uint8)).save(path)


def set_ref_fixtures(rm, z):
    for i, n in enumerate(str(x) for x in z["ref_body_names"]):
        if not n:
            continue
        try:
            bid = rm.body_name2id(n)
        except (ValueError, KeyError):
            continue
        rm.body_pos[bid] = z["ref_body_pos"][i]
        rm.body_quat[bid] = z["ref_body_quat"][i]


for tid in [int(t) for t in args.tasks.split(",")]:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    logs = []
    env = BodyEnv(ref, bddl, args.body, 256, 10, log=logs.append)
    for ep in [int(e) for e in args.eps.split(",")]:
        fs = glob.glob(os.path.join(args.runs, args.cond, f"t{tid}_e{ep}_i*.npz"))
        if not fs:
            print("missing", tid, ep, flush=True); continue
        z = np.load(fs[0], allow_pickle=True)
        env.reset()
        set_ref_fixtures(env.ref.sim.model, z)
        logs.clear()
        t0 = time.time()
        obs = env.set_init_state(inits[int(z["init"])])
        t_init = time.time() - t0
        rec = {"name": args.name, "task": tid, "ep": ep, "init": int(z["init"]), "align": env.align_info,
               "toy": getattr(env, "toy_info", None)}
        # hiding check (TOY_PARTS): hidden physical geoms must not appear in the segmentation, twin camera on physical
        if getattr(env, "toy_parts", ""):
            show, hide_t, hide_p = env._toy_sets()
            chk = {}
            for ok, cam in env.CAMS:
                gp = env._render(env.env.sim, cam, hide_p, depth=False, seg=True)[2]
                gt = env._render(env.twin.sim, cam, hide_t, depth=False, seg=True)[2]
                chk[cam] = {"hidden_phys_px": int(np.isin(gp, hide_p).sum()), "hidden_twin_px": int(np.isin(gt, hide_t).sum()),
                            "shown_twin_px": int(np.isin(gt, show).sum())}
            ct = env.twin.sim.model.camera_name2id("robot0_eye_in_hand")
            cp = env.env.sim.model.camera_name2id("robot0_eye_in_hand")
            chk["wrist_cam_twin_vs_phys_mm"] = float(np.linalg.norm(env.twin.sim.data.cam_xpos[ct] - env.env.sim.data.cam_xpos[cp]) * 1000)
            chk["n_show"], chk["n_hide_twin"], chk["n_hide_phys"] = len(show), len(hide_t), len(hide_p)
            rec["hide_check"] = chk
        closes = np.nonzero(z["action"][:, -1] > 0)[0]
        c = int(closes[0]) if len(closes) else 60
        want = {0: "1_step0", max(c - 8, 1): f"2_pre{max(c - 8, 1)}", c + 3: f"3_grasp{c + 3}"}
        tag = f"{args.name}_t{tid}e{ep}"
        shot(obs, os.path.join(args.shots, f"{tag}_{want[0]}.png"))
        cam_rel = []
        sim = env.env.sim
        cp = sim.model.camera_name2id("robot0_eye_in_hand")

        def cam_rel_mm():
            # policy wrist camera (twin if painted) relative to the physical camera, in the physical grip-site frame
            from embody import align
            p_s, R_s = align.site_pose(sim, env.env.robots[0].eef_site_id)
            src = env.twin.sim if (env.twin is not None and env.twin_cams in ("both", "wrist")) else sim
            cid = src.model.camera_name2id("robot0_eye_in_hand")
            return (R_s.T @ (src.data.cam_xpos[cid] - sim.data.cam_xpos[cp]) * 1000).tolist()
        cam_rel.append(cam_rel_mm())
        n, success, ts = 0, False, []
        traj = [sim.data.qpos.copy()]
        nmax = args.max_steps or len(z["action"])
        for k, act in enumerate(z["action"][:nmax]):
            t1 = time.time()
            obs, _, done, _ = env.step(act.tolist())
            ts.append(time.time() - t1)
            n += 1
            traj.append(sim.data.qpos.copy())
            if k + 1 in want:
                shot(obs, os.path.join(args.shots, f"{tag}_{want[k + 1]}.png"))
                cam_rel.append(cam_rel_mm())
            if done:
                success = True
                break
        traj = np.asarray(traj[:-1]) if len(traj) > len(z["qpos"]) else np.asarray(traj)
        m = min(len(traj), len(z["qpos"]))
        rec.update(steps=n, dump_steps=int(len(z["action"])), dump_success=bool(z["success"]), replay_success=success,
                   qpos_max_diff_vs_dump=float(np.abs(traj[:m] - z["qpos"][:m]).max()),
                   init_s=round(t_init, 2), ms_per_step=round(1000 * float(np.mean(ts)), 1),
                   wrist_cam_rel_phys_mm=cam_rel)
        st = getattr(env, "twin_stats", None)
        if st:
            pe = np.asarray(st["pos_err_mm"])
            rec["twin"] = {"n": len(pe), "pos_max": float(pe.max()), "q_over_2mm": int((pe[::8] > 2).sum()), "nq": len(pe[::8]),
                           "stale": int(st["stale"]), "t0": st["t0"]}
        toy = getattr(env, "toy_stats", None)
        if toy and toy.get("px"):
            px = np.asarray(toy["px"])
            rec["toy_px"] = {"cols": ["agent_panda", "agent_robotiq", "wrist_panda", "wrist_robotiq"],
                             "min": px.min(0).tolist(), "median": np.median(px, 0).tolist(), "max": px.max(0).tolist(),
                             "step0": px[0].tolist(), "n": len(px)}
        rec["log"] = [l for l in logs if "TOY1" in l or "TWIN_T0" in l]
        with open(os.path.join(args.out, f"{args.name}.jsonl"), "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        print(json.dumps(rec), flush=True)
