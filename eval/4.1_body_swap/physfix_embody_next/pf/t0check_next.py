"""physfix step-0 consistency (protocol_physfix.md 0dd4193c 7.2), 2026-11-06. H200 ONLY: asserts GL_RENDERER contains
"H200" right after the first GL context exists, else exit 2 before any output file is opened.
All pf_ BodyEnv cells of the suite that exist in next/cells_next.sh are built in ONE process; every key gets the same
initial state (seed 7, tasks in order, cells built in RNG windows, as t0check_m1.py). No policy, no outcome.
Per key (gripper geoms alpha 0 on both sides = "both hidden"):
  T0 vs S: both cameras mean / max;
  each twin cell vs its physics cell (GTfc' vs G', GTfc'' vs G'', GTfc'_b vs G'_b): wrist + agent both-hidden mean,
    first-paint camera error and IK residual;
  GTfc' (and GTfc'', GTfc'_b): finite IK search M = min over 64 seeds (rest = physical joints, robot init_qpos,
    BodyEnv seed library, the rest U(joint range) RandomState(0)) + 2 targeted descents (joint-near, link-near) of the
    agent both-hidden mean of the twin posture that keeps the pads aligned (grip-site pose of the painted twin);
    excess = actual - M; background diff outside the union of all robot / gripper pixels (both sims); max |physical
    joints (twin cell) - physical joints (G cell)|;
  GTloc' (TOY_PARTS=hand) vs G': diff inside the physical-arm mask not covered by the composited Panda hand (expect 0),
    diff outside (arm U Robotiq U Panda hand) (expect 0), joint gap = min pixel distance between the composited Panda
    hand-root (palm) pixels and the physical link7 pixels, per camera.
Usage: python t0check_next.py --suite libero_object --out FILE.jsonl [--trials 5] [--nseeds 64]
"""
import argparse, json, os, random, socket, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--nseeds", type=int, default=64)
ap.add_argument("--iters", type=int, default=2000)
ap.add_argument("--allow-non-h200", action="store_true", help="debug only; output is marked and gate json refuses it")
args = ap.parse_args()
sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_next")))
from embody.body_env import BodyEnv  # noqa
from embody import align  # noqa
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import mujoco  # noqa

CELLS_SH = os.path.expanduser("$PROJ_ROOT/vla_zoo/next/cells_next.sh")
OBJ = args.suite == "libero_object"
PRE = "pf_obj_" if OBJ else "pf_"
WANT = ["sham_cg", "sham_twin", "grpkp100a", "g_tfc_a", "g_tloc_a"] + (["grpkp100a_m20", "g_tfc_a_m20", "grpkp20", "g_tfc_kp20"] if OBJ else [])
PAIRS = {"g_tfc_a": "grpkp100a", "g_tloc_a": "grpkp100a", "g_tfc_a_m20": "grpkp100a_m20", "g_tfc_kp20": "grpkp20"}
CELLS = {}
for n in WANT:
    r = subprocess.run(["bash", CELLS_SH, "vars", PRE + n], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        o = r.stdout.split()
        CELLS[n] = dict(BODY=o[0], **dict(kv.split("=", 1) for kv in o[1:]))
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "R85_ARMATURE", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM",
        "TWIN_CAMS", "TWIN_CAM_SYNC", "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_PHYS_GRIPPER", "TOY_TWIN_POSE",
        "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM"}
A, W = "agentview_image", "robot0_eye_in_hand_image"
CAMS = ((A, "agentview"), (W, "robot0_eye_in_hand"))
LINKS = [f"robot0_link{i}" for i in range(1, 8)]
HOST = socket.gethostname()
GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def set_vars(v):
    for k in ALLV:
        os.environ.pop(k, None)
    os.environ.update(v)


def dstat(a, b, mask=None):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    if mask is not None:
        d = d[mask]
    return {"mean": float(d.mean()) if d.size else 0.0, "max": int(d.max()) if d.size else 0}


def gg(m, pre=("gripper0",)):
    return [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(pre)]


def hid(be, sim, cam):
    return be._render(sim, cam, gg(sim.model), depth=False, how="alpha")[0]


def segmask(be, sim, cam, pre):
    gid = be._render(sim, cam, [], depth=False, seg=True)[2]
    return np.isin(gid, gg(sim.model, pre))


def rot_vec(R, Rt):
    return 0.5 * sum(np.cross(R[:, i], Rt[:, i]) for i in range(3))


def ns_solve(env, p_t, R_t, q0, cost, target, iters):
    rb, sim = env.robots[0], env.sim
    m, d = sim.model._model, sim.data._data
    qi, di = np.array(rb._ref_joint_pos_indexes), np.array(rb._ref_joint_vel_indexes)
    jid = [sim.model.joint_name2id(j) for j in rb.robot_joints]
    lo, hi = m.jnt_range[jid, 0], m.jnt_range[jid, 1]
    saved = d.qpos.copy()
    q = np.array(q0, float)
    bids = [sim.model.body_name2id(n) for n in LINKS]
    jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
    for _ in range(iters):
        d.qpos[qi] = q
        mujoco.mj_kinematics(m, d); mujoco.mj_comPos(m, d)
        p = d.site_xpos[rb.eef_site_id].copy(); R = d.site_xmat[rb.eef_site_id].reshape(3, 3).copy()
        e = np.concatenate([p_t - p, rot_vec(R, R_t)])
        mujoco.mj_jacSite(m, d, jp, jr, rb.eef_site_id)
        J = np.vstack([jp[:, di], jr[:, di]])
        Jp = np.linalg.pinv(J)
        if cost == "joint":
            g = target - q
        else:
            g = np.zeros(len(q))
            for k, b in enumerate(bids):
                mujoco.mj_jacBodyCom(m, d, jp, jr, b)
                g += jp[:, di].T @ (target[k] - d.xipos[b])
        q = np.clip(q + Jp @ e + (np.eye(len(q)) - Jp @ J) @ (0.2 * g), lo, hi)
    d.qpos[qi] = q
    mujoco.mj_kinematics(m, d)
    p = d.site_xpos[rb.eef_site_id].copy(); R = d.site_xmat[rb.eef_site_id].reshape(3, 3).copy()
    ep, er = float(np.linalg.norm(p_t - p)), float(np.linalg.norm(rot_vec(R, R_t)))
    d.qpos[:] = saved
    mujoco.mj_forward(m, d)
    return q, ep, er


def search_M(e, g):
    """finite IK search over twin arm postures keeping the painted twin's grip-site pose; returns M and details."""
    tw = e.twin
    rb = tw.robots[0]
    qi = rb._ref_joint_pos_indexes
    q_cur = tw.sim.data.qpos[qi].copy()
    p_t, R_t = align.site_pose(tw.sim, rb.eef_site_id)
    q_ph = e.env.sim.data.qpos[e.env.robots[0]._ref_joint_pos_indexes].copy()
    x_ph = np.array([e.env.sim.data.xipos[e.env.sim.model.body_name2id(n)] for n in LINKS])
    jid = [tw.sim.model.joint_name2id(j) for j in rb.robot_joints]
    lo, hi = tw.sim.model._model.jnt_range[jid, 0], tw.sim.model._model.jnt_range[jid, 1]
    seeds = [q_ph, np.asarray(rb.init_qpos, float)]
    try:
        seeds += [np.asarray(s, float) for s in e._seed_postures(p_t)]
    except Exception:
        pass
    rs = np.random.RandomState(0)
    while len(seeds) < args.nseeds:
        seeds.append(lo + rs.rand(len(lo)) * (hi - lo))
    seeds = seeds[:args.nseeds]
    G_a = hid(g, g.env.sim, "agentview")
    cands = [("joint_near", q_ph, "joint", q_ph), ("link_near", q_ph, "link", x_ph)] + [(f"s{i}", s, "joint", s) for i, s in enumerate(seeds)]
    best, n_ok, vals = None, 0, []
    for name, q0, cost, tgt in cands:
        q, ep, er = ns_solve(tw, p_t, R_t, q0, cost, tgt, args.iters)
        if ep > 1e-4 or er > 1e-3:  # not on the pad-aligned constraint
            continue
        n_ok += 1
        tw.sim.data.qpos[qi] = q
        tw.sim.forward()
        v = dstat(hid(e, tw.sim, "agentview"), G_a)["mean"]
        vals.append(v)
        if best is None or v < best[1]:
            best = (name, v)
    tw.sim.data.qpos[qi] = q_cur
    tw.sim.forward()
    return {"M": best[1] if best else None, "M_from": best[0] if best else None, "n_valid": n_ok, "n_tried": len(cands),
            "p10": float(np.percentile(vals, 10)) if vals else None}


def min_pix_dist(a, b):
    pa, pb = np.argwhere(a), np.argwhere(b)
    if not len(pa) or not len(pb):
        return None
    if len(pa) > 4000:
        pa = pa[np.linspace(0, len(pa) - 1, 4000).astype(int)]
    best = np.inf
    for i in range(0, len(pa), 500):
        dd = np.sqrt(((pa[i:i + 500, None, :] - pb[None, :, :]) ** 2).sum(-1)).min()
        best = min(best, float(dd))
    return best


suite = OriginalObjectSuite() if OBJ else OriginalSpatialSuite()
np.random.seed(7)
random.seed(7)
fh, gl_ok = None, False
t0 = time.time()
for tid in [int(t) for t in args.tasks.split(",")]:
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    envs = {}
    for i, (name, v) in enumerate(CELLS.items()):
        st = np.random.get_state()
        set_vars(v)
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        ref.seed(0)
        envs[name] = BodyEnv(ref, bddl, v["BODY"], 256, 10, log=lambda m: None)
        if i > 0:
            np.random.set_state(st)
        if not gl_ok:
            from OpenGL import GL
            envs[name].env.sim.render(width=64, height=64, camera_name="agentview")
            ren = GL.glGetString(GL.GL_RENDERER)
            ren = ren.decode() if isinstance(ren, bytes) else str(ren)
            print(f"T0CHECK_NEXT host={HOST} gpu={GPU} GL_RENDERER={ren} suite={args.suite} cells={CELLS}", flush=True)
            if "H200" not in ren and not args.allow_non_h200:
                print("GL_RENDERER does not contain H200 -> exit 2, no output", flush=True)
                sys.exit(2)
            gl_ok = True
            fh = open(args.out, "a")
    perm = np.random.RandomState(5478 + tid).permutation(len(inits))
    for ep in range(args.offset, args.offset + args.trials):
        init_id = int(perm[ep])
        st = np.random.get_state()
        obs = {}
        for name, v in CELLS.items():
            np.random.set_state(st)
            set_vars(v)
            envs[name].reset()
            obs[name] = envs[name].set_init_state(inits[init_id])
        rec = {"suite": args.suite, "task": tid, "episode": ep, "init": init_id, "host": HOST, "gpu": GPU, "gl_renderer": ren,
               "non_h200_debug": "H200" not in ren}
        if "sham_twin" in envs:
            s, t = envs["sham_cg"], envs["sham_twin"]
            set_vars(CELLS["sham_twin"])
            rec["t0_vs_s"] = {c: dstat(hid(t, t.twin.sim, cam), hid(s, s.env.sim, cam)) for c, cam in CAMS}
            rec["t0_vs_s_raw"] = {c: dstat(obs["sham_twin"][c], obs["sham_cg"][c]) for c, _ in CAMS}
        for tn, gn in PAIRS.items():
            if tn not in envs or gn not in envs:
                continue
            e, g = envs[tn], envs[gn]
            set_vars(CELLS[tn])
            ts = e.twin_stats or {}
            r = {"cam_err_mm0": (ts.get("cam_err_mm") or [None])[0], "cam_err_deg0": (ts.get("cam_err_deg") or [None])[0],
                 "ik_err_mm0": (ts.get("pos_err_mm") or [None])[0], "ik_rot_deg0": (ts.get("rot_err_deg") or [None])[0]}
            qe = e.env.sim.data.qpos[e.env.robots[0]._ref_joint_pos_indexes]
            qg = g.env.sim.data.qpos[g.env.robots[0]._ref_joint_pos_indexes]
            r["phys_joint_diff_vs_G"] = float(np.abs(qe - qg).max())
            r["phys_qpos_diff_vs_G"] = float(np.abs(e.env.sim.data.qpos - g.env.sim.data.qpos).max())
            if e.toy_parts:  # GTloc'
                for c, cam in CAMS:
                    arm = segmask(g, g.env.sim, cam, ("robot0", "mount0"))
                    rob = segmask(g, g.env.sim, cam, ("gripper0",))
                    gid_t = e._render(e.twin.sim, cam, e._toy_sets()[1], depth=False, seg=True)[2]
                    pand = np.isin(gid_t, e._toy_sets()[0])
                    comp = np.any(np.asarray(obs[tn][c]) != np.asarray(e._render(e.env.sim, cam, e._toy_sets()[2], depth=False)[0]), -1)
                    r[f"{c[:5]}_arm_region"] = dstat(obs[tn][c], obs[gn][c], arm & ~pand & ~rob)
                    r[f"{c[:5]}_outside"] = dstat(obs[tn][c], obs[gn][c], ~(arm | rob | pand))
                    palm = np.isin(gid_t, e._grip_parts(e.twin.sim.model)[0])
                    l7 = np.isin(e._render(e.env.sim, cam, [], depth=False, seg=True)[2], gg(e.env.sim.model, ("robot0_link7", "robot0_right_hand")))
                    r[f"{c[:5]}_gap_px"] = min_pix_dist(palm & comp, l7)
                    r[f"{c[:5]}_panda_px"] = int((pand & comp).sum())
            else:
                for c, cam in CAMS:
                    r[f"{c[:5]}_both_hidden"] = dstat(hid(e, e.twin.sim, cam), hid(g, g.env.sim, cam))
                ua = segmask(e, e.twin.sim, "agentview", ("robot0", "gripper0", "mount0")) | segmask(g, g.env.sim, "agentview", ("robot0", "gripper0", "mount0"))
                r["agent_background"] = dstat(hid(e, e.twin.sim, "agentview"), hid(g, g.env.sim, "agentview"), ~ua)
                r["M_search"] = search_M(e, g)
                r["agent_excess"] = None if r["M_search"]["M"] is None else r["agent_both_hidden"]["mean"] - r["M_search"]["M"]
            rec[tn] = r
        rec["sec"] = round(time.time() - t0, 1)
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(json.dumps({k: rec[k] for k in ("task", "episode", "sec")}), flush=True)
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
print("T0CHECK_NEXT done", flush=True)
