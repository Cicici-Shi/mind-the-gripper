"""t0check background diagnostic (2026-11-07). Diagnosis only: no policy, no outcomes, writes nothing outside --outdir.
Rebuilds the same cells / initial states as next/pf/t0check_next.py for a few keys and, for each twin pair
(GTfc vs its G cell), decomposes the agentview 'agent_background' difference (pixels outside the union of
robot0/gripper0/mount0 segmentation of both sims, gripper geoms alpha 0 on both sides):
  * reproduces t0check's dstat (sanity: must equal the jsonl value);
  * per differing pixel: channel max diff, distance (px, chessboard) to the union mask, geom under it in each sim;
  * re-renders with variants: shadows off / reflections off / both off (scene flags, both sims);
    gripper hidden by geom group 3 instead of alpha 0; every robot geom in group 3 (no robot at all);
  * the same outside-union statistic on the raw observation images (what the policy saw).
H200 only: asserts GL_RENDERER contains H200 (exit 2) unless --allow-non-h200.
Usage: python diag_bg.py --suite libero_object --tasks 4 --trials 1 --outdir DIR
"""
import argparse, json, os, random, socket, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--suite", default="libero_object")
ap.add_argument("--tasks", default="4")
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--outdir", required=True)
ap.add_argument("--allow-non-h200", action="store_true")
ap.add_argument("--save", type=int, default=1, help="1 = write npz / png per key")
args = ap.parse_args()
os.makedirs(args.outdir, exist_ok=True)
sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_next")))
from embody.body_env import BodyEnv  # noqa
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa
import mujoco  # noqa

CELLS_SH = os.path.expanduser("$PROJ_ROOT/vla_zoo/next/cells_next.sh")
OBJ = args.suite == "libero_object"
PRE = "pf_obj_" if OBJ else "pf_"
# same list / order as t0check_next.py so the RNG windows and initial states are identical
WANT = ["sham_cg", "sham_twin", "grpkp100a", "g_tfc_a", "g_tloc_a"] + (["grpkp100a_m20", "g_tfc_a_m20", "grpkp20", "g_tfc_kp20"] if OBJ else [])
PAIRS = {"g_tfc_a": "grpkp100a", "g_tfc_kp20": "grpkp20", "sham_twin": "sham_cg"}
CELLS = {}
for n in WANT:
    r = subprocess.run(["bash", CELLS_SH, "vars", PRE + n], capture_output=True, text=True)
    if r.returncode == 0 and r.stdout.strip():
        o = r.stdout.split()
        CELLS[n] = dict(BODY=o[0], **dict(kv.split("=", 1) for kv in o[1:]))
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "R85_ARMATURE", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM",
        "TWIN_CAMS", "TWIN_CAM_SYNC", "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_PHYS_GRIPPER", "TOY_TWIN_POSE",
        "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM"}
A = "agentview_image"


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


def seg(be, sim, cam="agentview"):
    return be._render(sim, cam, [], depth=False, seg=True)[2]


def flags(sim, shadow=None, refl=None):
    sc = sim._render_context_offscreen.scn
    old = (int(sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW]), int(sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION]))
    if shadow is not None:
        sc.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = shadow
    if refl is not None:
        sc.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = refl
    return old


def chess_dist(mask, cap=40):
    """chessboard distance of every pixel to `mask` (0 inside), capped."""
    d = np.full(mask.shape, cap, np.int32)
    cur = mask.copy()
    d[cur] = 0
    for k in range(1, cap):
        p = np.pad(cur, 1)
        nb = np.zeros_like(cur)
        for dy in (0, 1, 2):
            for dx in (0, 1, 2):
                nb |= p[dy:dy + cur.shape[0], dx:dx + cur.shape[1]]
        new = nb & ~cur
        d[new] = k
        cur = nb
        if cur.all():
            break
    return d


def gname(m, g):
    if g < 0:
        return "none"
    return f"{m.geom_id2name(g) or ''}@{m.body_id2name(m.geom_bodyid[g]) or ''}"


def analyse(e, g, tn, gn, obs_t, obs_g, tag):
    ts, gs = e.twin.sim, g.env.sim
    is_twin = e.twin is not None
    hide_t, hide_g = gg(ts.model), gg(gs.model)
    R = lambda be, sim, hide, how="alpha": be._render(sim, "agentview", hide, depth=False, how=how)[0]
    at, ag = R(e, ts, hide_t), R(g, gs, hide_g)
    ROB = ("robot0", "gripper0", "mount0")
    sid_t, sid_g = seg(e, ts), seg(g, gs)
    ut, ugm = np.isin(sid_t, gg(ts.model, ROB)), np.isin(sid_g, gg(gs.model, ROB))
    ua = ut | ugm
    base = dstat(at, ag, ~ua)
    D = np.abs(at.astype(np.int16) - ag.astype(np.int16)).max(-1)
    nz = (D > 0) & ~ua
    dist = chess_dist(ua)
    out = {"tag": tag, "pair": f"{tn} vs {gn}", "t0check_stat": base, "n_bg_px": int((~ua).sum()),
           "n_diff_px": int(nz.sum()), "n_diff_gt10": int(((D > 10) & ~ua).sum()), "n_diff_gt30": int(((D > 30) & ~ua).sum()),
           "max_px": int(D[~ua].max()) if (~ua).any() else 0}
    bins = [(1, 1), (2, 2), (3, 3), (4, 5), (6, 10), (11, 20), (21, 39), (40, 40)]
    out["diff_px_by_dist"] = {f"{a}-{b}": int((nz & (dist >= a) & (dist <= b)).sum()) for a, b in bins}
    out["max_by_dist"] = {f"{a}-{b}": int(D[nz & (dist >= a) & (dist <= b)].max()) if (nz & (dist >= a) & (dist <= b)).any() else 0 for a, b in bins}
    o2 = dist > 2  # outside the union dilated by 2 px (chessboard)
    out["dil2"] = {"n_px": int((nz & o2).sum()), "max": int(D[o2].max()) if o2.any() else 0}
    out["dist_hist_full"] = {int(k): int(v) for k, v in zip(*np.unique(dist[nz], return_counts=True))}
    ys, xs = np.nonzero(nz)
    out["bbox_yx"] = [int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max())] if len(ys) else None
    # which geom is under the differing pixels (in each sim)
    def hist(sid, m, msk):
        u, c = np.unique(sid[msk], return_counts=True)
        o = sorted(zip(c.tolist(), u.tolist()), reverse=True)[:6]
        return [[gname(m, int(gid)), int(n)] for n, gid in o]
    out["geom_under_diff_twin"] = hist(sid_t, ts.model, nz)
    out["geom_under_diff_G"] = hist(sid_g, gs.model, nz)
    big = nz & (D > 10)
    out["geom_under_diff_gt10_twin"] = hist(sid_t, ts.model, big)
    out["dist_of_gt10"] = {f"{a}-{b}": int((big & (dist >= a) & (dist <= b)).sum()) for a, b in bins}
    # sign: twin brighter or darker in differing pixels (shadow = darker on the side that has it)
    lum = lambda x: x.astype(np.float32).mean(-1)
    dl = lum(at) - lum(ag)
    out["twin_minus_G_lum_on_diff"] = {"mean": float(dl[nz].mean()) if nz.any() else 0.0,
                                       "frac_twin_darker": float((dl[nz] < 0).mean()) if nz.any() else 0.0}
    # variants
    var = {}
    for name, sh, rf in (("shadow_off", 0, None), ("refl_off", None, 0), ("shadow_refl_off", 0, 0)):
        o1, o2 = flags(ts, sh, rf), flags(gs, sh, rf)
        a2, b2 = R(e, ts, hide_t), R(g, gs, hide_g)
        flags(ts, *o1); flags(gs, *o2)
        DD = np.abs(a2.astype(np.int16) - b2.astype(np.int16)).max(-1)
        var[name] = {"stat": dstat(a2, b2, ~ua), "n_diff_px": int(((DD > 0) & ~ua).sum()),
                     "n_diff_px_dist_ge3": int(((DD > 0) & ~ua & (dist >= 3)).sum()),
                     "n_diff_px_dist_le2": int(((DD > 0) & ~ua & (dist <= 2)).sum())}
        if name == "refl_off":
            # 2026-11-07 addendum: stove-reflection pixels = differing in the normal render, identical with reflections off
            refl_px = nz & (DD == 0)
            out["refl_only"] = {"n_px": int(refl_px.sum()), "max": int(D[refl_px].max()) if refl_px.any() else 0,
                                "geom_twin": hist(sid_t, ts.model, refl_px)}
            o2r = (dist > 2)
            out["dil2_refl_off"] = {"n_px": int(((DD > 0) & o2r).sum()), "max": int(DD[o2r].max()) if o2r.any() else 0}
            out["dil2_refl_off_px"] = [[int(y), int(x), int(dist[y, x]), int(DD[y, x]), gname(ts.model, int(sid_t[y, x])),
                                        gname(gs.model, int(sid_g[y, x]))] for y, x in zip(*np.nonzero((DD > 0) & o2r))][:20]
        if name == "shadow_refl_off" and args.save:
            np.save(os.path.join(args.outdir, f"{tag}_D_shadow_refl_off.npy"), DD.astype(np.uint8))
    a3, b3 = R(e, ts, hide_t, "group"), R(g, gs, hide_g, "group")
    DD = np.abs(a3.astype(np.int16) - b3.astype(np.int16)).max(-1)
    var["gripper_group3"] = {"stat": dstat(a3, b3, ~ua), "n_diff_px": int(((DD > 0) & ~ua).sum()),
                             "vs_alpha_twin": dstat(a3, at), "vs_alpha_G": dstat(b3, ag)}
    rt, rg = gg(ts.model, ROB), gg(gs.model, ROB)
    a4, b4 = R(e, ts, rt, "group"), R(g, gs, rg, "group")
    DD = np.abs(a4.astype(np.int16) - b4.astype(np.int16)).max(-1)
    var["no_robot_group3"] = {"stat_outside_union": dstat(a4, b4, ~ua), "stat_full": dstat(a4, b4),
                              "n_diff_px_full": int((DD > 0).sum())}
    # raw observations (gripper visible, what the policy saw at step 0)
    var["raw_obs_outside_union"] = dstat(obs_t[A], obs_g[A], ~ua)
    Dr = np.abs(np.asarray(obs_t[A], np.int16) - np.asarray(obs_g[A], np.int16)).max(-1)
    var["raw_obs_n_diff_px_outside_union"] = int(((Dr > 0) & ~ua).sum())
    out["variants"] = var
    if not args.save:
        return out
    np.savez_compressed(os.path.join(args.outdir, f"{tag}.npz"), twin=at, G=ag, ua=ua, D=D.astype(np.uint8), dist=dist,
                        sid_t=sid_t, sid_g=sid_g)
    try:
        from PIL import Image
        vis = np.concatenate([at, ag], 1)
        m = np.zeros_like(at)
        m[ua] = (0, 0, 120)
        m[nz] = (255, 255, 0)
        m[big] = (255, 0, 0)
        vis = np.concatenate([vis, m], 1)
        Image.fromarray(vis[::-1] if False else vis).resize((vis.shape[1] * 3, vis.shape[0] * 3), Image.NEAREST).save(
            os.path.join(args.outdir, f"{tag}.png"))
    except Exception as ex:
        out["png_err"] = repr(ex)
    return out


suite = OriginalObjectSuite() if OBJ else OriginalSpatialSuite()
np.random.seed(7)
random.seed(7)
gl_ok, ren = False, None
res = []
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
            print(f"DIAG_BG host={socket.gethostname()} GL_RENDERER={ren} suite={args.suite}", flush=True)
            if "H200" not in ren and not args.allow_non_h200:
                print("GL_RENDERER does not contain H200 -> exit 2", flush=True)
                sys.exit(2)
            gl_ok = True
            m = envs[name].env.sim.model._model
            info = {"offsamples": int(m.vis.quality.offsamples), "shadowsize": int(m.vis.quality.shadowsize),
                    "nlight": int(m.nlight), "light_castshadow": m.light_castshadow.tolist(),
                    "mat_reflectance": {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i): float(m.mat_reflectance[i])
                                        for i in range(m.nmat) if m.mat_reflectance[i] > 0}}
            print("MODEL_INFO", json.dumps(info), flush=True)
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
        for tn, gn in PAIRS.items():
            if tn not in envs or gn not in envs:
                continue
            set_vars(CELLS[tn])
            tag = f"{args.suite}_t{tid}_e{ep}_{tn}"
            r = analyse(envs[tn], envs[gn], tn, gn, obs[tn], obs[gn], tag)
            r.update({"suite": args.suite, "task": tid, "episode": ep, "init": init_id})
            print("DIAG", json.dumps(r), flush=True)
            res.append(r)
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
json.dump(res, open(os.path.join(args.outdir, f"diag_{args.suite}_{args.tasks.replace(',', '')}.json"), "w"), indent=1)
print("DIAG_BG done", flush=True)
