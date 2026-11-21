"""rescue PROTOTYPE (not a cell, no policy, no outcome), 2026-11-08: "outline rescue" on G2 (Robotiq kp20 physics).
R1 = the G2 image (physical Robotiq visible) with the outline of the twin Panda gripper (pose of GTfc2) drawn in black.
R2 = the physical render with the Robotiq gripper geoms at alpha 0 (maskinfo C3/C4 alpha0 way) + the same outline.
Outline rule = maskinfo C4: O = outline(G, w=3) (1 inner + 2 outer), drawn pixels = O minus P_full, colour (0,0,0).
  G      = visible twin Panda gripper pixels (geoms of gripper0* bodies) in the twin render of that camera; the twin is
           posed exactly as in GTfc2 (body_env._twin_paint_inner: IK of the twin grip site onto the Robotiq pads,
           TWIN_CAM=phys pins the twin wrist camera on the physical one, TWIN_CAM_SYNC=1).
  P_full = protected "interactive" set (maskinfo _p_set rule: obj_of_interest u LIBERO movable objects u articulated
           fixture parts) in the PHYSICAL render with the Robotiq at alpha 0 (gripper absent), as maskinfo C4.
erode4 / outline (maskinfo.py) and _dilate_sq (maskrepl.py) are exec'd from the frozen files' source (ast), not copied.
Everything is computed inside the GTfc2 paint (physical data forwarded by TWIN_CAM_SYNC), by wrapping the instance's
_twin_paint_inner; the frozen embody_next code is not modified.
Per checked frame: IoU of G with the Panda footprint of the GTfc2 image (pixels where the GTfc2 twin image differs from
a twin render with the twin gripper at alpha 0, an RGB-side check of the segmentation), O inner ring inside that
footprint, IoU(G, Robotiq G_phys), outline px drawn / truncated by P_full, pixels changed outside the drawn set (must
be 0), P_full pixels changed (must be 0), base-image lag (G2 obs vs synced re-render). Timing: ms per env step for
G2, GTfc2 and the extra R work, split into twin seg / physical alpha render / numpy.
Scripted arm motion only (down, close, up, sideways): no policy, outcome never computed.
Usage: python outline_proto.py --out DIR [--keys 0:0,4:0,8:0] [--steps 60] [--png-every 15]
"""
import argparse, ast, hashlib, json, os, random, socket, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--keys", default="0:0,4:0,8:0")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--png-every", type=int, default=15)
ap.add_argument("--width", type=int, default=3)
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
H = os.path.expanduser
sys.path.insert(0, H("$PROJ_ROOT/embody_next"))
from embody.body_env import BodyEnv  # noqa
sys.path.insert(0, H("$PROJ_ROOT/vla_zoo/common"))
import m1_fastsync  # noqa: E402  (same fast bitwise TWIN_CAM_SYNC snapshot as launch_pf.py / the collection runs)
m1_fastsync.patch()
from embody.original import OriginalSpatialSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa

MI, MR = H("$PROJ_ROOT/maskinfo/embody/maskinfo.py"), H("$PROJ_ROOT/maskinfo/embody/maskrepl.py")
NS = {"np": np}
for path, names in ((MR, ("_dilate_sq",)), (MI, ("erode4", "outline"))):
    src = open(path).read()
    for node in ast.parse(src).body:
        if isinstance(node, ast.FunctionDef) and node.name in names:
            exec(compile(ast.Module([node], []), path, "exec"), NS)
outline = NS["outline"]
SHA = {p: hashlib.sha256(open(p, "rb").read()).hexdigest()[:16] for p in (MI, MR, H("$PROJ_ROOT/embody_next/embody/body_env.py"))}

G2 = dict(BODY="PandaR85", GRP_MAP="kp20", COPY_GRIP_CMD="1")
TW = dict(TWIN_RENDER="1", TWIN_BODY="Panda", TWIN_CAMS="both", TWIN_CAM="phys", TWIN_CAM_SYNC="1")
CELLS = {"G2": dict(G2), "GT": dict(G2, **TW)}
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "R85_ARMATURE", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM",
        "TWIN_CAMS", "TWIN_CAM_SYNC", "HIDE_GRIPPER", "TOY_PARTS", "TOY_PHYS_GRIPPER", "TOY_TWIN_POSE", "TOY_PIXSTATS"}
CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))
GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()


def set_vars(v):
    for k in ALLV:
        os.environ.pop(k, None)
    os.environ.update(v)


def grip_geoms(m):
    return [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")]


def p_geoms(rs):
    """maskinfo MaskInfoHook._p_set, protect_set = interactive (re-implemented on the physical env)."""
    m = rs.sim.model
    g2i = {int(g): inst for g, inst in rs.model.geom_ids_to_instances.items()}
    objs = set(getattr(rs, "objects_dict", {}).keys()) | set(getattr(rs, "obj_of_interest", []) or [])
    geoms = {g for g, i in g2i.items() if i in objs}
    for fname, fx in getattr(rs, "fixtures_dict", {}).items():
        if fname in objs:
            continue
        root = m.body_name2id(fx.root_body)
        for g, inst in g2i.items():
            if inst != fname:
                continue
            b, chain = int(m.geom_bodyid[g]), []
            while b != root and b > 0:
                chain.append(b)
                b = int(m.body_parentid[b])
            if any(m.body_jntnum[c] > 0 for c in chain):
                geoms.add(g)
    return sorted(geoms)


def iou(a, b):
    u = int((a | b).sum())
    return None if u == 0 else float((a & b).sum()) / u


class Rescue:
    """Wraps env._twin_paint_inner of a GTfc2 BodyEnv; after the GTfc2 paint builds R1 / R2 for both cameras."""

    def __init__(self, env, width):
        self.env, self.w = env, width
        self.orig = env._twin_paint_inner
        env._twin_paint_inner = self.paint
        self.orig_probe = env._twin_obs_probe
        env._twin_obs_probe = self.probe
        self.rows, self.imgs, self.tm = [], {}, {"gt_inner": [], "twin_rgb_obs": [], "twin_seg": [], "phys_alpha": [], "np": [], "check": []}
        self.pg = None

    def probe(self):
        t0 = time.perf_counter()
        o = self.orig_probe()
        self.tm["twin_rgb_obs"].append(time.perf_counter() - t0)
        return o

    def paint(self, obs):
        e = self.env
        phys = {k: np.array(obs[k], copy=True) for k, _ in CAMS}  # the G2 image (rendered by env.step, before paint)
        t0 = time.perf_counter()
        out = self.orig(obs)
        self.tm["gt_inner"].append(time.perf_counter() - t0)
        if self.pg is None:  # per episode (reset may rebuild the model)
            self.pg = p_geoms(e.env.env)
        mt, mp = e.twin.sim.model, e.env.sim.model
        tg, pgrip = grip_geoms(mt), grip_geoms(mp)
        res = {"R1": {}, "R2": {}, "GT": {k: np.array(out[k], copy=True) for k, _ in CAMS}, "G2": phys}
        row = {}
        for key, cam in CAMS:
            t1 = time.perf_counter()
            _, _, gid_t = e._render(e.twin.sim, cam, [], depth=False, seg=True)
            t2 = time.perf_counter()
            rgb_ph, _, gid_ph = e._render(e.env.sim, cam, pgrip, depth=False, seg=True, how="alpha")
            t3 = time.perf_counter()
            G = np.isin(gid_t, tg)
            Pfull = np.isin(gid_ph, self.pg)
            O = outline(G, self.w)
            draw = O & ~Pfull
            r1 = phys[key].copy(); r1[draw] = 0
            r2 = np.asarray(rgb_ph).copy(); r2[draw] = 0
            t4 = time.perf_counter()
            self.tm["twin_seg"].append(t2 - t1); self.tm["phys_alpha"].append(t3 - t2); self.tm["np"].append(t4 - t3)
            # ---- checks (not part of the cell cost)
            rgb_tn = e._render(e.twin.sim, cam, tg, depth=False, seg=False, how="alpha")[0]
            foot = np.any(np.asarray(res["GT"][key]) != np.asarray(rgb_tn), axis=2)
            _, _, gid_p = e._render(e.env.sim, cam, [], depth=False, seg=True)
            rgb_p0 = e._render(e.env.sim, cam, [], depth=False, seg=False)[0]
            Gp = np.isin(gid_p, pgrip)
            Pvis = np.isin(gid_p, self.pg)
            inner = O & G
            ch1 = np.any(r1 != phys[key], axis=2)
            ch2 = np.any(r2 != np.asarray(rgb_ph), axis=2)
            self.tm["check"].append(time.perf_counter() - t4)
            row[cam] = {"G_px": int(G.sum()), "foot_px": int(foot.sum()), "iou_G_footGT": iou(G, foot),
                        "inner_in_foot": None if inner.sum() == 0 else float((inner & foot).sum()) / int(inner.sum()),
                        "Gphys_px": int(Gp.sum()), "iou_G_Gphys": iou(G, Gp),
                        "O_px": int(O.sum()), "draw_px": int(draw.sum()), "trunc_Pfull_px": int((O & Pfull).sum()),
                        "trunc_if_Pvis_px": int((O & Pvis).sum()), "Pfull_px": int(Pfull.sum()), "Pvis_px": int(Pvis.sum()),
                        "O_on_Robotiq_px": int((draw & Gp).sum()),
                        "R1_changed_outside_draw": int((ch1 & ~draw).sum()), "R2_changed_outside_draw": int((ch2 & ~draw).sum()),
                        "R1_Pfull_changed": int((ch1 & Pfull).sum()), "R2_Pfull_changed": int((ch2 & Pfull).sum()),
                        "base_lag_px": int(np.any(phys[key] != np.asarray(rgb_p0), axis=2).sum()),
                        "base_lag_mean": float(np.abs(phys[key].astype(np.int16) - np.asarray(rgb_p0, np.int16)).mean()),
                        "orient_check_iou_Gphys_obs": None}
            res["R1"][key], res["R2"][key] = r1, r2
            res.setdefault("O", {})[key] = O
        self.rows.append(row)
        self.last = res
        return out


def strip(res, key):
    ims = [res[n][key] for n in ("G2", "GT", "R1", "R2")]
    return np.concatenate([np.asarray(x)[::-1, ::-1] for x in ims], 1)  # policy orientation (180 deg), as hand_feas_pf


def script_action(i):
    # down 20 (open), close 15, up 15, sideways 10 (closed): arm deltas only, no policy
    if i < 20:
        return [0, 0, -0.6, 0, 0, 0, -1]
    if i < 35:
        return [0, 0, 0, 0, 0, 0, 1]
    if i < 50:
        return [0, 0, 0.6, 0, 0, 0, 1]
    return [0.5, 0.5, 0, 0, 0, 0.3, 1]


suite = OriginalSpatialSuite()
keys = [tuple(int(x) for x in k.split(":")) for k in args.keys.split(",")]
summ = {"host": socket.gethostname(), "gpu": GPU, "sha": SHA, "width": args.width, "keys": keys, "per_key": []}
print(f"OUTLINE_PROTO host={summ['host']} gpu={GPU} sha={SHA}", flush=True)
import imageio  # noqa
np.random.seed(7); random.seed(7)
for tid, ep in keys:
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
    rsc = Rescue(envs["GT"], args.width)
    init_id = int(np.random.RandomState(5478 + tid).permutation(len(inits))[ep])
    st = np.random.get_state()
    obs = {}
    for name, v in CELLS.items():
        np.random.set_state(st)
        set_vars(v)
        envs[name].reset()
        obs[name] = envs[name].set_init_state(inits[init_id])
    # G2 env and the GT env's physical image must agree (same physics, same scene)
    g2_vs_base = {k: int(np.any(np.asarray(obs["G2"][k]) != rsc.last["G2"][k], axis=2).sum()) for k, _ in CAMS}
    tm = {"G2": [], "GT": []}
    frames = [0]
    for name in ("G2", "GT"):
        set_vars(CELLS[name])
    for s in range(args.steps):
        a = script_action(s)
        for name in ("G2", "GT"):
            set_vars(CELLS[name])
            t0 = time.perf_counter()
            o, _, _, _ = envs[name].step(a)
            tm[name].append(time.perf_counter() - t0)
            obs[name] = o
        if (s + 1) % args.png_every == 0:
            frames.append(s + 1)
            for key, cam in CAMS:
                imageio.imwrite(os.path.join(args.out, f"t{tid}_e{ep}_s{s + 1:03d}_{cam[:5]}.png"), strip(rsc.last, key))
        if s == 0:  # also the first frame strips (set_init_state paint is overwritten by now: use step 1)
            for key, cam in CAMS:
                imageio.imwrite(os.path.join(args.out, f"t{tid}_e{ep}_s001_{cam[:5]}.png"), strip(rsc.last, key))
    rows = rsc.rows

    def agg(cam, k, f=np.nanmedian):
        v = [r[cam][k] for r in rows if r[cam][k] is not None]
        return None if not v else float(f(np.asarray(v, float)))
    def at(cam, k, idx):
        return [rows[i][cam][k] for i in idx if i < len(rows)]
    pk_lag = {cam: {"init_paint": at(cam, "base_lag_px", [0]), "static_close_phase_med": float(np.median(at(cam, "base_lag_px", range(22, 35)))),
                    "moving_down_med": float(np.median(at(cam, "base_lag_px", range(2, 20))))} for _, cam in CAMS}
    pk = {"task": tid, "base_lag_by_phase": pk_lag, "ep": ep, "init": init_id, "n_paints": len(rows), "g2_vs_GTbase_px_t0": g2_vs_base,
          "ms_step": {n: round(1000 * float(np.mean(v)), 1) for n, v in tm.items()},
          "ms_paint_parts": {k: round(1000 * float(np.mean(v)) * (1 if k == "gt_inner" else 1), 2) for k, v in rsc.tm.items()},
          "per_cam": {}}
    for _, cam in CAMS:
        pk["per_cam"][cam] = {
            "iou_G_footGT_med": agg(cam, "iou_G_footGT"), "iou_G_footGT_min": agg(cam, "iou_G_footGT", np.nanmin),
            "inner_in_foot_min": agg(cam, "inner_in_foot", np.nanmin),
            "iou_G_Gphys_med": agg(cam, "iou_G_Gphys"), "G_px_med": agg(cam, "G_px"), "Gphys_px_med": agg(cam, "Gphys_px"),
            "O_px_med": agg(cam, "O_px"), "draw_px_med": agg(cam, "draw_px"),
            "trunc_Pfull_px_max": agg(cam, "trunc_Pfull_px", np.max), "trunc_frac_frames": float(np.mean([r[cam]["trunc_Pfull_px"] > 0 for r in rows])),
            "trunc_if_Pvis_px_max": agg(cam, "trunc_if_Pvis_px", np.max),
            "O_on_Robotiq_frac_med": float(np.median([r[cam]["O_on_Robotiq_px"] / max(1, r[cam]["draw_px"]) for r in rows])),
            "R1_changed_outside_draw_max": agg(cam, "R1_changed_outside_draw", np.max),
            "R2_changed_outside_draw_max": agg(cam, "R2_changed_outside_draw", np.max),
            "R1_Pfull_changed_max": agg(cam, "R1_Pfull_changed", np.max), "R2_Pfull_changed_max": agg(cam, "R2_Pfull_changed", np.max),
            "base_lag_px_med": agg(cam, "base_lag_px"), "base_lag_px_max": agg(cam, "base_lag_px", np.max),
            "base_lag_mean_max": agg(cam, "base_lag_mean", np.max)}
    summ["per_key"].append(pk)
    json.dump(rows, open(os.path.join(args.out, f"rows_t{tid}_e{ep}.json"), "w"))
    print(json.dumps(pk), flush=True)
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
json.dump(summ, open(os.path.join(args.out, "summary.json"), "w"), indent=1)
print("done", flush=True)
