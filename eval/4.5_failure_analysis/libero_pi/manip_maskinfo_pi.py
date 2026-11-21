"""maskinfo-pi (2026-11-08): copy of $PROJ_ROOT/maskinfo/tools/manip_maskinfo.py with the openpi branch of
protocol_maskinfo_pi.md v2 2.6: production hook embody.maskinfo_pi.MaskInfoPiHook (cadence --every 5, lag --lag 15,
openpi model-input path), armed every 5 steps with its step number, cells C0..C8 (C8 expected inside G, same mask as
C1), item 6 edge on both native preprocessings, item 7 IoU at --lag and --lag-alt (decision rule 15 -> 20 if the
agentview median > 0.7), item 9 = three md5 (hook 224 / client make_element 224 / independent PIL 224) + same-moment
state check, item 15 = visibility checks (i) actual input == independently generated expected input, (ii) protected
outline not completely vanished (C4 vs C3, C5 vs C3, intersected with the projected protected outline), plus the
reported quantities (visible px, local contrast, truncation, P_vis retention, wrist bottom-edge G, effective edit
region; both preprocessings). Everything else is the frozen tool unchanged.

Original docstring:
maskinfo manipulation checks on cached states (protocol_maskinfo.md 2.4, items 1-10) + same-state render determinism
(4 item 1) + physics invariance (4 item 5). GPU (EGL), no policy, outcomes never read. 2026-11-06.

State source (protocol 2.4): Spatial = maskrepl S_hook dumps (ep 40-49, merged OFT), Object = third-batch obj_s dumps
(ep 0-19). Only the ACTIONS (and step-0 qpos / reference fixtures for the scene check) are read.
Per key, ONE process, ONE BodyEnv: the np.random stream is replayed as in the eval runner (seed 7 at the shard start,
tasks in order, one env per task, one reset per episode); set_init_state with the hook armed in C0 (the real frame-0
path), then a toy2 snapshot; every cell is restored from that snapshot, its frame 0 re-rendered under the armed hook
(forced observable update) and the dump actions replayed with the hook armed every 8 steps (as the runner does).
All eight cells therefore see bit-identical physics (checked: item 10) and the comparisons are per pixel.

Output: one jsonl row per (key, query frame, view) with every item's raw numbers, one row per key with physics /
scene / replay fidelity, and side-by-side sheets (8 cells x 2 views, policy orientation) for the first --png-keys keys.
Usage: python tools/manip_maskinfo.py --suite spatial|object --out FILE [--offsets 40,45] [--tasks 0..9]
       [--png-dir DIR] [--lag 16] [--width 2] [--protect interactive] [--det-cells C0,C3]
"""
import argparse, glob, hashlib, json, os, random, socket, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--suite", required=True, choices=["spatial", "object"])
ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--only-first", type=int, default=1, help="keys = the first N episodes of each shard")
ap.add_argument("--dumps", default="")
ap.add_argument("--cells", default="C0,C1,C2,C3,C4,C5,C6,C7,C8")
ap.add_argument("--lag", type=int, default=15)
ap.add_argument("--lag-alt", type=int, default=20)
ap.add_argument("--every", type=int, default=5)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--protect", default="interactive")
ap.add_argument("--det-cells", default="C0,C3")
ap.add_argument("--png-dir", default="")
ap.add_argument("--png-keys", type=int, default=3)
ap.add_argument("--fix-scene", type=int, default=1)
ap.add_argument("--max-steps", type=int, default=0, help="cap the replay (debug; t0 mode uses 24)")
ap.add_argument("--t0", action="store_true", help="protocol 4 item 3 (t0gate_mi): every cell from its own real "
                "set_init_state (same RNG state, reference fixtures of the first configuration written back after each "
                "reset), frame 0 + replay of --max-steps actions; actions borrowed from --action-ep-shift")
ap.add_argument("--action-ep-shift", type=int, default=None, help="t0 mode: actions of key (t, ep + shift) from "
                "--dumps (default: spatial +20 -> S_hook ep 40-49, object -20 -> obj_s ep 0-9; snapshot mode 0)")
args = ap.parse_args()
if args.action_ep_shift is None:
    args.action_ep_shift = (20 if args.suite == "spatial" else -20) if args.t0 else 0
MI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MI)
from embody import maskinfo as mi  # noqa: E402
from embody import maskinfo_pi as mip  # noqa: E402
from embody import maskrepl as mr  # noqa: E402
from embody import snapshot as snapmod  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalObjectSuite, OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
import importlib.util as _ilu  # noqa: E402
_sp = _ilu.spec_from_file_location("zoo_client_mi", os.path.join(MI, "pi", "zoo_client_mi.py"))
ZC = _ilu.module_from_spec(_sp)
_sp.loader.exec_module(ZC)  # the client module itself: make_element is what the client sends
PI_IMG, PI_MASK = mip.PREP["openpi"]
OFT_IMG, OFT_MASK = mip.PREP["oft"]  # frozen OFT functions, captured before the swap


def indep224(img):
    """independent implementation of the openpi image path: PIL bilinear 256 -> 224 after the 180 deg rotation."""
    return np.asarray(Image.fromarray(np.ascontiguousarray(np.asarray(img)[::-1, ::-1])).resize((224, 224), Image.BILINEAR))


def indep_outline(G, width):
    """independent outline (scipy.ndimage; frame outside treated as G for the inner boundary), 1 in + 2 out."""
    k_in, k_out = width // 2, width - width // 2
    inner = (G & ~ndimage.binary_erosion(G, structure=ndimage.generate_binary_structure(2, 1), iterations=k_in,
                                         border_value=1)) if k_in else np.zeros_like(G)
    outer = ndimage.binary_dilation(G, structure=np.ones((3, 3), bool), iterations=k_out) & ~G
    return inner | outer

for k in ("BODY", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "HIDE_GRIPPER", "TOY_PIXSTATS", "MASK_CELL", "MI_CELL"):
    os.environ.pop(k, None)
os.environ["COPY_GRIP_CMD"] = "1"
try:
    GPU = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"], capture_output=True,
                         text=True, timeout=30).stdout.strip().splitlines()[0]
except Exception as e:
    GPU = f"unknown ({e!r})"
HOST = socket.gethostname()
CELLS = args.cells.split(",")
DET = set(args.det_cells.split(",")) if args.det_cells else set()
SUITE = OriginalSpatialSuite() if args.suite == "spatial" else OriginalObjectSuite()
DUMPS = args.dumps or (os.path.expanduser("$PROJ_ROOT/maskrepl/out/runs/S_hook") if args.suite == "spatial"
                       else os.path.expanduser("$PROJ_ROOT/runs/obj_s"))
OFFS = [int(x) for x in (args.offsets or ("40,45" if args.suite == "spatial" else "0,5")).split(",")]
VIEWS = (("agentview_image", "agentview", "agent"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand", "wrist"))
OPEN_THR = 0.039 / 2  # protocol 2.4 item 7: half of Panda full opening (finger joint 1)
md5 = lambda a: hashlib.md5(np.ascontiguousarray(a).tobytes()).hexdigest()


def load_dump(tid, ep):
    ep = ep + args.action_ep_shift
    f = glob.glob(os.path.join(DUMPS, f"t{tid}_e{ep}_i*.npz"))
    if not f:
        return None
    z = np.load(f[0], allow_pickle=True)
    out = {"action": np.asarray(z["action"]), "qpos0": np.asarray(z["qpos"][0]),
           "fix": ([str(x) for x in z["ref_body_names"]], np.asarray(z["ref_body_pos"]), np.asarray(z["ref_body_quat"]))}
    if "mr_img_agent" in z.files:  # S_hook query images (frame steps mr_steps) for replay fidelity
        out["img"] = {"agent": np.asarray(z["mr_img_agent"]), "wrist": np.asarray(z["mr_img_wrist"]),
                      "steps": np.asarray(z["mr_steps"])}
    return out  # success is never read


def write_fixtures(e, fix):
    names, pos, quat = fix
    rm = e.ref.sim.model
    for i, n in enumerate(names):
        if not n:
            continue
        try:
            bid = rm.body_name2id(n)
        except (ValueError, KeyError):
            continue
        rm.body_pos[bid] = pos[i]
        rm.body_quat[bid] = quat[i]


def ndiff(a, b, region=None):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).max(axis=2)
    if region is not None:
        d = d[region]
    return int((d > 0).sum()), int(d.max()) if d.size else 0


def tol_ok(n, mx):
    """maskrepl cross-configuration rule: identical, or at most 10 pixels with max diff <= 1."""
    return n == 0 or (n <= 10 and mx <= 1)


def grab(hook, obs, step):
    """Copy what the checks need out of the hook after finish(): images, masks, rows."""
    fr = hook.ep["frames"][-1]
    row = hook.ep["rows"][-1]
    out = {"step": step, "row": row, "mi": row.get("mi", {}), "P_names": hook.ep["protect_names"],
           "inst_names": hook.ep["instance_names"]}
    for key, cam, v in VIEWS:
        f = fr.get(v, {})
        out[v] = {"final": np.array(obs[key]), "bg": f.get("bg"), "G": f.get("Gm"),
                  "model_md5": f.get("model_md5"), "E": np.unpackbits(f["M"])[: 256 * 256].reshape(256, 256).astype(bool)}
        c = hook._last_cap.get(cam, {})
        for k in ("P_full", "P_vis", "R", "union", "mi_O", "mi_lag_step"):
            if k in c:
                out[v][k] = c[k]
        tg = hook.ep["target_geoms"]
        if "bg_gid" in c:
            out[v]["T_full"] = np.isin(c["bg_gid"], tg)
            out[v]["T_vis"] = np.isin(c["gid"], tg)
            ids = set(np.unique(c["bg_gid"]).tolist())
            out[v]["Pnames"] = {n: int(bool(ids & set(gs))) for n, gs in hook.ep["P_name_geoms"].items()}
            out[v]["Ppx"] = {n: int(np.isin(c["bg_gid"], gs).sum()) for n, gs in hook.ep["P_name_geoms"].items()}
            if "_np_geoms" not in hook.ep:  # instance geoms that are neither P nor robot / gripper (cabinet frame, stove)
                rob = set(hook.ep["robot_names"]) | {hook.ep["gripper_inst_name"]}
                hook.ep["_np_geoms"] = np.asarray(sorted(int(g) for g, i in hook.rs.model.geom_ids_to_instances.items()
                                                         if i not in rob and int(g) not in set(hook.ep["P_geoms"].tolist())))
            out[v]["NP"] = np.isin(c["bg_gid"], hook.ep["_np_geoms"])
        # protocol 2.4 item 9: model input md5 along the runner's own path (get_libero_image + resize_image_for_policy)
        if step == 0 and "bg_gid" in c:
            m = hook.rs.sim.model
            u, n = np.unique(c["bg_gid"][c["bg_gid"] >= 0], return_counts=True)
            bp = {}
            for g, k in zip(u, n):
                b = m.body_id2name(int(m.geom_bodyid[int(g)]))
                bp[b] = bp.get(b, 0) + int(k)
            out.setdefault("_body_px", {})[v] = bp
        el = out.setdefault("_el", ZC.make_element(obs, "x", 224))
        out[v]["runner_md5"] = md5(el["observation/image" if v == "agent" else "observation/wrist_image"])
        out[v]["indep_md5"] = md5(indep224(obs[key]))
        out[v]["clean"] = f.get("clean")
    out["mi_S"] = row.get("mi_S")
    return out


class Hook(mip.MaskInfoPiHook):
    """Keeps a reference to the last capture dict (finish() drops it); everything else is the production hook."""
    def finish(self, obs, step, model_md5=None):
        self._last_cap = self.cap or {}
        return super().finish(obs, step, model_md5)


OBS_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value")


def obs_timers(e):
    """robosuite Observable sampling state (not part of the toy2 snapshot): which sub-step of a control step samples
    the camera depends on it, so it must be restored with the physics for per-pixel comparisons across cells."""
    return {k: {f: np.array(getattr(o, f), copy=True) if isinstance(getattr(o, f), np.ndarray) else getattr(o, f)
                for f in OBS_FIELDS} for k, o in e.env.env._observables.items()}


def set_obs_timers(e, st):
    for k, o in e.env.env._observables.items():
        for f, v in st[k].items():
            setattr(o, f, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)


def run_cell(e, hook, snap, cell, actions):
    snapmod.restore(e, snap)
    set_obs_timers(e, snap["mi_obs_timers"])
    hook.set_cell(cell)
    hook.det_check = cell in DET
    hook.begin_episode(e._mi_task, e._mi_ep)
    hook.arm(0)
    obs = e._fix(e.env.env._get_observations(force_update=True))
    hook.finish(obs, 0)
    frames = [grab(hook, obs, 0)]
    set_obs_timers(e, snap["mi_obs_timers"])  # the forced frame-0 render must not shift later sampling
    q = [np.array(e.env.sim.data.qpos, copy=True)]
    gq = [np.array(obs["robot0_gripper_qpos"], copy=True)]
    n = len(actions) if not args.max_steps else min(args.max_steps, len(actions))
    for i in range(n):
        armed = (i + 1) % args.every == 0
        if armed:
            hook.arm(i + 1)
        obs, _, done, _ = e.step(np.asarray(actions[i]).tolist())
        q.append(np.array(e.env.sim.data.qpos, copy=True))
        gq.append(np.array(obs["robot0_gripper_qpos"], copy=True))
        if armed:
            hook.finish(obs, i + 1)
            frames.append(grab(hook, obs, i + 1))
        if done:
            break
    return frames, np.asarray(q), np.asarray(gq)


def light(fr):
    """Drop the big arrays of a frame that later checks do not need (keep final + masks)."""
    for v in ("agent", "wrist"):
        for k in ("clean", "gid", "bg_gid"):
            fr[v].pop(k, None)
    return fr


def body_px(hook, fr0):
    """frame-0 pixel count per MuJoCo body of every segmentation instance (both views), for the P-list check."""
    return fr0.get("_body_px", {})


def iou(a, b):
    u = (a | b).sum()
    return None if u == 0 else float((a & b).sum() / u)


def edge224(img, bg, G, prep="openpi"):
    f_img, f_mask = (PI_IMG, PI_MASK) if prep == "openpi" else (OFT_IMG, OFT_MASK)
    fe, _ = f_img(img)
    be, _ = f_img(bg)
    Gm = f_mask(G)
    z = np.zeros_like(Gm)
    return mr.section5(fe, be, z, Gm, z, z)["edge"]


def frame_checks(t, ep, F, gqs):
    """F[cell] = list of frames (same query steps across cells). Returns jsonl rows, one per (frame, view)."""
    rows = []
    base = F["C0"]
    steps = [f["step"] for f in base]
    gq0 = gqs["C0"]
    for qi, st in enumerate(steps):
        for key, cam, v in VIEWS:
            r = {"task": t, "ep": ep, "step": st, "view": v}
            b = base[qi][v]
            G, Tf, Tv = b["G"], b["T_full"], b["T_vis"]
            Pf, Pv = b["P_full"], b["P_vis"]
            S0 = np.any(b["final"] != b["bg"], axis=2) & ~G  # C0 vs its own alpha-0 render outside G (shadow band)
            cat = {}
            for c in CELLS[1:]:
                x = F[c][qi][v]
                D = np.any(x["final"] != b["final"], axis=2)
                cat[c] = {"n": int(D.sum()), "in_G": int((D & G).sum()), "in_B_not_G": int((D & x.get("union", np.zeros_like(G)) & ~G).sum()),
                          "in_T_full": int((D & Tf).sum()), "in_T_vis": int((D & Tv).sum()), "in_P_full": int((D & Pf).sum()),
                          "in_P_vis": int((D & Pv).sum())}
            # item 1: changed pixels vs C0 outside each cell's expected set (cross-render noise rule)
            exp = {}
            B2 = F["C2"][qi][v]["union"] if "C2" in F else None
            if "C1" in F:
                exp["C1"] = G
            if "C2" in F:
                exp["C2"] = B2
            if "C3" in F:
                exp["C3"] = G | S0
            if "C4" in F:
                exp["C4"] = G | S0 | F["C4"][qi][v]["E"]
            if "C5" in F:
                exp["C5"] = G | S0 | F["C5"][qi][v]["E"]
            if "C6" in F:
                exp["C6"] = F["C6"][qi][v]["union"] & ~F["C6"][qi][v]["P_vis"]
            if "C7" in F:
                exp["C7"] = F["C7"][qi][v]["E"]
            if "C8" in F:
                exp["C8"] = G
            item1 = {}
            for c, Ex in exp.items():
                n, mx = ndiff(F[c][qi][v]["final"], b["final"], ~Ex)
                item1[c] = {"out_n": n, "out_max": mx, "ok": tol_ok(n, mx)}
            # finer: alpha cells against C0's own I_bg, C4/C5 against C3, C6 against C2 / C0
            if "C3" in F:
                n, mx = ndiff(F["C3"][qi][v]["final"], b["bg"])
                item1["C3_vs_C0_Ibg"] = {"n": n, "max": mx, "ok": tol_ok(n, mx)}
                for c in ("C4", "C5"):
                    if c in F:
                        E = F[c][qi][v]["E"]
                        n, mx = ndiff(F[c][qi][v]["final"], F["C3"][qi][v]["final"], ~E)
                        drawn_black = bool((F[c][qi][v]["final"][E] == 0).all())
                        item1[f"{c}_vs_C3_outside_O"] = {"n": n, "max": mx, "ok": tol_ok(n, mx), "drawn_all_black": drawn_black}
            if "C6" in F and "C2" in F:
                Pv6 = F["C6"][qi][v]["P_vis"]
                n1, m1 = ndiff(F["C6"][qi][v]["final"], F["C2"][qi][v]["final"], ~Pv6)
                n2, m2 = ndiff(F["C6"][qi][v]["final"], b["final"], Pv6)
                item1["C6_vs_C2_outside_Pvis"] = {"n": n1, "max": m1, "ok": tol_ok(n1, m1)}
                item1["C6_vs_C0_in_Pvis"] = {"n": n2, "max": m2, "ok": tol_ok(n2, m2)}
                item1["C6_union_eq_C2_union"] = bool(np.array_equal(F["C6"][qi][v]["union"], B2))
            if "C8" in F and "C1" in F:  # protocol v2 2.6 item 1: same MASK, not same changed-pixel set
                item1["C8_mask_eq_C1"] = {"ok": bool(np.array_equal(F["C8"][qi][v]["E"], F["C1"][qi][v]["E"]))}
            r["item1_categories"] = cat
            r["item1"] = item1
            # item 2: areas
            it2 = {}
            if "C7" in F and "C2" in F:
                m7 = F["C7"][qi]["mi"].get(v, {})
                it2["B_px_C2"] = int(B2.sum())
                it2["B_px_C7"] = int(F["C7"][qi][v]["union"].sum())
                it2["M7_px"] = m7.get("M7_px")
                it2["M7_over_B"] = m7.get("M7_over_B")
                it2["c7_qualified"] = m7.get("c7_qualified")
                it2["M7_trunc_px"] = m7.get("c7_trunc_px")
                it2["B_eq_M7_plus_trunc"] = (it2["M7_px"] is not None and it2["B_px_C2"] == it2["M7_px"] + it2["M7_trunc_px"]
                                             and it2["B_px_C7"] == it2["B_px_C2"])
                it2["M7_ge_08B"] = (it2["B_px_C2"] == 0) or (it2["M7_px"] is not None and it2["M7_px"] >= 0.8 * it2["B_px_C2"])
            if "C4" in F and "C5" in F:
                a4, a5 = int(F["C4"][qi][v]["E"].sum()), int(F["C5"][qi][v]["E"].sum())
                it2.update(drawn_C4=a4, drawn_C5=a5, ratio_C5_C4=None if a4 == 0 else a5 / a4)
            r["item2"] = it2
            # item 3: outline location (actual black pixels added over C3 lie in dilate8(G) / dilate8(G_lag))
            it3 = {}
            if "C3" in F:
                c3 = F["C3"][qi][v]["final"]
                for c in ("C4", "C5"):
                    if c not in F:
                        continue
                    x = F[c][qi][v]
                    added = np.all(x["final"] == 0, axis=2) & np.any(c3 != 0, axis=2)
                    lag = x.get("mi_lag_step")
                    Gref = G if (c == "C4" or lag is None) else F["C0"][steps.index(lag)][v]["G"] if lag in steps else None
                    border = np.zeros_like(G)
                    border[0, :] = border[-1, :] = border[:, 0] = border[:, -1] = True
                    ko = args.width - args.width // 2
                    # v3 2.1 / 2.4 item 3: no line along the image edge. On the border the drawn pixels must be exactly
                    # the border-aware outline (outer ring + inner boundary where the contour crosses the edge), minus
                    # P_full; the edge-wise line of the old definition (every G pixel on the border) would fail this.
                    exp_b = None if Gref is None else (mi.outline(Gref, args.width) & ~x["P_full"] & border)
                    it3[c + "_border"] = {"drawn_on_border": int((x["E"] & border).sum()),
                                          "expected_on_border": None if exp_b is None else int(exp_b.sum()),
                                          "outer_ring_on_border": None if Gref is None else
                                          int((mr._dilate_sq(Gref, ko) & ~Gref & ~x["P_full"] & border).sum()),
                                          "G_on_border": None if Gref is None else int((Gref & border).sum()),
                                          "ok": exp_b is not None and bool(np.array_equal(x["E"] & border, exp_b))}
                    it3[c] = {"lag_step": lag, "added_px": int(added.sum()),
                              "outside_dilate8_px": None if Gref is None else
                              int((added & ~mr._dilate_sq(Gref, args.width - args.width // 2)).sum())}  # dilate8^k_out
            r["item3"] = it3
            # item 4: occlusion: occ (obj_of_interest, the hook's section 5 value) and occ_P (P):
            # occ_P = 1 - |P pixels shown| / |P_full|; shown = P_vis minus the cell's hiding region (alpha cells: P_full)
            occ = {c: F[c][qi]["row"]["views"].get(v, {}).get("occ") for c in CELLS}
            r["item4_occ"] = occ
            occP = {}
            nPf = int(Pf.sum())
            for c in CELLS:
                x = F[c][qi][v]
                if nPf == 0:
                    occP[c] = None
                    continue
                if c in ("C3", "C4", "C5"):
                    shown = x["P_full"] & ~(x["E"] & ~x["P_full"])  # outline never on P_full
                elif c == "C0":
                    shown = x["P_vis"]
                elif c == "C1":
                    shown = x["P_vis"] & ~G
                elif c == "C2":
                    shown = x["P_vis"] & ~x["union"]
                else:  # C6 (E = B \ P_vis), C7 (E = M7)
                    shown = x["P_vis"] & ~x["E"]
                occP[c] = 1.0 - float(shown.sum()) / nPf
            r["item4_occP"] = occP
            r["item4_d_occP_C7"] = None if occP.get("C7") is None or occP.get("C0") is None else occP["C7"] - occP["C0"]
            # item 5: protection truncation
            m0 = F["C0"][qi]["mi"].get(v, {})
            it5 = {"dG_px": m0.get("dG_px"), "dG_adj_Pvis_px": m0.get("dG_adj_Pvis_px"),
                   "dG_adj_frac": None if not m0.get("dG_px") else m0["dG_adj_Pvis_px"] / m0["dG_px"]}
            for c in ("C4", "C5"):
                if c in F:
                    m = F[c][qi]["mi"].get(v, {})
                    it5[c] = {"O_px": m.get("O_px"), "O_protected_px": m.get("O_protected_px"),
                              "frac": None if not m.get("O_px") else m["O_protected_px"] / m["O_px"]}
            r["item5"] = it5
            # item 6: 224-space edge along dG (C4: G(t); C5: G(t - lag)); C1 reference
            it6 = {"openpi": {}, "oft": {}}
            for prep in ("openpi", "oft"):
                if "C1" in F:
                    it6[prep]["C1"] = edge224(F["C1"][qi][v]["final"], b["bg"], G, prep)
                if "C4" in F:
                    it6[prep]["C4"] = edge224(F["C4"][qi][v]["final"], b["bg"], G, prep)
                if "C5" in F:
                    lag = F["C5"][qi][v].get("mi_lag_step")
                    Gl = F["C0"][steps.index(lag)][v]["G"] if lag in steps else G
                    it6[prep]["C5"] = edge224(F["C5"][qi][v]["final"], b["bg"], Gl, prep)
            r["item6_edge224"] = it6
            # item 7: lag displacement (both 16 and 24 for the decision rule)
            it7 = {}
            for L in (args.lag, args.lag_alt):
                if st - L >= 0 and (st - L) in steps:
                    j = steps.index(st - L)
                    it7[f"iou_{L}"] = iou(mi.outline(G, args.width), mi.outline(F["C0"][j][v]["G"], args.width))
                    it7[f"open_differs_{L}"] = bool((gq0[st][0] > OPEN_THR) != (gq0[st - L][0] > OPEN_THR))
            if v == "wrist":
                for L in (args.lag, args.lag_alt):
                    if f"iou_{L}" in it7:
                        it7[f"vis_changed_{L}"] = it7[f"iou_{L}"] is not None and it7[f"iou_{L}"] < 0.9
            r["item7"] = it7
            # item 8: C7 placement
            if "C7" in F:
                m7 = F["C7"][qi]["mi"].get(v, {})
                r["item8"] = {"cand": m7.get("c7_cand"), "overlap": m7.get("c7_overlap"), "trunc_px": m7.get("c7_trunc_px"),
                              "B_px": m7.get("B_px")}
            # item 9: model input md5 (hook path == runner path)
            r["item9"] = {c: F[c][qi][v]["model_md5"] == F[c][qi][v]["runner_md5"] == F[c][qi][v]["indep_md5"] for c in CELLS}
            r["item9_S"] = {c: (F[c][qi].get("mi_S") or {}).get("pre_eq_post") is True and
                            (F[c][qi].get("mi_S") or {}).get("obs_eq_pre") is True for c in CELLS}
            r["item15"] = item15(F, qi, v, steps, b, G)
            # hook self checks (identity outside E vs the cell's own clean render; same sub-step)
            r["self"] = {c: {"out_E_diff_px": F[c][qi]["mi"].get(v, {}).get("out_E_diff_px"),
                             "same_substep": F[c][qi]["row"]["checks"].get(v, {}).get("same_substep"),
                             "final_eq_edited": F[c][qi]["row"]["checks"].get(v, {}).get("final_eq_edited")} for c in CELLS}
            r["det"] = {c: F[c][qi]["mi"].get(v, {}).get("det") for c in CELLS if c in DET}
            rows.append(r)
    return rows


def _proj_frac(m256, f_mask):
    """share of a 256 mask that survives into a 224 input (area-scaled)."""
    n = int(np.asarray(m256).sum())
    return None if n == 0 else float(f_mask(m256).sum()) * (256.0 / 224.0) ** 2 / n


def _cover(m256, prep):
    """224 pixels with ANY bilinear coverage by the 256 mask (> 0 instead of >= 0.5), same geometry as the prep."""
    im = Image.fromarray(np.where(np.asarray(m256, bool)[::-1, ::-1], 255, 0).astype(np.uint8), mode="L")
    im = im.resize((224, 224), Image.BILINEAR)
    if prep == "oft":  # centre crop of area 0.9 resized back (frozen maskrepl.mask_to_model geometry)
        side = 224 * np.sqrt(0.9)
        o = (224 - side) / 2.0
        im = im.resize((224, 224), Image.BILINEAR, box=(o, o, o + side, o + side))
    return np.asarray(im) > 0


def _contrast(img224, m224):
    """local contrast: median grey of the 3 px band outside the projected mask minus median grey inside it."""
    if not m224.any():
        return None
    g = np.asarray(img224, np.float64).mean(axis=2)
    ring = ndimage.binary_dilation(m224, structure=np.ones((3, 3), bool), iterations=3) & ~m224
    return None if not ring.any() else float(np.median(g[ring]) - np.median(g[m224]))


def item15(F, qi, v, steps, b, G):
    """protocol v2 2.6 item 3 (checks (i), (ii) and the reported quantities)."""
    out = {"i_input_eq_expected": {c: F[c][qi][v]["runner_md5"] == F[c][qi][v]["indep_md5"] for c in CELLS}}
    Gp = np.asarray(G)[::-1, ::-1]
    out["wrist_bottom_G"] = None if v != "wrist" else {
        "G256_last_row": int(Gp[-1].sum()), "openpi224_last_row": int(PI_MASK(G)[-1].sum()),
        "oft224_last_row": int(OFT_MASK(G)[-1].sum()),
        "G256_last8": int(Gp[-8:].sum()), "openpi224_last7": int(PI_MASK(G)[-7:].sum()), "oft224_last7": int(OFT_MASK(G)[-7:].sum())}
    out["effE"] = {c: {"E256": int(F[c][qi][v]["E"].sum()), "openpi": _proj_frac(F[c][qi][v]["E"], PI_MASK),
                       "oft": _proj_frac(F[c][qi][v]["E"], OFT_MASK)} for c in CELLS}
    out["Pvis_retained"] = {"openpi": _proj_frac(b["P_vis"], PI_MASK), "oft": _proj_frac(b["P_vis"], OFT_MASK)}
    if "C3" not in F:
        return out
    c3 = F["C3"][qi][v]["final"]
    for c in ("C4", "C5"):
        if c not in F:
            continue
        x = F[c][qi][v]
        st = steps[qi]
        want = max(st - args.lag, 0)  # independent of the hook's own report (review 11-08 item 6b)
        lag_ok = c == "C4" or x.get("mi_lag_step") == want
        Gs = G if c == "C4" else (F["C0"][steps.index(want)][v]["G"] if want in steps else None)
        if Gs is None:
            out[c] = {"error": f"lag frame {want} not among replayed steps"}
            continue
        O = indep_outline(np.asarray(Gs, bool), args.width)
        drawn = O & ~x["P_full"]
        exp256 = np.array(x["clean"] if x.get("clean") is not None else c3, copy=True)  # the cell's own I_bg render
        exp256[drawn] = 0
        d = {"expected256_eq_obs": bool(np.array_equal(exp256, x["final"])),
             "expected224_eq_input": md5(indep224(exp256)) == x["runner_md5"], "drawn_256": int(drawn.sum()),
             "own_Ibg_eq_C3": bool(x.get("clean") is not None and np.array_equal(x["clean"], c3)), "lag_ok": bool(lag_ok),
             "trunc_protected": None if not O.any() else float((O & x["P_full"]).sum() / O.sum())}
        for prep, f_img, f_mask in (("openpi", PI_IMG, PI_MASK), ("oft", OFT_IMG, OFT_MASK)):
            a224, _ = f_img(x["final"])
            b224, _ = f_img(c3)
            m224 = f_mask(drawn)
            cov = _cover(drawn, prep)  # every 224 pixel the drawn mask touches at all (review 11-08 item 6a)
            near = ndimage.binary_dilation(cov, structure=np.ones((3, 3), bool), iterations=1)
            chg = np.any(np.asarray(a224) != np.asarray(b224), axis=2)
            vis = int((chg & near).sum())
            d[prep] = {"visible_px": vis, "vanished": bool(drawn.any() and vis == 0), "proj_px": int(m224.sum()),
                       "cover_px": int(cov.sum()), "proj_empty": bool(drawn.any() and not m224.any()),
                       "contrast": _contrast(a224, m224), "in_input_frac": _proj_frac(drawn, f_mask)}
        out[c] = d
    return out


def save_sheet(t, ep, F, actions):
    from PIL import Image, ImageDraw
    os.makedirs(args.png_dir, exist_ok=True)
    steps = [f["step"] for f in F["C0"]]
    close = np.where(np.asarray(actions)[:, -1] > 0)[0]
    s1 = next((s for s in steps if len(close) and s >= close[0]), steps[len(steps) // 2])
    pick = [s for s in (s1 - args.lag, s1) if s in steps]
    rot = lambda a: np.asarray(a)[::-1, ::-1]
    rows = []
    for c in CELLS:
        tiles = []
        for s in pick:
            fr = F[c][steps.index(s)]
            for v in ("agent", "wrist"):
                tiles.append(rot(fr[v]["final"]))
        rows.append(np.concatenate(tiles, axis=1))
    tiles = []  # P list check row (protocol 2.4 item 1): P_full green, G red, R blue, non-P instances yellow over C0
    for st_ in pick:
        fr = F["C0"][steps.index(st_)]
        for v in ("agent", "wrist"):
            x = fr[v]
            im = np.asarray(x["final"], np.float64) * 0.45
            for mk, col in ((x["NP"], (255, 220, 0)), (x["P_full"], (0, 255, 0)), (x["G"], (255, 0, 0)), (x["R"], (0, 0, 255))):
                im[mk] = 0.35 * np.asarray(x["final"], np.float64)[mk] + 0.65 * np.asarray(col)
            tiles.append(rot(im))
    rows.append(np.concatenate(tiles, axis=1))
    labels = CELLS + ["P/G/R"]
    img = np.concatenate(rows, axis=0).astype(np.uint8)
    pad = 60
    canvas = Image.new("RGB", (img.shape[1] + pad, img.shape[0] + 20), (255, 255, 255))
    canvas.paste(Image.fromarray(img), (pad, 20))
    d = ImageDraw.Draw(canvas)
    for i, c in enumerate(labels):
        d.text((4, 20 + i * 256 + 120), c, fill=(0, 0, 0))
    for j, s in enumerate(pick):
        for k, v in enumerate(("agent", "wrist")):
            d.text((pad + (2 * j + k) * 256 + 4, 4), f"step {s} {v}", fill=(0, 0, 0))
    name = f"manip_{args.suite}_t{t}_e{ep}.png"
    canvas.save(os.path.join(args.png_dir, name))
    with open(os.path.join(args.png_dir, name[:-4] + ".txt"), "w") as f:
        f.write(f"rows: {labels} (last row: P_full green, G red, R blue, non-P scene instances yellow, on C0)\ncolumns: steps {pick} x agent|wrist, "
                f"policy orientation\nP names: {F['C0'][0]['P_names']}\nP-list segmentation instances: "
                f"{F['C0'][0]['inst_names']}\n")
    return name, pick


def t0_key(e, hook, init, st, dmp, tid, ep):
    """protocol 4 item 3: each cell from the same RNG state through the real set_init_state path."""
    F, Q, GQ, fix = {}, {}, {}, None
    nsteps = min(args.max_steps or 24, len(dmp["action"]))
    for c in CELLS:
        np.random.set_state(st)
        e.reset()
        rm = e.ref.sim.model
        if fix is None:
            fix = ([rm.body_id2name(b) or "" for b in range(rm.nbody)], rm.body_pos.copy(), rm.body_quat.copy())
        else:
            write_fixtures(e, fix)
        hook.set_cell(c)
        hook.det_check = False
        hook.begin_episode(tid, ep)
        hook.arm(0)
        obs = e.set_init_state(init)
        hook.finish(obs, 0)
        fr = [grab(hook, obs, 0)]
        q = [np.array(e.env.sim.data.qpos, copy=True)]
        gq = [np.array(obs["robot0_gripper_qpos"], copy=True)]
        for i in range(nsteps):
            armed = (i + 1) % args.every == 0
            if armed:
                hook.arm(i + 1)
            obs, _, done, _ = e.step(np.asarray(dmp["action"][i]).tolist())
            q.append(np.array(e.env.sim.data.qpos, copy=True))
            gq.append(np.array(obs["robot0_gripper_qpos"], copy=True))
            if armed:
                hook.finish(obs, i + 1)
                fr.append(grab(hook, obs, i + 1))
            if done:
                break
        F[c], Q[c], GQ[c] = fr, np.asarray(q), gq
        if c == "C0":
            objchk = {"gripper_instance": hook.ep["gripper_inst_name"],
                      "P_full_nonempty": [bool(fv[v]["P_full"].any()) for fv in fr for v in ("agent", "wrist")],
                      "P_name_present": {n: sum(int(fv[v].get("Pnames", {}).get(n, 0)) for fv in fr for v in ("agent", "wrist"))
                                         for n in hook.ep["protect_names"]},
                      "site_px": [r["checks"].get(v, {}).get("nongeom_px") for r in hook.ep["rows"] for v in ("agent", "wrist")],
                      "T_full_nonempty": [bool(fv[v]["T_full"].any()) for fv in fr for v in ("agent", "wrist")]}
    rows = frame_checks(tid, ep, F, GQ)
    qb = Q["C0"]
    key = {"task": tid, "ep": ep, "key_row": True, "t0_mode": True, "host": HOST, "gpu": GPU,
           "action_source_ep": ep + args.action_ep_shift, "frames": [f["step"] for f in F["C0"]],
           "physics_qpos_max_vs_C0": {c: (None if Q[c].shape != qb.shape else float(np.abs(Q[c] - qb).max())) for c in Q if c != "C0"},
           "object_checks": objchk}
    for r in rows:
        r["t0_mode"] = True
        fh.write(json.dumps(r) + "\n")
    fh.write(json.dumps(key) + "\n")
    fh.flush()
    print(json.dumps({"task": tid, "ep": ep, "t0": True, "frames": key["frames"]}), flush=True)


os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fh = open(args.out, "a")
fh.write(json.dumps({"header": True, "host": HOST, "gpu": GPU, "suite": args.suite, "cells": CELLS, "lag": args.lag,
                     "width": args.width, "protect": args.protect, "det_cells": sorted(DET), "dumps": DUMPS,
                     "maskinfo_sha256": hashlib.sha256(open(mi.__file__, "rb").read()).hexdigest(),
                     "maskrepl_sha256": hashlib.sha256(open(mr.__file__, "rb").read()).hexdigest(),
                     "maskinfo_pi_sha256": hashlib.sha256(open(mip.__file__, "rb").read()).hexdigest(),
                     "every": args.every, "lag_alt": args.lag_alt}) + "\n")
print(f"MANIP_MI host={HOST} gpu={GPU} suite={args.suite}", flush=True)
n_png = 0
for off in OFFS:
    np.random.seed(7)
    random.seed(7)
    for tid in [int(x) for x in args.tasks.split(",")]:
        task = SUITE.get_task(tid)
        inits = SUITE.get_task_init_states(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        ref.seed(0)
        e = BodyEnv(ref, bddl, "Panda", 256, 10, log=lambda m: None)
        hook = Hook(e, None, lag_steps=args.lag, outline_width=args.width, protect_set=args.protect,
                    query_every=args.every, prep="openpi", log=lambda m: None)
        hook.keep_debug = True
        perm = np.random.RandomState(5478 + tid).permutation(len(inits))
        for ep in range(off, off + args.trials):
            st = np.random.get_state()
            if ep - off >= args.only_first:
                e.reset()
                continue
            dmp = load_dump(tid, ep)
            if dmp is None:
                e.reset()
                print(json.dumps({"task": tid, "ep": ep, "missing_dump": True}), flush=True)
                continue
            e._mi_task, e._mi_ep = tid, ep
            if args.t0:
                t0_key(e, hook, inits[int(perm[ep])], st, dmp, tid, ep)
                np.random.set_state(st)
                e.reset()
                continue
            np.random.set_state(st)
            e.reset()
            if args.fix_scene:
                write_fixtures(e, dmp["fix"])
            hook.set_cell("C0")
            hook.begin_episode(tid, ep)
            hook.arm(0)
            obs0 = e.set_init_state(inits[int(perm[ep])])
            hook.finish(obs0, 0)
            real0 = {v: np.array(obs0[k]) for k, _, v in VIEWS}
            q0 = np.array(e.env.sim.data.qpos, copy=True)
            snap = snapmod.snapshot(e, obs0, [])
            snap["mi_obs_timers"] = obs_timers(e)
            F, Q, GQ = {}, {}, {}
            for c in CELLS + ["C0_again"]:
                fr, q, gq = run_cell(e, hook, snap, "C0" if c == "C0_again" else c, dmp["action"])
                Q[c], GQ[c] = q, gq
                if c != "C0_again":
                    F[c] = fr
            rows = frame_checks(tid, ep, F, GQ)
            # key-level: physics (item 10), scene, replay fidelity, restored frame 0 vs the real frame 0
            qb = Q["C0"]
            phys = {c: (None if Q[c].shape != qb.shape else float(np.abs(Q[c] - qb).max())) for c in Q if c != "C0"}
            key = {"task": tid, "ep": ep, "key_row": True, "host": HOST, "gpu": GPU, "steps_replayed": int(len(qb) - 1),
                   "n_query_frames": len(F["C0"]), "physics_qpos_max_vs_C0": phys,
                   "scene_qpos0_maxdiff": float(np.abs(q0 - dmp["qpos0"]).max()) if q0.shape == dmp["qpos0"].shape else None,
                   "frame0_restored_vs_real": {v: ndiff(F["C0"][0][v]["final"], real0[v]) for v in ("agent", "wrist")},
                   "protect_names": hook.ep["protect_names"], "robot_names": hook.ep["robot_names"],
                   "P_px_by_name": {v: {n: sum(fr[v]["Ppx"].get(n, 0) for fr in F["C0"]) for n in hook.ep["protect_names"]}
                                    for v in ("agent", "wrist")},
                   "frame0_inst_px_by_body": body_px(hook, F["C0"][0])}
            if "img" in dmp:  # replay fidelity vs the S_hook run's own query images
                fid = []
                steps = [f["step"] for f in F["C0"]]
                for j, s in enumerate(dmp["img"]["steps"]):
                    if int(s) in steps:
                        k = steps.index(int(s))
                        fid.append([int(s)] + [ndiff(F["C0"][k][v]["final"], dmp["img"][v][j]) for v in ("agent", "wrist")])
                key["replay_vs_dump_img"] = fid
            if args.png_dir and n_png < args.png_keys:
                key["sheet"] = save_sheet(tid, ep, F, dmp["action"])
                n_png += 1
            for r in rows:
                fh.write(json.dumps(r) + "\n")
            fh.write(json.dumps(key) + "\n")
            fh.flush()
            print(json.dumps({"task": tid, "ep": ep, "frames": len(F["C0"]), "phys_max": max(v or 0 for v in phys.values()),
                              "scene": key["scene_qpos0_maxdiff"]}), flush=True)
            del F, Q, GQ
            np.random.set_state(st)
            e.reset()  # leave the stream where one reset per episode leaves it
print("MANIP_MI done", flush=True)
