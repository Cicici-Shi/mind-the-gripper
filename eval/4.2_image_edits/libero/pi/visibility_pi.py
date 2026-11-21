"""maskinfo-pi visibility / effective-edit checks on cached states (astra_three_protocols_rulings_2026-11-08 §1,
items 2 and 7). GPU (EGL), conda env oftpi, no policy, outcomes never read.

State: as tools/edit_equiv_pi.py (one BodyEnv per key, set_init_state, toy2 snapshot, every cell restarts from it and
replays the first --steps actions of the maskrepl S_hook dump of the key, hook armed every 5 steps; production hook
embodied.maskinfo_pi.MaskInfoPiHook with the production parameters: lag 15, width 3, protect interactive, openpi prep).

Per query frame and view:
  * C4 / C5 (the PROTECTED outline is the expected edit, not the full O): an EXPECTED composite is generated
    independently of the hook code -- outline by scipy.ndimage (inner = G minus its 4-neighbour erosion with the
    frame outside treated as G; outer = 3x3 dilation twice minus G), minus P_full, painted black on the cell's own
    clean render (I_bg) -- then sent through an independent PIL bilinear 256 -> 224 after the 180 deg rotation.
    Checks (hard): returned 256 observation == expected composite; 224 input actually produced by the client path
    (openpi_client resize_with_pad) == expected 224; protected outline not completely vanished in the final 224 input
    (some changed pixel vs. the same frame without outline, near the resized drawn mask). Reported (no thresholds):
    drawn / visible pixels, local contrast (mean |delta| on the resized drawn mask), protected truncation
    |O n P_full| / |O|, edge on dG (maskrepl section 5 definition) for C1 / C4 / C5 -> ratios C4/C1, C5/C1, both for
    the openpi input and for OFT's input (JPEG + lanczos + crop 0.9), so the two native preprocessings are reported
    side by side.
  * every cell: effective edit region after each native preprocessing: |E| at 256, |mask_to_model(E)| on the openpi
    224 input and on the OFT 224 input (rotate, bilinear, centre crop 0.9).
  * wrist: gripper G retained at the policy-bottom edge: G pixels in the last --band rows of the policy-oriented 256
    image and in the last round(band * 224 / 256) rows of each 224 input.
Usage: python tools/visibility_pi.py --out FILE.jsonl [--keys 0:40,1:40] [--steps 60]; --summary FILE.jsonl
"""
import argparse, glob, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="")
ap.add_argument("--summary", default="")
ap.add_argument("--keys", default="0:40,1:40")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--band", type=int, default=8)
ap.add_argument("--lag", type=int, default=15)
ap.add_argument("--width", type=int, default=3)
ap.add_argument("--dumps", default=os.path.expanduser("$PROJ_ROOT/maskrepl/out/runs/S_hook"))
args = ap.parse_args()


def q(a):
    a = [x for x in a if x is not None]
    return None if not a else {"n": len(a), "median": float(np.median(a)), "p10": float(np.percentile(a, 10)),
                               "p90": float(np.percentile(a, 90)), "min": float(np.min(a))}


if args.summary:
    R = [json.loads(l) for l in open(args.summary)]
    out = {}
    for v in ("agent", "wrist"):
        for c in ("C4", "C5"):
            rr = [r for r in R if r["cell"] == c and r["view"] == v]
            out[f"{c}_{v}"] = {"frames": len(rr), "obs_eq_expected_256": all(r["eq256"] for r in rr),
                               "input_eq_expected_224": all(r["eq224"] for r in rr),
                               "vanished_frames": sum(1 for r in rr if r["vanished"]),
                               "drawn_px": q([r["drawn_px"] for r in rr]), "visible_px_224": q([r["visible_px"] for r in rr]),
                               "contrast": q([r["contrast"] for r in rr]), "trunc": q([r["trunc"] for r in rr])}
            for prep in ("openpi", "oft"):
                ratios = []
                for r in rr:
                    c1 = [x for x in R if x["cell"] == "C1" and x["view"] == v and x["key"] == r["key"] and x["i"] == r["i"]]
                    if c1 and c1[0]["edge"][prep] and r["edge"][prep] is not None:
                        ratios.append(r["edge"][prep] / c1[0]["edge"][prep])
                out[f"{c}_{v}"][f"edge_ratio_vs_C1_{prep}"] = q(ratios)
        for c in sorted({r["cell"] for r in R}):
            rr = [r for r in R if r["cell"] == c and r["view"] == v]
            out[f"effE_{c}_{v}"] = {k: q([r["effE"][k] for r in rr]) for k in ("E256", "openpi224", "oft224")}
        if v == "wrist":
            rr = [r for r in R if r["cell"] == "C0" and r["view"] == v]
            out["wrist_bottom_G"] = {k: q([r["Gbot"][k] for r in rr]) for k in ("G256", "openpi224", "oft224")}
    c1 = {(r["key"], r["i"], r["view"]): r["Emd5"] for r in R if r["cell"] == "C1"}
    c8 = [(r["key"], r["i"], r["view"], r["Emd5"]) for r in R if r["cell"] == "C8"]
    out["C8_mask_eq_C1_mask"] = {"frames": len(c8), "all_equal": bool(c8) and all(c1.get(k[:3]) == k[3] for k in c8)}
    print(json.dumps(out, indent=1))
    hard = all(out[f"{c}_{v}"]["obs_eq_expected_256"] and out[f"{c}_{v}"]["input_eq_expected_224"]
               and out[f"{c}_{v}"]["vanished_frames"] == 0 for c in ("C4", "C5") for v in ("agent", "wrist")) \
        and out["C8_mask_eq_C1_mask"]["all_equal"]
    print("VISIBILITY_HARD_CHECKS_OK", hard)
    sys.exit(0 if hard else 1)

from PIL import Image  # noqa: E402
from scipy import ndimage  # noqa: E402
MI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MI)
for k in ("BODY", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "HIDE_GRIPPER", "TOY_PIXSTATS", "MASK_CELL", "MI_CELL"):
    os.environ.pop(k, None)
os.environ["COPY_GRIP_CMD"] = "1"
from embody import maskinfo_pi as mip  # noqa: E402
from embody import maskrepl as mr  # noqa: E402
from embody import snapshot as snapmod  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

OFT_IMG, OFT_MASK = mip.PREP["oft"]  # frozen OFT functions (captured before the openpi swap)
PI_IMG, PI_MASK = mip.PREP["openpi"]
CELLS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
VIEWS = (("agentview_image", "agentview", "agent"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand", "wrist"))
OBS_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value")
ST = np.ones((3, 3), bool)
CROSS = ndimage.generate_binary_structure(2, 1)


class Hook(mip.MaskInfoPiHook):
    def finish(self, obs, step, model_md5=None):
        self._last_cap = self.cap or {}
        return super().finish(obs, step, model_md5)


def indep_outline(G):
    inner = G & ~ndimage.binary_erosion(G, structure=CROSS, border_value=1)
    outer = ndimage.binary_dilation(G, structure=ST, iterations=2) & ~G
    return inner | outer


def pil224(img):
    return np.asarray(Image.fromarray(np.ascontiguousarray(np.asarray(img)[::-1, ::-1])).resize((224, 224), Image.BILINEAR))


def edge(img224, bg224, G224):
    z = np.zeros_like(G224)
    return mr.section5(img224, bg224, z, G224, z, z)["edge"]


def obs_timers(e):
    return {k: {f: np.array(getattr(o, f), copy=True) if isinstance(getattr(o, f), np.ndarray) else getattr(o, f)
                for f in OBS_FIELDS} for k, o in e.env.env._observables.items()}


def set_obs_timers(e, st):
    for k, o in e.env.env._observables.items():
        for f, v in st[k].items():
            setattr(o, f, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)


SUITE = OriginalSpatialSuite()
fh = open(args.out, "w")
kb = int(round(args.band * 224 / 256))
for tid, ep in [tuple(int(x) for x in k.split(":")) for k in args.keys.split(",")]:
    with np.load(glob.glob(os.path.join(args.dumps, f"t{tid}_e{ep}_i*.npz"))[0], allow_pickle=True) as z:
        acts = np.asarray(z["action"])[: args.steps]  # success never read
    np.random.seed(7); random.seed(7)
    task = SUITE.get_task(tid)
    inits = SUITE.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    e = BodyEnv(ref, bddl, "Panda", 256, 10, log=lambda m: None)
    perm = np.random.RandomState(5478 + tid).permutation(len(inits))
    e.reset()
    rs = e.env.env
    pristine = (rs._update_observables, {k: rs._observables[k]._sensor for k, _, _ in VIEWS})

    def unhook():
        rs._update_observables = pristine[0]
        for k, s_ in pristine[1].items():
            rs._observables[k]._sensor = s_

    mk = lambda c: Hook(e, c, lag_steps=args.lag, outline_width=args.width, protect_set="interactive", query_every=5,
                        prep="openpi", log=lambda m: None)
    h0 = mk("C0"); h0.begin_episode(tid, ep); h0.arm(0)
    obs0 = e.set_init_state(inits[int(perm[ep])]); h0.finish(obs0, 0)
    snap = snapmod.snapshot(e, obs0, []); timers = obs_timers(e)
    unhook()
    for cell in CELLS:
        snapmod.restore(e, snap); set_obs_timers(e, timers)
        h = mk(cell); h.begin_episode(tid, ep); h.arm(0)
        frames = []
        obs = e._fix(e.env.env._get_observations(force_update=True)); h.finish(obs, 0)
        frames.append((0, obs, h._last_cap)); set_obs_timers(e, timers)
        for i in range(len(acts)):
            armed = (i + 1) % 5 == 0
            if armed:
                h.arm(i + 1)
            obs, _, done, _ = e.step(np.asarray(acts[i]).tolist())
            if armed:
                h.finish(obs, i + 1); frames.append((i + 1, obs, h._last_cap))
            if done:
                break
        Ghist = {v: {} for _, _, v in VIEWS}
        for j, (st, ob, cap) in enumerate(frames):
            for key, cam, v in VIEWS:
                c = cap[cam]
                final = np.asarray(ob[key])
                G = c["G"]
                Ghist[v][st] = G.copy()
                E = np.unpackbits(h.ep["frames"][j][v]["M"])[: 256 * 256].reshape(256, 256).astype(bool)
                row = {"key": f"{tid}:{ep}", "cell": cell, "i": j, "step": st, "view": v, "Emd5": mr._md5(E),
                       "effE": {"E256": int(E.sum()), "openpi224": int(PI_MASK(E).sum()), "oft224": int(OFT_MASK(E).sum())}}
                Gp = G[::-1, ::-1]
                row["Gbot"] = {"G256": int(Gp[-args.band:].sum()), "openpi224": int(PI_MASK(G)[-kb:].sum()),
                               "oft224": int(OFT_MASK(G)[-kb:].sum())}
                pi_in, _ = PI_IMG(final)
                oft_in, _ = OFT_IMG(final)
                if cell in ("C1", "C4", "C5"):
                    bg_pi, _ = PI_IMG(c["bg"]); bg_oft, _ = OFT_IMG(c["bg"])
                    row["edge"] = {"openpi": edge(pi_in, bg_pi, PI_MASK(G)), "oft": edge(oft_in, bg_oft, OFT_MASK(G))}
                if cell in ("C4", "C5"):
                    Gs = G if cell == "C4" else Ghist[v][max(st - args.lag, 0)]
                    O = indep_outline(Gs)
                    Pf = np.isin(c["bg_gid"], h.ep["P_geoms"])
                    drawn = O & ~Pf
                    exp256 = np.array(c["clean"], copy=True); exp256[drawn] = 0
                    exp224 = pil224(exp256)
                    base224 = pil224(c["clean"])
                    near = ndimage.binary_dilation(PI_MASK(drawn), structure=ST, iterations=1)
                    chg = np.any(pi_in != base224, axis=2)
                    vis = int((chg & near).sum())
                    m224 = PI_MASK(drawn)
                    row.update(eq256=bool(np.array_equal(exp256, final)), eq224=bool(np.array_equal(exp224, pi_in)),
                               drawn_px=int(drawn.sum()), visible_px=vis, vanished=bool(drawn.any() and vis == 0),
                               contrast=None if not m224.any() else float(np.abs(pi_in.astype(np.int16) - base224.astype(np.int16)).mean(axis=2)[m224].mean()),
                               trunc=None if not O.any() else float((O & Pf).sum() / O.sum()),
                               lag_step=None if cell == "C4" else max(st - args.lag, 0))
                fh.write(json.dumps(row) + "\n")
        unhook()
    fh.flush()
    print(f"VIS_KEY t{tid} e{ep} done", flush=True)
print("VIS_DONE", flush=True)
