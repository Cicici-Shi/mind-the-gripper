"""maskinfo-pi: the 256 edited observation of the openpi hook == the frozen OFT hook's, per cell, on the SAME cached
state (Astra ruling 14: only same-state 256 edits are claimed byte-identical). GPU (EGL), no policy, outcomes
never read. Two runs of this script, then --compare:
  --mode frozen  (conda env oft, PYTHONPATH=$PROJ_ROOT/openvla-oft): embody.maskinfo.MaskInfoHook (lag 16) for
                 C0..C7 and embody.maskrepl.MaskHook cell R1 for C8, OFT model-input path (as in the OFT runs)
  --mode pi      (conda env oftpi): embody.maskinfo_pi.MaskInfoPiHook, openpi model-input path; C5 with lag 10
State: one BodyEnv per key, set_init_state under the armed hook, a toy2 snapshot (embody/snapshot.py, as
tools/manip_maskinfo.py); every cell restarts from the snapshot (+ observable timers) and replays the same actions
(the first --steps actions of the maskrepl S_hook dump of the key: a block already read; success never loaded)
with the hook armed every 5 steps. Cadence 5 with the frozen lag 16 takes index q - 2 = step t - 10, which is what
the pi hook takes with lag 10, so C5 compares the same drawing code on the same lag frame (the 15-step lag itself
is asserted in the smoke run). Output jsonl: md5 of each view's returned observation, of the edit mask and qpos.
"""
import argparse, glob, hashlib, json, os, random, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--mode", choices=["frozen", "pi", "compare"], required=True)
ap.add_argument("--out", default="")
ap.add_argument("--a", default="")
ap.add_argument("--b", default="")
ap.add_argument("--keys", default="0:40,5:40")
ap.add_argument("--steps", type=int, default=60)
ap.add_argument("--dumps", default=os.path.expanduser("$PROJ_ROOT/maskrepl/out/runs/S_hook"))
args = ap.parse_args()
md5 = lambda x: hashlib.md5(np.ascontiguousarray(x).tobytes()).hexdigest()

if args.mode == "compare":
    A = {(r["task"], r["ep"], r["cell"], r["i"]): r for r in map(json.loads, open(args.a)) if "cell" in r}
    B = {(r["task"], r["ep"], r["cell"], r["i"]): r for r in map(json.loads, open(args.b)) if "cell" in r}
    cells = sorted({k[2] for k in A} | {k[2] for k in B})
    allok = set(A) == set(B)
    for c in cells:
        ks = [k for k in A if k[2] == c and k in B]
        img = all(A[k]["img"] == B[k]["img"] for k in ks)
        if not img and os.path.exists(args.a + ".npz") and os.path.exists(args.b + ".npz"):
            za, zb = np.load(args.a + ".npz"), np.load(args.b + ".npz")
            st = []
            for k in ks:
                for v in ("agent", "wrist"):
                    n = f"{k[0]}_{k[1]}_{k[2]}_{k[3]}_{v}"
                    d = np.abs(za[n].astype(np.int16) - zb[n].astype(np.int16)).max(axis=2)
                    st.append((int((d > 0).sum()), int(d.max())))
            print(f"  {c}: frames_with_diff={sum(1 for x in st if x[0])}/{len(st)} max_px={max(x[0] for x in st)} max_absdiff={max(x[1] for x in st)}")
        msk = all(A[k]["E"] == B[k]["E"] for k in ks)
        qp = all(A[k]["qpos"] == B[k]["qpos"] for k in ks)
        allok &= img and msk and qp and bool(ks)
        print(f"EDIT_EQUIV {c}: frames={len(ks)} obs_bytes_equal={img} edit_mask_equal={msk} qpos_equal={qp}")
    print("EDIT_EQUIV_ALL", allok)
    sys.exit(0 if allok else 1)

MI = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, MI)
for k in ("BODY", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "HIDE_GRIPPER", "TOY_PIXSTATS", "MASK_CELL", "MI_CELL"):
    os.environ.pop(k, None)
os.environ["COPY_GRIP_CMD"] = "1"
from embody import maskinfo as mi  # noqa: E402
from embody import maskrepl as mr  # noqa: E402
from embody import snapshot as snapmod  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
if args.mode == "pi":
    from embody import maskinfo_pi as mip  # noqa: E402

CELLS = ["C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"]
VIEWS = (("agentview_image", "agent"), ("robot0_eye_in_hand_image", "wrist"))
OBS_FIELDS = ("_time_since_last_sample", "_sampled", "_current_delay", "_current_observed_value")


def obs_timers(e):
    return {k: {f: np.array(getattr(o, f), copy=True) if isinstance(getattr(o, f), np.ndarray) else getattr(o, f)
                for f in OBS_FIELDS} for k, o in e.env.env._observables.items()}


def set_obs_timers(e, st):
    for k, o in e.env.env._observables.items():
        for f, v in st[k].items():
            setattr(o, f, np.array(v, copy=True) if isinstance(v, np.ndarray) else v)


def make_hook(e, cell):
    if args.mode == "pi":
        lag = 10 if cell == "C5" else 15
        return mip.MaskInfoPiHook(e, cell, lag_steps=lag, outline_width=3, protect_set="interactive", query_every=5,
                                  prep="openpi", log=lambda m: None)
    if cell == "C8":
        return mr.MaskHook(e, "R1", log=lambda m: None)
    return mi.MaskInfoHook(e, cell, lag_steps=16, outline_width=3, protect_set="interactive", log=lambda m: None)


def arm(h, step):
    h.arm(step) if args.mode == "pi" else h.arm()


def E_of(h, step_idx):
    fr = h.ep["frames"][step_idx]
    return {v: md5(fr[v]["M"]) for _, v in VIEWS if v in fr}


SUITE = OriginalSpatialSuite()
fh = open(args.out, "w")
IMGS = {}
keys = [tuple(int(x) for x in k.split(":")) for k in args.keys.split(",")]
for tid, ep in keys:
    f = glob.glob(os.path.join(args.dumps, f"t{tid}_e{ep}_i*.npz"))
    assert f, f"no dump for t{tid} e{ep}"
    with np.load(f[0], allow_pickle=True) as z:
        acts = np.asarray(z["action"])[: args.steps]  # success never read
    np.random.seed(7)
    random.seed(7)
    task = SUITE.get_task(tid)
    inits = SUITE.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    e = BodyEnv(ref, bddl, "Panda", 256, 10, log=lambda m: None)
    perm = np.random.RandomState(5478 + tid).permutation(len(inits))
    e.reset()  # scene fidelity to the dump is not needed: both modes build the identical scene this way
    rs = e.env.env
    pristine = (rs._update_observables, {k: rs._observables[k]._sensor for k, _ in VIEWS})

    def unhook():
        rs._update_observables = pristine[0]
        for k, s_ in pristine[1].items():
            rs._observables[k]._sensor = s_
    h0 = make_hook(e, "C0")
    h0.begin_episode(tid, ep)
    arm(h0, 0)
    obs0 = e.set_init_state(inits[int(perm[ep])])
    h0.finish(obs0, 0)
    snap = snapmod.snapshot(e, obs0, [])
    timers = obs_timers(e)
    unhook()  # each cell installs a fresh hook on the pristine env below
    for cell in CELLS:
        snapmod.restore(e, snap)
        set_obs_timers(e, timers)
        h = make_hook(e, cell)
        h.begin_episode(tid, ep)
        arm(h, 0)
        obs = e._fix(e.env.env._get_observations(force_update=True))
        h.finish(obs, 0)
        set_obs_timers(e, timers)
        rows = [(0, obs, np.array(e.env.sim.data.qpos, copy=True))]
        for i in range(len(acts)):
            armed = (i + 1) % 5 == 0
            if armed:
                arm(h, i + 1)
            obs, _, done, _ = e.step(np.asarray(acts[i]).tolist())
            if armed:
                h.finish(obs, i + 1)
                rows.append((i + 1, obs, np.array(e.env.sim.data.qpos, copy=True)))
            if done:
                break
        for j, (st, ob, q) in enumerate(rows):
            Em = E_of(h, j)
            for k_, v_ in VIEWS:
                IMGS[f"{tid}_{ep}_{cell}_{j}_{v_}"] = np.asarray(ob[k_])
            fh.write(json.dumps({"task": tid, "ep": ep, "cell": cell, "i": j, "step": st,
                                 "img": {v: md5(np.asarray(ob[k])) for k, v in VIEWS}, "E": Em, "qpos": md5(q)}) + "\n")
        unhook()
    fh.flush()
    print(f"EDIT_EQUIV_KEY t{tid} e{ep} mode={args.mode} cells={len(CELLS)}", flush=True)
    e.close() if hasattr(e, "close") else None
np.savez_compressed(args.out + ".npz", **IMGS)
print("EDIT_EQUIV_DONE", args.mode, flush=True)
