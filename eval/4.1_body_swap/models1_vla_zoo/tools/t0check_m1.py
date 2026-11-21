"""models1 step-0 consistency (protocol_models1.md 7.2), 2026-11-04. Adapted from $PROJ_ROOT/embody_toy/tools/t0check_arm.py.
All models1 BodyEnv cells are built in ONE process (same node / GPU); every key gets the same initial state (global
np.random stream replayed as run_libero_eval / zoo_client do per shard: seed 7, tasks in order, TRIALS episodes each).
Per key, per camera, mean and max |pixel diff| of the raw 256x256 uint8 policy images (before any model preprocessing):
  T0 vs S (gate: per-cell average < 0.11 per camera);
  GTfc / GTw / GTa / G_T vs G: the camera that is NOT replaced must be identical (diff 0); replaced camera: agent mean
  < 1.0, wrist mean < 8 (G_T design cost 5.6);
  twin vs physical wrist camera at the first paint (twin_stats cam_err), IK residual at the first paint.
No policy, no outcome. EMBODY_DIR must be $PROJ_ROOT/embody_toy.
Usage: python tools/t0check_m1.py --suite libero_spatial --offset 0 --trials 5 --out FILE
"""
import argparse, json, os, random, socket, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--offset", type=int, default=0)
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=5)
args = ap.parse_args()
sys.path.insert(0, os.path.expanduser(os.environ["EMBODY_DIR"]))
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

CELLS_SH = os.path.expanduser("$PROJ_ROOT/vla_zoo/tools/zoo_cells.sh")
NAMES = ["sham_cg", "sham_twin", "grpkp100", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"]
CELLS = {}
for n in NAMES:
    out = subprocess.run([CELLS_SH, "vars", n], capture_output=True, text=True, check=True).stdout.split()
    CELLS[n] = dict(BODY=out[0], **dict(kv.split("=", 1) for kv in out[1:]))
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM", "TWIN_CAMS",
        "TWIN_CAM_SYNC", "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM"}
CAMS = ("agentview_image", "robot0_eye_in_hand_image")
try:
    GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=30).stdout.strip().splitlines()[0]
except Exception as e:
    GPU = f"unknown ({e!r})"
HOST = socket.gethostname()


def set_vars(v):
    for k in ALLV:
        os.environ.pop(k, None)
    os.environ.update(v)


def diff(a, b):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return {"mean": float(d.mean()), "max": int(d.max())}


def grip_geoms(m):
    return [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")]


suite = OriginalObjectSuite() if args.suite == "libero_object" else OriginalSpatialSuite()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fh = open(args.out, "a")
print(f"T0CHECK_M1 host={HOST} gpu={GPU} suite={args.suite} cells={CELLS}", flush=True)
np.random.seed(7)
random.seed(7)
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
    perm = np.random.RandomState(5478 + tid).permutation(len(inits))
    for ep in range(args.offset, args.offset + args.trials):
        init_id = int(perm[ep])
        st = np.random.get_state()
        obs, tw = {}, {}
        for name, v in CELLS.items():
            np.random.set_state(st)
            set_vars(v)
            e = envs[name]
            e.reset()
            obs[name] = e.set_init_state(inits[init_id])
            ts = getattr(e, "twin_stats", None)
            if ts is not None:
                tw[name] = {"cam_err_mm0": (ts.get("cam_err_mm") or [None])[0], "cam_err_deg0": (ts.get("cam_err_deg") or [None])[0],
                            "pos_err_mm0": (ts.get("pos_err_mm") or [None])[0], "rot_err_deg0": (ts.get("rot_err_deg") or [None])[0]}
        rec = {"suite": args.suite, "task": tid, "episode": ep, "init": init_id, "host": HOST, "gpu": GPU, "twin": tw,
               "t0_vs_s": {c: diff(obs["sham_twin"][c], obs["sham_cg"][c]) for c in CAMS}}
        g = envs["grpkp100"]
        for n in ("g_tfc", "g_tw_fc", "g_ta_fc", "g_t"):
            rec[f"{n}_vs_g"] = {c: diff(obs[n][c], obs["grpkp100"][c]) for c in CAMS}
            # gate metric (toy1 8.2 / t0check_toy.py gtfc_vs_g_both_hidden): replaced camera, twin render vs G render,
            # gripper geoms of BOTH alpha 0, so only non-gripper pixels (arm pose of the IK twin, camera pose) count
            e = envs[n]
            rec[f"{n}_vs_g_both_hidden"] = {}
            for c, cam in zip(CAMS, ("agentview", "robot0_eye_in_hand")):
                if e.twin_cams not in ("both", "wrist" if cam != "agentview" else "agent"):
                    continue
                a_ = e._render(e.twin.sim, cam, grip_geoms(e.twin.sim.model), depth=False, how="alpha")[0]
                b_ = g._render(g.env.sim, cam, grip_geoms(g.env.sim.model), depth=False, how="alpha")[0]
                rec[f"{n}_vs_g_both_hidden"][c] = diff(a_, b_)
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        print(json.dumps(rec), flush=True)
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
print("T0CHECK_M1 done", flush=True)
