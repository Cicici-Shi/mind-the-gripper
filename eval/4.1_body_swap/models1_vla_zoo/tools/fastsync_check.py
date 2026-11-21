"""Bitwise check of m1_fastsync against the original TWIN_CAM_SYNC=1 path (2026-11-04). Replays the published OFT
grpkp100 actions of a key on a twin cell three times in one process: original, original again (render noise floor),
patched; compares per step physical qpos/qvel, both policy images, twin stats (IK residual, camera error)."""
import argparse, glob, os, sys, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("--task", type=int, default=0); ap.add_argument("--ep", type=int, default=0)
ap.add_argument("--steps", type=int, default=80); a = ap.parse_args()
sys.path.insert(0, os.path.expanduser(os.environ["EMBODY_DIR"])); sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/vla_zoo/common"))
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import m1_fastsync
m1_fastsync.patch()
suite = OriginalSpatialSuite(); task = suite.get_task(a.task); inits = suite.get_task_init_states(a.task)
bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
f = glob.glob(os.path.expanduser(f"$PROJ_ROOT/runs/grpkp100/t{a.task}_e{a.ep}_i*.npz"))[0]
z = np.load(f, allow_pickle=True); acts = np.asarray(z["action"])[: a.steps]; init = int(f.split("_i")[-1][:-4])
def run(fast):
    BodyEnv._twin_paint = (BodyEnv.__dict__["_twin_paint"] if fast else BodyEnv._twin_paint_orig)
    np.random.seed(7)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); ref.seed(0)
    e = BodyEnv(ref, bddl, os.environ["BODY"], 256, 10, log=lambda m: None)
    e.reset(); obs = e.set_init_state(inits[init])
    Q, I = [], []
    for act in acts:
        obs, _, done, _ = e.step(np.asarray(act).tolist())
        d = e.env.sim.data
        Q.append(np.concatenate([d.qpos, d.qvel]).copy()); I.append([obs["agentview_image"].copy(), obs["robot0_eye_in_hand_image"].copy()])
        if done: break
    ts = {k: list(map(float, e.twin_stats.get(k, []))) for k in ("pos_err_mm", "rot_err_deg", "cam_err_mm", "cam_err_deg")}
    e.close()
    return np.array(Q), I, ts
fast_paint = BodyEnv._twin_paint
BodyEnv.__dict__  # noqa
import time
t = time.time(); Qo, Io, To = run(False); t1 = time.time() - t
t = time.time(); Qo2, Io2, To2 = run(False); t2 = time.time() - t
BodyEnv._twin_paint_fast = fast_paint
def run_fast():
    BodyEnv._twin_paint = fast_paint
    return run.__wrapped__() if hasattr(run, "__wrapped__") else None
BodyEnv._twin_paint = fast_paint
t = time.time()
# run with fast paint
np.random.seed(7)
ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256); ref.seed(0)
e = BodyEnv(ref, bddl, os.environ["BODY"], 256, 10, log=lambda m: None)
e.reset(); obs = e.set_init_state(inits[init]); Qf, If = [], []
for act in acts:
    obs, _, done, _ = e.step(np.asarray(act).tolist()); d = e.env.sim.data
    Qf.append(np.concatenate([d.qpos, d.qvel]).copy()); If.append([obs["agentview_image"].copy(), obs["robot0_eye_in_hand_image"].copy()])
    if done: break
Tf = {k: list(map(float, e.twin_stats.get(k, []))) for k in ("pos_err_mm", "rot_err_deg", "cam_err_mm", "cam_err_deg")}
t3 = time.time() - t; Qf = np.array(Qf)
def imgdiff(A, B): return sum(int(not np.array_equal(x[c], y[c])) for x, y in zip(A, B) for c in (0, 1)), max(int(np.abs(x[c].astype(int) - y[c].astype(int)).max()) for x, y in zip(A, B) for c in (0, 1))
out = {"key": [a.task, a.ep, init], "cell_env": {k: os.environ.get(k) for k in ("BODY", "TWIN_CAMS", "TWIN_CAM", "TWIN_CAM_SYNC")},
       "steps": [len(Qo), len(Qo2), len(Qf)], "wall_s": [round(t1, 1), round(t2, 1), round(t3, 1)],
       "qpos_qvel_maxdiff_orig_vs_orig": float(np.abs(Qo - Qo2).max()), "qpos_qvel_maxdiff_orig_vs_fast": float(np.abs(Qo - Qf).max()) if Qo.shape == Qf.shape else "shape",
       "img_frames_diff_orig_vs_orig(n,maxpix)": imgdiff(Io, Io2), "img_frames_diff_orig_vs_fast(n,maxpix)": imgdiff(Io, If),
       "twin_stats_identical_orig_vs_fast": To == Tf, "twin_stats_identical_orig_vs_orig": To == To2}
print("FASTSYNC_CHECK " + json.dumps(out), flush=True)
