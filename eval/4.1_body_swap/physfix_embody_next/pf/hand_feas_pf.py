"""physfix (Astra ruling 4, 2026-11-06) FEASIBILITY of the new cell "same Robotiq physics, physical arm posture kept,
only the gripper repainted as the Panda gripper" (local repaint). Uses the existing embody_next TOY_PARTS switches
(toy1 c5 / c6 part compositing, depth-tested, WORKLOG.md ch.10): TOY_PARTS=hand TOY_PHYS_GRIPPER=hide_same, twin pose
ik (Panda pads on the Robotiq pads = GTfc' target) or flange (Panda hand on the physical flange). No policy, no outcome.
Cells (stabilised config, GRP_MAP=kp100): Gp = G', Tfc = GTfc', Hik = hand repaint (pose ik), Hfl = hand repaint (pose
flange); REF = the Panda reference env (source-domain image).
Per key at the first frame: gap between the physical flange and the composited Panda hand root (mm, along the hand z);
agent / wrist mean |diff| between cells; "gap pixels": agent pixels that are robot (arm) in Tfc's twin and are scene in
Hik (where the retracted physical arm leaves background) and the converse; physical Robotiq pixels left in Hik (must be
0); twin-vs-physical wrist camera error; per-step wall time over 20 zero-arm steps. PNG strips for --png-keys.
Usage: python hand_feas_pf.py --suite libero_object --out FILE.jsonl --png DIR
"""
import argparse, json, os, random, socket, subprocess, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=1)
ap.add_argument("--png", default="")
ap.add_argument("--png-keys", default="0:0,5:0")
ap.add_argument("--time-steps", type=int, default=20)
args = ap.parse_args()
sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_next")))
from embody.body_env import BodyEnv  # noqa
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs import OffScreenRenderEnv  # noqa

G = dict(BODY="PandaR85", GRP_KP="100", GRP_MAP="kp100", R85_ARMATURE="1e-3", COPY_GRIP_CMD="1")
TW = dict(TWIN_RENDER="1", TWIN_BODY="Panda", TWIN_CAMS="both", TWIN_CAM="phys", TWIN_CAM_SYNC="1")
CELLS = {"Gp": dict(G), "Tfc": dict(G, **TW),
         "Hik": dict(G, **TW, TOY_PARTS="hand", TOY_PHYS_GRIPPER="hide_same", TOY_TWIN_POSE="ik", TOY_PIXSTATS="1"),
         "Hfl": dict(G, **TW, TOY_PARTS="hand", TOY_PHYS_GRIPPER="hide_same", TOY_TWIN_POSE="flange", TOY_PIXSTATS="1")}
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "R85_ARMATURE", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM",
        "TWIN_CAMS", "TWIN_CAM_SYNC", "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_PHYS_GRIPPER", "TOY_TWIN_POSE",
        "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM"}
A, W = "agentview_image", "robot0_eye_in_hand_image"
GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
HOST = socket.gethostname()


def set_vars(v):
    for k in ALLV:
        os.environ.pop(k, None)
    os.environ.update(v)


def md(a, b):
    return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean())


def robot_mask(be, sim, cam):
    gid = be._render(sim, cam, [], depth=False, seg=True)[2]
    m = sim.model
    ids = [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(("robot0", "gripper0", "mount0"))]
    return np.isin(gid, ids)


suite = OriginalObjectSuite() if args.suite == "libero_object" else OriginalSpatialSuite()
fh = open(args.out, "a")
PK = {tuple(int(x) for x in k.split(":")) for k in args.png_keys.split(",") if k}
print(f"HAND_FEAS host={HOST} gpu={GPU} suite={args.suite}", flush=True)
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
    for ep in range(args.trials):
        init_id = int(perm[ep])
        st = np.random.get_state()
        obs = {}
        for name, v in CELLS.items():
            np.random.set_state(st)
            set_vars(v)
            envs[name].reset()
            obs[name] = envs[name].set_init_state(inits[init_id])
        g, t, h, hf = envs["Gp"], envs["Tfc"], envs["Hik"], envs["Hfl"]
        set_vars(CELLS["Hik"])
        ref_img = {c: h.ref.sim.render(width=256, height=256, camera_name=cam) for c, cam in ((A, "agentview"), (W, "robot0_eye_in_hand"))}
        # gap: physical flange (eef body of the Panda arm) vs twin Panda hand root, in the physical hand frame
        hb_p = h.env.sim.model.body_name2id(h.env.robots[0].robot_model.eef_name)
        hb_t = h.twin.sim.model.body_name2id(h.twin.robots[0].robot_model.eef_name)
        Rp = h.env.sim.data.xmat[hb_p].reshape(3, 3)
        dv = Rp.T @ (h.twin.sim.data.xpos[hb_t] - h.env.sim.data.xpos[hb_p]) * 1000
        dvf = Rp.T @ (hf.twin.sim.data.xpos[hf.twin.sim.model.body_name2id(hf.twin.robots[0].robot_model.eef_name)] - h.env.sim.data.xpos[hb_p]) * 1000
        rm_t = robot_mask(t, t.twin.sim, "agentview")       # Tfc twin arm (what GTfc' shows)
        rm_p = robot_mask(h, h.env.sim, "agentview")        # physical arm (+ hidden Robotiq counted, drawn normally here)
        hp = h.toy_stats["px"][-1] if h.toy_stats and h.toy_stats.get("px") else None
        rec = {"suite": args.suite, "task": tid, "episode": ep, "init": init_id, "host": HOST, "gpu": GPU,
               "gap_hand_frame_mm_ik": dv.round(2).tolist(), "gap_norm_mm_ik": float(np.linalg.norm(dv)),
               "gap_hand_frame_mm_flange": dvf.round(2).tolist(),
               "agent": {"Hik_vs_Tfc": md(obs["Hik"][A], obs["Tfc"][A]), "Hik_vs_Gp": md(obs["Hik"][A], obs["Gp"][A]),
                         "Tfc_vs_Gp": md(obs["Tfc"][A], obs["Gp"][A]), "Hfl_vs_Tfc": md(obs["Hfl"][A], obs["Tfc"][A]),
                         "Hik_vs_REF": md(obs["Hik"][A], ref_img[A]), "Tfc_vs_REF": md(obs["Tfc"][A], ref_img[A]),
                         "Gp_vs_REF": md(obs["Gp"][A], ref_img[A]), "Hfl_vs_REF": md(obs["Hfl"][A], ref_img[A])},
               "wrist": {"Hik_vs_Tfc": md(obs["Hik"][W], obs["Tfc"][W]), "Hfl_vs_Tfc": md(obs["Hfl"][W], obs["Tfc"][W]),
                         "Hik_vs_Gp": md(obs["Hik"][W], obs["Gp"][W]), "Tfc_vs_REF": md(obs["Tfc"][W], ref_img[W]),
                         "Hik_vs_REF": md(obs["Hik"][W], ref_img[W])},
               "agent_px": {"twin_arm_only_Tfc": int((rm_t & ~rm_p).sum()), "phys_arm_only": int((rm_p & ~rm_t).sum()),
                            "either": int((rm_t | rm_p).sum())},
               "toy_px_Hik": hp, "cam_err_mm0": {n: (envs[n].twin_stats.get("cam_err_mm") or [None])[0] for n in ("Tfc", "Hik", "Hfl")},
               "ik_err_mm0": {n: (envs[n].twin_stats.get("pos_err_mm") or [None])[0] for n in ("Tfc", "Hik", "Hfl")}}
        if (tid, ep) in PK and args.png:
            import imageio
            os.makedirs(args.png, exist_ok=True)
            for c in (A, W):
                strip = np.concatenate([np.asarray(x)[::-1, ::-1] for x in [ref_img[c]] + [obs[n][c] for n in ("Gp", "Tfc", "Hik", "Hfl")]], 1)
                imageio.imwrite(os.path.join(args.png, f"{args.suite}_t{tid}_e{ep}_{c[:5]}.png"), strip)
        # per-step wall time, zero arm action, gripper open
        tm = {}
        for name in ("Gp", "Tfc", "Hik"):
            set_vars(CELLS[name])
            t0 = time.time()
            for _ in range(args.time_steps):
                envs[name].step([0, 0, 0, 0, 0, 0, -1])
            tm[name] = round((time.time() - t0) / args.time_steps * 1000, 1)
        rec["ms_per_step"] = tm
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(json.dumps({k: rec[k] for k in ("task", "gap_norm_mm_ik", "agent", "wrist", "ms_per_step")}), flush=True)
    for e in envs.values():
        try:
            e.close()
        except Exception:
            pass
print("done", flush=True)
