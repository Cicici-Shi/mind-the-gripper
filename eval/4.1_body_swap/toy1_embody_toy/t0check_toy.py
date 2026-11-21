"""TOY1 step-0 checks (protocol_toy1.md 8.2-8.4 as amended by protocol_toy1_astra_decisions.md (a)), 2026-11-03.
All toy cells are built in ONE process (same node / GPU), every key gets the same initial state: the global
np.random stream is replayed as in run_libero_eval for shards EP_OFFSET 40 and 45 (seed 7, tasks in order, 5
episodes each), each cell's reset/set_init_state runs from the same stream position.
Per key (jsonl row), all means over pixels x 3 channels of the raw 256x256 uint8 policy images:
  8.2' G_T vs G_T^fc (same state): twin arm joints max |diff| (deg) and agentview mean |diff| (expected 0: same twin
       pose, agentview camera is world-fixed); G_T^fc twin wrist camera vs G physical wrist camera (mm, deg; gate 0.01);
       descriptive: G_T^fc vs G with both grippers alpha 0 (the original 8.2 test; arm pixels differ by design).
  8.3  G_hid vs G with G's gripper geoms alpha 0, S_hid vs S likewise (gate: per-cell average < 0.11 per camera);
       hidden geom lists.
  8.4  G_cam physical wrist camera vs G_T twin wrist camera (gate <= 0.01 mm) and vs G physical wrist camera in the
       G grip-site frame (expected [0, 0, -6.4] mm).
--save-png N: t0 policy images (agent | wrist, policy orientation) of all cells for the first N keys.
Usage: python tools/t0check_toy.py --out FILE [--offsets 40,45] [--tasks 0,...,9] [--save-png 5 --png-dir DIR]
"""
import argparse, json, os, random, socket, subprocess, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="40,45")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--save-png", type=int, default=5)
ap.add_argument("--png-dir", default="")
args = ap.parse_args()
TOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOY)
sys.path.insert(0, os.path.join(TOY, "tools"))
from embody import align  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

G = {"BODY": "PandaR85", "GRP_KP": "100", "GRP_MAP": "kp100", "COPY_GRIP_CMD": "1"}
CELLS = {"toy_g": G, "toy_s": {"BODY": "Panda", "COPY_GRIP_CMD": "1"},
         "toy_s_hid": {"BODY": "Panda", "COPY_GRIP_CMD": "1", "HIDE_GRIPPER": "all"},
         "toy_g_hid": dict(G, HIDE_GRIPPER="all"), "toy_g_tfc": dict(G, TWIN_RENDER="1", TWIN_CAM="phys"),
         "toy_g_t": dict(G, TWIN_RENDER="1"), "toy_g_cam": dict(G, WRIST_CAM_DZ_MM="-6.4")}
ALLV = {"BODY", "GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAM", "TWIN_CAMS",
        "HIDE_GRIPPER", "WRIST_CAM_DZ_MM", "TOY_PARTS", "TOY_WRIST_CAM", "TOY_PIXSTATS", "TOY_TIP_SHIFT_MM"}
CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))
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


def md(a, b):
    return float(np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).mean())


def grip_geoms(m):
    return [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")]


def cam_world(sim):
    c = sim.model.camera_name2id("robot0_eye_in_hand")
    return sim.data.cam_xpos[c].copy(), sim.data.cam_xmat[c].reshape(3, 3).copy()


suite = OriginalSpatialSuite()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fh = open(args.out, "a")
print(f"T0CHECK_TOY host={HOST} gpu={GPU}", flush=True)
n_png = 0
for off in [int(x) for x in args.offsets.split(",")]:
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
        for ep in range(off, off + args.trials):
            init_id = int(perm[ep])
            st = np.random.get_state()
            obs, fix = {}, {}
            for name, v in CELLS.items():
                np.random.set_state(st)
                set_vars(v)
                e = envs[name]
                e.reset()
                obs[name] = e.set_init_state(inits[init_id])
                rm = e.ref.sim.model
                fix[name] = np.array([rm.body_pos[b] for b in range(rm.nbody)])
            rec = {"task": tid, "episode": ep, "init": init_id, "host": HOST, "gpu": GPU,
                   "scene_vs_toy_s_mm": {n: float(np.abs(fix[n] - fix["toy_s"]).max() * 1000) for n in CELLS}}
            g, gt, gtf, gh, s, sh, gc = (envs[n] for n in ("toy_g", "toy_g_t", "toy_g_tfc", "toy_g_hid", "toy_s", "toy_s_hid", "toy_g_cam"))
            # 8.2'
            qt = gt.twin.sim.data.qpos[gt.twin.robots[0]._ref_joint_pos_indexes]
            qf = gtf.twin.sim.data.qpos[gtf.twin.robots[0]._ref_joint_pos_indexes]
            pg, Rg = cam_world(g.env.sim)
            pf, Rf = cam_world(gtf.twin.sim)
            rec["gt_vs_gtfc"] = {"twin_arm_deg_max": float(np.degrees(np.abs(qt - qf).max())),
                                 "agent_mean": md(obs["toy_g_t"]["agentview_image"], obs["toy_g_tfc"]["agentview_image"]),
                                 "agent_identical": bool(np.array_equal(obs["toy_g_t"]["agentview_image"], obs["toy_g_tfc"]["agentview_image"]))}
            rec["gtfc_cam_vs_g"] = {"mm": float(np.linalg.norm(pf - pg) * 1000), "deg": float(align.rot_err_deg(Rf, Rg)),
                                    "twin_stats_mm": float(gtf.twin_stats["cam_err_mm"][0])}
            rec["gtfc_vs_g_both_hidden"] = {}
            for ok, cam in CAMS:
                a = gtf._render(gtf.twin.sim, cam, grip_geoms(gtf.twin.sim.model), depth=False, how="alpha")[0]
                b = g._render(g.env.sim, cam, grip_geoms(g.env.sim.model), depth=False, how="alpha")[0]
                rec["gtfc_vs_g_both_hidden"][cam] = md(a, b)
            # 8.3
            for hid, base, nm in ((gh, g, "g_hid"), (sh, s, "s_hid")):
                d = {}
                for ok, cam in CAMS:
                    b = base._render(base.env.sim, cam, grip_geoms(base.env.sim.model), depth=False, how="alpha")[0]
                    d[cam] = md(obs["toy_" + nm][ok], b)
                    d[cam + "_vs_unhidden"] = md(obs["toy_" + nm][ok], obs["toy_" + nm.split("_")[0]][ok])
                d["hidden_geoms"] = hid.align_info.get("hidden_gripper_geoms")
                d["n_visible_gripper_geoms_base"] = int(sum(1 for gg in grip_geoms(base.env.sim.model) if base.env.sim.model.geom_rgba[gg][3] > 0))
                m = hid.env.sim.model
                d["collision_flags_equal"] = bool(all(m.geom_contype[gg] == base.env.sim.model.geom_contype[gg] and m.geom_conaffinity[gg] == base.env.sim.model.geom_conaffinity[gg] for gg in grip_geoms(m)))
                rec[nm] = d
            # 8.4
            pc, Rc = cam_world(gc.env.sim)
            pt, Rt = cam_world(gt.twin.sim)
            _, Rs = align.site_pose(g.env.sim, g.env.robots[0].eef_site_id)
            rec["gcam"] = {"vs_gt_twin_mm": float(np.linalg.norm(pc - pt) * 1000), "vs_gt_twin_deg": float(align.rot_err_deg(Rc, Rt)),
                           "vs_g_in_site_mm": (Rs.T @ (pc - pg) * 1000).tolist(), "vs_g_deg": float(align.rot_err_deg(Rc, Rg))}
            if args.png_dir and n_png < args.save_png:
                from PIL import Image
                os.makedirs(args.png_dir, exist_ok=True)
                names = ["toy_s", "toy_s_hid", "toy_g", "toy_g_hid", "toy_g_cam", "toy_g_t", "toy_g_tfc"]
                row = lambda kk: np.concatenate([np.asarray(obs[n][kk])[::-1, ::-1] for n in names], axis=1)
                Image.fromarray(np.concatenate([row("agentview_image"), row("robot0_eye_in_hand_image")], 0).astype(np.uint8)).save(
                    os.path.join(args.png_dir, f"t0_t{tid}_e{ep}.png"))
                n_png += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps({k: rec[k] for k in ("task", "episode", "gt_vs_gtfc", "gtfc_cam_vs_g", "gcam")}), flush=True)
print("T0CHECK_TOY done", flush=True)
