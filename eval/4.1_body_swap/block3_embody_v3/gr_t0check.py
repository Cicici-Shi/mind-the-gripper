"""G_R step-0 consistency check, protocol_block3_amendment1.md ruling 2 (i)(ii) (written 2026-10-31).

For each (task, episode) of the given shard(s), in ONE process (same node, same GPU, separate render contexts):
  G_R = BodyEnv("Panda", TWIN_RENDER=1, TWIN_BODY=PandaR85, TWIN_MOUNT=sunk, COPY_GRIP_CMD=1)
  T0  = BodyEnv("Panda", TWIN_RENDER=1, TWIN_BODY=Panda,    COPY_GRIP_CMD=1)        (sham_twin)
  G   = BodyEnv("PandaR85", GRP_KP=100, COPY_GRIP_CMD=1)                          (grpkp100)
all on the same initial state. Scene: the global np.random stream is replayed as in run_libero_eval (seed 7, tasks
in order, episodes [off, off+5)); each env's reset sees the same stream position, and the reference fixtures are
then overwritten with the sham_twin dump of the same key (as tools/regress.py does), so the three scenes are
identical; natural_fixture_mm reports the stream-drawn fixtures against the dump.

(i)  G_R vs T0, step 0, policy images (agentview and wrist, both twin-painted): M = union of the pixels whose geom
     belongs to a gripper body (name prefix "gripper0": Panda hand and fingers, Robotiq links and coupling) in either
     frame, from MuJoCo segmentation renders of each twin. Mean |diff| outside M (gate < 0.11 for each camera) and
     inside M (reported). Fallback/companion: gripper geoms of both twins made invisible (alpha 0), both re-rendered,
     whole-frame mean |diff| (gate < 0.11, used only if the segmentation path fails).
(ii) G_R vs G: wrist mean (gate < 8) and agentview mean (reported, no gate); G vs reference Panda and G_R vs
     reference Panda agentview means (reported).
Means are over all pixels and the 3 colour channels of the 256x256 uint8 images, as in BodyEnv._twin_t0.
Output: one jsonl line per key with host and GPU model (amendment ruling 4: pixel numbers only count within a job).

Usage: python tools/gr_t0check.py --out FILE [--offsets 0] [--tasks 0,...,9] [--runs $PROJ_ROOT/runs]
"""
import argparse
import glob
import json
import os
import random
import socket
import subprocess
import sys

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--offsets", default="0")
ap.add_argument("--tasks", default="0,1,2,3,4,5,6,7,8,9")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--code", default=os.path.expanduser("$PROJ_ROOT/embody_v3"))
ap.add_argument("--save-png", default="", help="dir: save the step-0 frames and masks of the first 3 keys")
args = ap.parse_args()

sys.path.insert(0, args.code)
from embody.original import OriginalSpatialSuite  # noqa: E402
from embody.body_env import BodyEnv  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

RES, WAIT = 256, 10
CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))
CELLS = {
    "G_R": ("Panda", {"TWIN_RENDER": "1", "TWIN_BODY": "PandaR85", "TWIN_MOUNT": "sunk", "COPY_GRIP_CMD": "1"}),
    "T0": ("Panda", {"TWIN_RENDER": "1", "TWIN_BODY": "Panda", "COPY_GRIP_CMD": "1"}),
    "G": ("PandaR85", {"GRP_KP": "100", "COPY_GRIP_CMD": "1"}),
}
ALLVARS = {"TWIN_RENDER", "TWIN_BODY", "TWIN_MOUNT", "TWIN_CAMS", "GRP_KP", "COPY_GRIP_CMD", "NO_FOREARM_CONTACT",
           "HIDE_ARM", "RECOLOR_ARM", "GRP_PAD_ALIGN"}

try:
    GPU = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True, text=True,
                         timeout=30).stdout.strip().splitlines()[0]
except Exception as e:
    GPU = f"unknown ({e!r})"
HOST = socket.gethostname()


def set_vars(v):
    for k in ALLVARS:
        os.environ.pop(k, None)
    os.environ.update(v)


def mean_diff(a, b, mask=None):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16)).astype(np.float64)
    if mask is None:
        return float(d.mean())
    sel = d[mask]
    return float(sel.mean()) if sel.size else None


def gripper_mask(sim, cam):
    """Pixels of geoms on gripper bodies, from a segmentation render (decoded here in int64)."""
    ctx = sim._render_context_offscreen
    ctx.render(width=RES, height=RES, camera_id=sim.model.camera_name2id(cam), segmentation=True)
    raw = ctx.read_pixels(RES, RES, segmentation=False).astype(np.int64)  # ID colours, not decoded by robosuite
    seg = raw[:, :, 0] + raw[:, :, 1] * 256 + raw[:, :, 2] * 65536
    scn = ctx.scn
    m = sim.model
    grip_segids = set()
    for i in range(scn.ngeom):
        g = scn.geoms[i]
        if g.segid != -1 and int(g.objtype) == 5 and 0 <= g.objid < m.ngeom:  # mjOBJ_GEOM
            if (m.body_id2name(m.geom_bodyid[g.objid]) or "").startswith("gripper0"):
                grip_segids.add(int(g.segid) + 1)
    return np.isin(seg, list(grip_segids)), len(grip_segids)


def render_rgb(sim, cam):
    return sim.render(width=RES, height=RES, camera_name=cam)


def hidden_render(sim, cam):
    m = sim.model
    gs = [g for g in range(m.ngeom) if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("gripper0")]
    old = m.geom_rgba[gs, 3].copy()
    m.geom_rgba[gs, 3] = 0.0
    try:
        return render_rgb(sim, cam)
    finally:
        m.geom_rgba[gs, 3] = old


def overwrite_fixtures(model, z):
    names = [str(x) for x in z["ref_body_names"]]
    worst = 0.0
    for i, n in enumerate(names):
        if not n:
            continue
        try:
            bid = model.body_name2id(n)
        except (ValueError, KeyError):
            continue
        if not (n == "world" or n.startswith(("robot0", "gripper0", "mount0"))):
            worst = max(worst, float(np.abs(model.body_pos[bid] - z["ref_body_pos"][i]).max()) * 1000)
        model.body_pos[bid] = z["ref_body_pos"][i]
        model.body_quat[bid] = z["ref_body_quat"][i]
    return worst


suite = OriginalSpatialSuite()
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
fh = open(args.out, "a")
print(f"GR_T0CHECK host={HOST} gpu={GPU}", flush=True)
n_saved = 0
for off in [int(x) for x in args.offsets.split(",")]:
    np.random.seed(7)
    random.seed(7)
    for tid in [int(t) for t in args.tasks.split(",")]:
        task = suite.get_task(tid)
        inits = suite.get_task_init_states(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        envs = {}
        for name, (body, v) in CELLS.items():
            st = np.random.get_state()  # constructing the extra envs must not move the stream
            set_vars(v)
            ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES)
            ref.seed(0)
            envs[name] = BodyEnv(ref, bddl, body, RES, WAIT, log=lambda m: None)
            if name != "G_R":
                np.random.set_state(st)
        perm = np.random.RandomState(5478 + tid).permutation(len(inits))
        for ep in range(off, off + args.trials):
            init_id = int(perm[ep])
            fs = glob.glob(os.path.join(args.runs, "sham_twin", f"t{tid}_e{ep}_i*.npz"))
            z = np.load(fs[0], allow_pickle=True) if fs else None
            st = np.random.get_state()
            obs, natural = {}, {}
            for name, (body, v) in CELLS.items():
                np.random.set_state(st)
                set_vars(v)
                e = envs[name]
                e.reset()
                if z is not None:
                    natural[name] = overwrite_fixtures(e.ref.sim.model, z)
                obs[name] = e.set_init_state(inits[init_id])
            rec = {"task": tid, "episode": ep, "init": init_id, "host": HOST, "gpu": GPU,
                   "natural_fixture_mm": natural, "dump_found": z is not None}
            gr, t0, g = envs["G_R"], envs["T0"], envs["G"]
            # reference Panda frame (the G_R reference env: same scene, same init, after the wait steps)
            gr.ref.env._update_observables(force=True)
            ref_obs = gr.ref.env._get_observations()
            # sanity: re-rendering reproduces the policy images (so masks and hidden renders line up with them)
            same = {}
            for name, e in (("G_R", gr), ("T0", t0)):
                for ok, cam in CAMS:
                    same[f"{name}_{cam}"] = bool(np.array_equal(render_rgb(e.twin.sim, cam), obs[name][ok]))
            rec["rerender_equals_obs"] = same
            out_i, out_h = {}, {}
            try:
                for ok, cam in CAMS:
                    m1, n1 = gripper_mask(gr.twin.sim, cam)
                    m2, n2 = gripper_mask(t0.twin.sim, cam)
                    M = m1 | m2
                    short = "agent" if cam == "agentview" else "wrist"
                    out_i[f"{short}_mean_outside_mask"] = mean_diff(obs["G_R"][ok], obs["T0"][ok], ~M)
                    out_i[f"{short}_mean_inside_mask"] = mean_diff(obs["G_R"][ok], obs["T0"][ok], M)
                    out_i[f"mask_px_{short}"] = int(M.sum())
                    out_i[f"grip_segids_{short}"] = [n1, n2]
                    if args.save_png and n_saved < 3:
                        os.makedirs(args.save_png, exist_ok=True)
                        np.savez_compressed(os.path.join(args.save_png, f"t{tid}_e{ep}_{short}.npz"),
                                            gr=obs["G_R"][ok], t0=obs["T0"][ok], g=obs["G"][ok], mask=M)
                out_i["method"] = "segmentation"
            except Exception as ex:  # fallback documented in ruling 2 (i)
                out_i = {"method": "failed", "error": repr(ex)}
            for ok, cam in CAMS:
                short = "agent" if cam == "agentview" else "wrist"
                out_h[f"{short}_mean"] = mean_diff(hidden_render(gr.twin.sim, cam), hidden_render(t0.twin.sim, cam))
            rec["gr_vs_t0"] = out_i
            rec["gr_vs_t0_hidden"] = out_h
            rec["gr_vs_g"] = {"agent_mean": mean_diff(obs["G_R"]["agentview_image"], obs["G"]["agentview_image"]),
                              "wrist_mean": mean_diff(obs["G_R"]["robot0_eye_in_hand_image"], obs["G"]["robot0_eye_in_hand_image"])}
            rec["g_vs_ref"] = {"agent_mean": mean_diff(obs["G"]["agentview_image"], ref_obs["agentview_image"])}
            rec["gr_vs_ref"] = {"agent_mean": mean_diff(obs["G_R"]["agentview_image"], ref_obs["agentview_image"])}
            rec["t0_vs_ref"] = {"agent_mean": mean_diff(obs["T0"]["agentview_image"], ref_obs["agentview_image"]),
                                "wrist_mean": mean_diff(obs["T0"]["robot0_eye_in_hand_image"], ref_obs["robot0_eye_in_hand_image"])}
            rec["gr_twin_pad_mm"] = float(gr.twin_stats["pos_err_mm"][0])
            rec["gr_root_off_mm"] = gr.twin_stats.get("root_off_mm", [None])[0]
            if args.save_png and n_saved < 3:
                n_saved += 1
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(json.dumps({k: rec[k] for k in ("task", "episode", "gr_vs_t0", "gr_vs_t0_hidden", "gr_vs_g")}), flush=True)
        # envs are not closed (EGL close + new context segfaulted on node03, as in run_libero_eval)
print("GR_T0CHECK done", flush=True)
