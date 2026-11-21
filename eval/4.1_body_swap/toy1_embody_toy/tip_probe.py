"""TOY1 virtual fingertip shift probe (2026-11-03). Frozen OpenVLA-OFT, no rollout.

For each cached episode of --cond (dumps with per-step qpos, e.g. a toy cell run by oft/run_libero_eval_toy.py, or a
published twin cell), the scene is rebuilt (reference fixtures from the dump, same init state, BodyEnv with the
TOY_* / TWIN_* variables of the caller), then the dumped physical states are replayed by setting qpos (no physics
step), painting every step so the twin IK warm-start chain is the same as in the episode. At each selected frame the
policy is queried on several renders that differ only in the twin Panda finger bodies:
  base  : the frame as in the episode (shift off)
  +d/-d : both fingers translated by +-d mm along the hand y axis (finger opening axis, = left/right in the wrist view)
  sham  : shift 0 but re-painted after the shifted renders (same code path as +-d) -> re-render noise
Every query stores the raw action chunk (un-normalised, before binarising the gripper; NUM_ACTIONS_CHUNK x 7).
Also stored: mean |pixel diff| of each variant against base (agent, wrist), changed-pixel counts, and the
proprio vector (identical across variants by construction).
Consistency check per frame: process_action(base chunk) vs the dumped executed actions of steps [k, k+8) of that
episode (only meaningful for query frames of a dump made with the same cell and code).

Frames: --frames preclose:N = the last N policy-query steps (k % 8 == 0, or the dump's query_step) before the first
close command of the episode (approach / pre-grasp); or an explicit comma list of steps.

Output: --out DIR/probe_t{task}_e{ep}.npz and DIR/probe.jsonl (no success rates are read or printed).
Usage (cwd $PROJ_ROOT/embody_toy, PYTHONPATH=$PROJ_ROOT/openvla-oft):
  env BODY=PandaR85 GRP_KP=100 COPY_GRIP_CMD=1 TWIN_RENDER=1 TOY_PARTS=fingers \
    python tools/tip_probe.py --runs out/runs --cond toy_fingers --keys all --d 5 --out out/probe_fingers
"""
import argparse, glob, importlib.util, json, os, sys, time
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--runs", required=True)
ap.add_argument("--cond", required=True)
ap.add_argument("--keys", default="all", help="all or 't0e0,t2e3,...'")
ap.add_argument("--frames", default="preclose:5")
ap.add_argument("--d", type=float, default=5.0, help="shift in mm")
ap.add_argument("--axis", default="y", choices=["x", "y", "z"])
ap.add_argument("--out", required=True)
ap.add_argument("--checkpoint", default="moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10")
ap.add_argument("--suite", default="libero_spatial")
args = ap.parse_args()
os.makedirs(args.out, exist_ok=True)
TOY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, TOY)
spec = importlib.util.spec_from_file_location("rle_toy", os.path.join(TOY, "oft", "run_libero_eval_toy.py"))
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)
from embody.body_env import BodyEnv  # noqa: E402
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

BODY = os.environ["BODY"]
cfg = R.GenerateConfig(pretrained_checkpoint=args.checkpoint, task_suite_name=args.suite)
R.set_seed_everywhere(cfg.seed)
model, action_head, proprio_projector, noisy_action_projector, processor = R.initialize_model(cfg)
resize_size = R.get_image_resize_size(cfg)
suite = OriginalSpatialSuite()
AX = {"x": 0, "y": 1, "z": 2}[args.axis]


def query(obs, lang):
    o, _ = R.prepare_observation(obs, resize_size)
    a = R.get_action(cfg, model, o, lang, processor=processor, action_head=action_head,
                     proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector, use_film=cfg.use_film)
    return np.asarray(a, dtype=np.float64), o["state"]


def set_ref_fixtures(rm, z):
    for i, n in enumerate(str(x) for x in z["ref_body_names"]):
        if not n:
            continue
        try:
            bid = rm.body_name2id(n)
        except (ValueError, KeyError):
            continue
        rm.body_pos[bid] = z["ref_body_pos"][i]
        rm.body_quat[bid] = z["ref_body_quat"][i]


def state_obs(env, qpos):
    """Physical state = dumped qpos (velocities zero; images depend on positions only), then the BodyEnv paint."""
    e = env.env
    e.sim.data.qpos[:] = qpos
    e.sim.data.qvel[:] = 0.0
    e.sim.forward()
    e._update_observables(force=True)
    obs = e._get_observations()
    if env.twin is not None:
        obs = env._twin_paint(obs)
    return env._fix(obs)


def imdiff(a, b):
    d = np.abs(np.asarray(a, np.int16) - np.asarray(b, np.int16))
    return float(d.mean()), int((d.max(-1) > 0).sum())


files = sorted(glob.glob(os.path.join(args.runs, args.cond, "t*_e*_i*.npz")))
if args.keys != "all":
    want = set(args.keys.split(","))
    files = [f for f in files if "_".join(os.path.basename(f).split("_")[:2]).replace("_", "") in want]
by_task = {}
for f in files:
    b = os.path.basename(f).split("_")
    by_task.setdefault(int(b[0][1:]), []).append((int(b[1][1:]), f))
fh = open(os.path.join(args.out, "probe.jsonl"), "a")
print(f"TIP_PROBE body={BODY} cond={args.cond} keys={len(files)} d={args.d} axis={args.axis}", flush=True)
for tid in sorted(by_task):
    task = suite.get_task(tid)
    inits = suite.get_task_init_states(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    env = BodyEnv(ref, bddl, BODY, 256, 10, log=lambda m: None)
    assert env.twin is not None and env.twin_gk == "panda", "the probe shifts twin Panda fingers: TWIN_RENDER=1 needed"
    for ep, f in sorted(by_task[tid]):
        t_start = time.time()
        z = np.load(f, allow_pickle=True)
        env.reset()
        set_ref_fixtures(env.ref.sim.model, z)
        env.set_tip_shift(None)
        obs = env.set_init_state(inits[int(z["init"])])
        q0_diff = float(np.abs(env.env.sim.data.qpos - z["qpos"][0]).max())
        acts = z["action"]
        if args.frames.startswith("preclose:"):
            n = int(args.frames.split(":")[1])
            closes = np.nonzero(acts[:, -1] > 0)[0]
            c = int(closes[0]) if len(closes) else len(acts)
            qs = [int(k) for k in z["query_step"]] if "query_step" in z.files else list(range(0, len(acts), 8))
            frames = [k for k in qs if k < c][-n:]
        else:
            frames = [int(k) for k in args.frames.split(",") if int(k) < len(z["qpos"])]
        variants = [("base", None), ("plus", args.d), ("minus", -args.d), ("sham", 0.0)]
        chunks = {v: [] for v, _ in variants}
        diffs = {v: [] for v, _ in variants if v != "base"}
        states, cons = [], []
        kmax = max(frames) if frames else -1
        # walk the dumped states in order (step 0 was painted by set_init_state) so the twin IK chain is the episode's
        for k in range(0, kmax + 1):
            if k > 0:
                env.set_tip_shift(None)
                obs = state_obs(env, z["qpos"][k])
            if k not in frames:
                continue
            base_img = {kk: np.array(obs[kk], copy=True) for kk in ("agentview_image", "robot0_eye_in_hand_image")}
            for v, dmm in variants:
                if v == "base":
                    o = obs
                else:
                    sh = np.zeros(3)
                    sh[AX] = dmm / 1000.0
                    env.set_tip_shift(sh)
                    o = state_obs(env, z["qpos"][k])
                    diffs[v].append([*imdiff(o["agentview_image"], base_img["agentview_image"]),
                                     *imdiff(o["robot0_eye_in_hand_image"], base_img["robot0_eye_in_hand_image"])])
                ch, stt = query(o, task.language)
                chunks[v].append(ch)
                if v == "base":
                    states.append(stt)
                    exe = np.stack([R.process_action(a.copy(), cfg.model_family) for a in ch])
                    m = min(len(exe), len(acts) - k)
                    cons.append(float(np.abs(exe[:m] - acts[k:k + m]).max()) if m > 0 else None)
            env.set_tip_shift(None)
        rec = {"task": tid, "ep": ep, "init": int(z["init"]), "cond": args.cond, "body": BODY, "frames": frames,
               "q0_vs_dump": q0_diff, "base_vs_dump_exec_maxdiff": cons, "d_mm": args.d, "axis": args.axis,
               "pixdiff_cols": ["agent_mean", "agent_npx", "wrist_mean", "wrist_npx"],
               "pixdiff": {v: np.asarray(diffs[v]).tolist() for v in diffs}, "sec": round(time.time() - t_start, 1),
               "toy": getattr(env, "toy_info", None)}
        np.savez_compressed(os.path.join(args.out, f"probe_t{tid}_e{ep}.npz"), frames=np.asarray(frames),
                            state=np.asarray(states), **{f"chunk_{v}": np.asarray(chunks[v]) for v in chunks},
                            **{f"pixdiff_{v}": np.asarray(diffs[v]) for v in diffs}, d_mm=args.d, axis=args.axis)
        fh.write(json.dumps(rec) + "\n"); fh.flush()
        print(json.dumps({k: rec[k] for k in ("task", "ep", "frames", "q0_vs_dump", "base_vs_dump_exec_maxdiff", "sec")}), flush=True)
print("TIP_PROBE done", flush=True)
