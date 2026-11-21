"""next (2026-11-06): copy of vla_zoo/common/zoo_client.py (frozen models1 client, sha256 e0182776...) for the next
model batch (pi0-FAST via openpi z-score server, SmolVLA via zoo_server + adapter_lerobot). Changes, all additive:
  * --blind: the record and the log carry NO outcome and nothing that reveals episode length (no success, steps,
    queries, wall_s, contacts.n_steps); throughput is recorded per control step instead (sim_ms_step_med,
    loop_ms_step_mean, infer_ms_*). For interface smoke runs before preregistration.
  * twin: rot_err_deg_q and miss_q per query frame (tools/PATCH_NEXT_zoo_client_rot.md, ruling section 4.3).
  * physics breakdown detector D online (thresholds of tools/physics_breakdown_scan.py, unchanged): D-A arm joint
    jump > pi/2 between consecutive control steps or beyond Panda limits by 0.1 rad; D-B Robotiq finger_joint outside
    [-0.5, 1.2] (loose [-1, 2] flagged too); plus the MuJoCo BADQACC warning counter of the physical MjData. Only
    first_step and booleans are written (no counts that reveal episode length).
  * libero_goal / libero_10 suites (as g10/zoo_client_g10.py; embody.original first if EMBODY_DIR provides them).
  * --dump-inputs DIR: pickle the exact policy request at query indices --dump-queries (load-test replay).
Everything else (init selection, wait steps, BodyEnv, fixtures, contact log, twin fields) is byte-for-byte the
models1 logic, so keys pair with models1 / published cells.

vla_zoo client (2026-11-03): copy of $PROJ_ROOT/pi05/pi05_body_eval.py, generalised to any policy server.

--proto openpi : unchanged pi0.5 element (flip 180, resize_with_pad, 8-d state) -> openpi serve_policy (pi0, pi0-FAST).
--proto raw    : sends the raw env observation (256x256 images exactly as rendered, i.e. NOT flipped; eef pos,
                 eef quat xyzw, gripper qpos, joint pos, prompt). The model-specific server (vla_zoo/common/zoo_server.py
                 + an adapter) does all preprocessing and returns actions in LIBERO env convention
                 (7-d: delta pos, delta axis-angle, gripper -1 open / +1 close).
Everything else (suite, init-state selection, wait steps, BodyEnv, records) is identical to pi05_body_eval.py, so
episodes pair key-by-key with the OFT / pi0.5 runs.

Original docstring:
pi0.5-LIBERO websocket client on the 10 ORIGINAL libero_spatial tasks.

Mirrors openpi examples/libero/main.py (preprocessing, 8-d state, replan every 5 steps)
but uses embody.original.OriginalSpatialSuite and the OpenVLA-OFT init-state selection
(np.random.RandomState(5478 + task_id).permutation(n)[episode_idx]) so episodes pair with OFT runs.

Run from the `oftpi` conda env while a pi05_libero policy server listens on --port.
"""
import argparse
import collections
import json
import math
import os
import random
import sys
import time

import numpy as np

sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody")))
from embody.original import OriginalSpatialSuite  # noqa: E402
if os.environ.get("M1_FASTSYNC") == "1":  # models1 2026-11-04: bitwise-equivalent fast TWIN_CAM_SYNC snapshot (m1_fastsync.py)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import m1_fastsync  # noqa: E402
    m1_fastsync.patch()
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256          # openpi renders LIBERO at 256 (OFT also uses env_img_res=256)
MAX_STEPS = 220        # libero_spatial (same in OFT and openpi)
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280,  # NUS-PATCH 2026-10-27 (block 3: --suite)
                   "libero_goal": 300, "libero_10": 520}  # next: as g10/zoo_client_g10.py (openpi examples/libero/main.py)
PANDA_LIM = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973],
                      [-0.0175, 3.7525], [-2.8973, 2.8973]])  # = tools/physics_breakdown_scan.py LIM
FIN_LO, FIN_HI = -0.5, 1.2  # detector D-B (physics_breakdown_scan defaults); loose [-1, 2]


class BreakdownD:
    """Detector D of tools/physics_breakdown_scan.py, evaluated online on the PHYSICAL sim every control step."""

    def __init__(self, penv):
        self.penv = penv
        sim = penv.sim
        self.ai = list(penv.robots[0]._ref_joint_pos_indexes)
        try:
            jid = sim.model.joint_name2id("gripper0_finger_joint")
            self.fi = int(sim.model.jnt_qposadr[jid])
        except Exception:
            self.fi = None  # Panda gripper: finger slide joints, D-B not defined
        self.prev, self.A, self.B, self.loose, self.mj = None, None, None, False, 0.0
        self.k, self.qacc0 = 0, self._qacc()

    def _qacc(self):
        try:
            import mujoco
            return int(self.penv.sim.data._data.warning[int(mujoco.mjtWarning.mjWARN_BADQACC)].number)
        except Exception:
            return -1

    def update(self):
        q = np.asarray(self.penv.sim.data.qpos, np.float64)
        qa = q[self.ai]
        if self.prev is not None:
            dq = float(np.abs(qa - self.prev).max())
            self.mj = max(self.mj, dq)
            if dq > np.pi / 2 and self.A is None:
                self.A = {"first_step": self.k, "jump": True}
        if self.A is None and ((qa < PANDA_LIM[:, 0] - 0.1) | (qa > PANDA_LIM[:, 1] + 0.1)).any():
            self.A = {"first_step": self.k, "jump": False}
        if self.fi is not None:
            f = float(q[self.fi])
            if self.B is None and (f < FIN_LO or f > FIN_HI):
                self.B = {"first_step": self.k}
            self.loose = self.loose or f < -1.0 or f > 2.0
        self.prev = qa.copy()
        self.k += 1

    def summary(self):
        q1 = self._qacc()
        return {"A": self.A, "B": self.B, "breakdown": bool(self.A or self.B), "B_loose": bool(self.loose),
                "max_jump_deg": round(float(np.degrees(self.mj)), 1), "finger_joint": self.fi is not None,
                "qacc_warn": (q1 - self.qacc0) if (q1 >= 0 and self.qacc0 >= 0) else None}
NUM_STEPS_WAIT = 10


def quat2axisangle(quat):
    """robosuite transform_utils.quat2axisangle (xyzw), as copied in openpi's example."""
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def make_raw(obs, prompt, suite):
    return {
        "agentview": np.ascontiguousarray(obs["agentview_image"]),
        "wrist": np.ascontiguousarray(obs["robot0_eye_in_hand_image"]),
        "eef_pos": np.asarray(obs["robot0_eef_pos"], np.float64),
        "eef_quat": np.asarray(obs["robot0_eef_quat"], np.float64),
        "gripper_qpos": np.asarray(obs["robot0_gripper_qpos"], np.float64),
        "joint_pos": np.asarray(obs["robot0_joint_pos"], np.float64),
        "prompt": str(prompt), "suite": suite,
    }


def make_element(obs, prompt, resize):
    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
    wrist = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(img, resize, resize))
    wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(wrist, resize, resize))
    state = np.concatenate(
        (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
    ).astype(np.float64)
    return {"observation/image": img, "observation/wrist_image": wrist, "observation/state": state, "prompt": str(prompt)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--trials", type=int, default=2, help="episodes per task")
    p.add_argument("--ep-offset", type=int, default=0, help="first episode index (permutation position)")
    p.add_argument("--tasks", default="all", help="comma list of task ids or 'all'")
    p.add_argument("--replan", type=int, default=0, help="actions executed per query; 0 = server metadata 'replan' (else 5)")
    p.add_argument("--max-steps", type=int, default=0, help="override the suite step budget (baseline alignment only)")
    p.add_argument("--proto", choices=("openpi", "raw"), default="raw")
    p.add_argument("--model", default="", help="model name written into every record")
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7, help="global seed, same as OFT set_seed_everywhere(7)")
    p.add_argument("--blind", action="store_true", help="next: no outcome / episode-length fields in record or log")
    p.add_argument("--dump-inputs", default="", help="next: dir, pickle the exact request at --dump-queries indices")
    p.add_argument("--suite", default="libero_spatial", choices=list(SUITE_MAX_STEPS),
                   help="NUS-PATCH 2026-10-27: libero_object uses embody.original.OriginalObjectSuite (embody_v3)")
    p.add_argument("--body", default="", help="Panda|IIWA|UR5e: run on another body via embody.BodyEnv (HIDE_ARM env var hides the arm)")
    p.add_argument("--out", required=True, help="jsonl, one line per episode (appended)")
    p.add_argument("--skip-done", action="store_true",
                   help="skip (task, episode) already in --out without error; the env is still reset so the "
                        "global np.random stream (fixture placement) advances exactly as in a full run")
    p.add_argument("--dump-obs", default="", help="dir: save the policy input of selected queries as .npz")
    p.add_argument("--dump-queries", default="0,10", help="query indices per episode to dump")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    global MAX_STEPS
    MAX_STEPS = args.max_steps or SUITE_MAX_STEPS[args.suite]
    if args.suite in ("libero_goal", "libero_10"):  # next: embody.original if EMBODY_DIR has it, else g10 rebuild
        import embody.original as _eo
        _cls = {"libero_goal": "OriginalGoalSuite", "libero_10": "Original10Suite"}[args.suite]
        if hasattr(_eo, _cls):
            suite = getattr(_eo, _cls)()
        else:
            sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/vla_zoo/g10"))
            from original_g10 import SUITES as _G10
            suite = _G10[args.suite]()
        print(f"suite {args.suite} from {type(suite).__module__}.{type(suite).__name__}", flush=True)
    elif args.suite == "libero_object":
        from embody.original import OriginalObjectSuite
        suite = OriginalObjectSuite()
    else:
        suite = OriginalSpatialSuite()
    try:  # NUS-PATCH 2026-10-27: online arm-environment contacts (embody_v3 only; older EMBODY_DIR has none)
        from embody import contacts as _contacts
    except ImportError:
        _contacts = None
    task_ids = range(suite.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    dump_q = {int(x) for x in args.dump_queries.split(",") if x != ""}
    if args.dump_obs:
        os.makedirs(args.dump_obs, exist_ok=True)
    if args.dump_inputs:
        os.makedirs(args.dump_inputs, exist_ok=True)
        import pickle

    # The server compiles on its first inference (tens of seconds on A100) and blocks its event loop
    # meanwhile, so the client's 20 s keepalive ping times out and closes the connection; every later
    # episode then fails with ConnectionClosedError and is recorded as a policy failure. Localhost
    # needs no keepalive: disable it, and reconnect + retry the query if the connection still drops.
    import websockets.sync.client as _wsc
    import websockets.exceptions as _wse
    _connect = _wsc.connect
    _wsc.connect = lambda *a, **k: _connect(*a, **{"ping_interval": None, **k})
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    meta = client.get_server_metadata()
    print("server metadata:", meta, flush=True)
    if not args.replan:
        args.replan = int((meta or {}).get("replan", 5))
    print(f"replan={args.replan} proto={args.proto} model={args.model}", flush=True)

    def infer(el):
        nonlocal client
        for attempt in range(4):
            try:
                return client.infer(el)
            except (_wse.ConnectionClosed, ConnectionError, OSError) as e:
                print(f"INFER_RETRY attempt={attempt} {e!r}", flush=True)
                time.sleep(5)
                client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
        return client.infer(el)

    done_keys = set()
    if args.skip_done and os.path.exists(args.out):
        for line in open(args.out):
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    done_keys.add((r["task"], r["episode"]))

    import socket  # NUS-PATCH 2026-10-31 (block 3 amendment 1, R8/R10): node and GPU model in every record
    host, gpu = socket.gethostname(), os.environ.get("NUS_GPU", "")
    if not gpu:
        try:
            import subprocess
            gpu = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"], capture_output=True,
                                 text=True, timeout=30).stdout.strip().splitlines()[0]
        except Exception as e:
            gpu = f"unknown ({e!r})"
    n_ep = n_succ = 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(0)
        if args.body:  # same wrapper as the OFT runs: Panda reference does the wait steps, scene transferred
            from embody.body_env import BodyEnv
            env = BodyEnv(env, bddl, args.body, ENV_RES, NUM_STEPS_WAIT, log=lambda m: print(m, flush=True))
        perm = np.random.RandomState(5478 + task_id).permutation(len(init_states))
        for ep in range(args.ep_offset, args.ep_offset + args.trials):
            init_id = int(perm[ep])
            t0 = time.time()
            env.reset()
            if (task_id, ep) in done_keys:
                continue
            obs = env.set_init_state(init_states[init_id])
            # scene fingerprint: fixture positions of the Panda reference scene, compared with the OFT
            # dumps (ref_body_names/pos) to confirm both models saw the same scene for this (task, episode)
            ref_env = env.ref if args.body else env
            from embody import align as _align
            fix = {k: [round(float(x), 5) for x in v[0]] for k, v in _align.static_body_pose(ref_env).items()}
            objq = {k: [round(float(x), 5) for x in v[0]] for k, v in _align.object_state_by_name(ref_env).items()}
            clog, clog_err = None, None
            plan = collections.deque()
            t = NUM_STEPS_WAIT if getattr(env, "skip_wait", False) else 0
            n_q, infer_ms, success, err = 0, [], False, None
            a0, chunk_len = None, None  # models1 (2026-11-04): first executed block (E1/E2 bitwise check), chunk length
            penv = env.env if args.body else env  # next: physical env (BodyEnv target body, or the native env)
            det = BreakdownD(penv)
            sim_ms, srv_ms, t_loop0, n_exec = [], [], None, 0
            try:
                while t < MAX_STEPS + NUM_STEPS_WAIT:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    if not plan:
                        el = (make_element(obs, task.language, args.resize) if args.proto == "openpi"
                              else make_raw(obs, task.language, args.suite))
                        if args.proto == "raw":
                            el["new_episode"] = n_q == 0
                        if args.dump_obs and n_q in dump_q and args.proto == "openpi":
                            np.savez_compressed(
                                os.path.join(args.dump_obs, f"t{task_id}_e{ep}_q{n_q}.npz"),
                                image=el["observation/image"], wrist_image=el["observation/wrist_image"],
                                state=el["observation/state"], prompt=el["prompt"],
                            )
                        if args.dump_inputs and n_q in dump_q:
                            with open(os.path.join(args.dump_inputs, f"t{task_id}_e{ep}_q{n_q}.pkl"), "wb") as fh:
                                pickle.dump({k: v for k, v in el.items()}, fh)
                        if t_loop0 is None:
                            t_loop0 = time.time()
                        tq = time.time()
                        res = infer(el)
                        infer_ms.append((time.time() - tq) * 1000)
                        st_ = res.get("server_timing") if isinstance(res, dict) else None
                        if st_ and "infer_ms" in st_:
                            srv_ms.append(float(st_["infer_ms"]))
                        chunk = res["actions"]
                        chunk = np.asarray(chunk, np.float64).reshape(-1, 7)
                        assert len(chunk) >= args.replan, chunk.shape
                        if a0 is None:
                            a0, chunk_len = chunk[: args.replan].tolist(), int(len(chunk))
                        plan.extend(chunk[: args.replan])
                        n_q += 1
                    action = plan.popleft()
                    if clog is None and _contacts is not None and clog_err is None:
                        try:
                            clog = _contacts.make(env)
                        except Exception as ce:  # never let the contact log break an episode
                            clog_err = repr(ce)
                    ts_ = time.time()
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
                    sim_ms.append((time.time() - ts_) * 1000)
                    n_exec += 1
                    try:
                        det.update()
                    except Exception as de:
                        det.A = det.A or {"error": repr(de)}
                    if clog is not None:
                        try:
                            clog.update(np.asarray(action))
                        except Exception as ce:
                            clog_err, clog = repr(ce), None
                    if done:
                        success = True
                        break
                    t += 1
            except Exception as e:  # keep going; record the error
                import traceback
                err = f"{e!r}"
                traceback.print_exc()
            n_ep += 1
            n_succ += int(success)
            rec = {
                "model": args.model, "proto": args.proto, "max_steps": MAX_STEPS, "replan": args.replan, "task": task_id, "task_name": task.name, "episode": ep, "init": init_id, "success": success,
                "steps": t - NUM_STEPS_WAIT, "queries": n_q, "wall_s": round(time.time() - t0, 2),
                "suite": args.suite, "objq": objq, "host": host, "gpu": gpu,
                "align": getattr(env, "align_info", None),
                "contacts": clog.summary() if clog is not None else ({"error": clog_err} if clog_err else None),
                "a0": a0, "chunk_len": chunk_len, "server_host": os.environ.get("NUS_SRV_HOST", args.host),
                "server_gpu": os.environ.get("NUS_SRV_GPU", ""), "embody_dir": os.environ.get("EMBODY_DIR", ""), "fastsync": os.environ.get("M1_FASTSYNC", "0"),
                "rtt_ms_med": round(float(np.median(infer_ms)), 1) if infer_ms else None,
                "cell_env": {k: os.environ.get(k) for k in ("BODY", "GRP_KP", "GRP_MAP", "COPY_GRIP_CMD", "TWIN_RENDER",
                             "TWIN_BODY", "TWIN_CAMS", "TWIN_CAM", "TWIN_CAM_SYNC") if os.environ.get(k) is not None},
                "body": args.body or "SRC", "hide_arm": os.environ.get("HIDE_ARM", "0"), "rtt_ms_mean": round(float(np.mean(infer_ms)), 1) if infer_ms else None, "error": err, "fix": fix,
                "twin": None if getattr(env, "twin_stats", None) is None else {
                    "pos_err_mm_max": float(np.max(env.twin_stats["pos_err_mm"])),
                    "rot_err_deg_max": float(np.max(env.twin_stats["rot_err_deg"])),
                    "n_paint": len(env.twin_stats["pos_err_mm"]),
                    "miss": int(sum(1 for a, b in zip(env.twin_stats["pos_err_mm"], env.twin_stats["rot_err_deg"]) if a > 2 or b > 1)),
                    "posture_deg_p95": float(np.percentile(env.twin_stats["posture_deg"], 95)) if env.twin_stats["posture_deg"] else None,
                    "stale": env.twin_stats["stale"], "t0": env.twin_stats["t0"],
                    "over_1mm": int(sum(1 for a in env.twin_stats["pos_err_mm"] if a > 1)),
                    "over_1mm_q": int(sum(1 for a in env.twin_stats["pos_err_mm"][::args.replan] if a > 1)),
                    **({} if not env.twin_stats.get("cam_err_mm") else {  # models1: twin vs physical wrist camera
                        "cam_err_mm_max": float(np.max(env.twin_stats["cam_err_mm"])),
                        "cam_err_deg_max": float(np.max(env.twin_stats["cam_err_deg"])),
                        "cam_err_mm_mean": float(np.mean(env.twin_stats["cam_err_mm"])),
                        "cam_err_mm_q": [round(float(a), 5) for a in env.twin_stats["cam_err_mm"][::args.replan]],
                        "cam_err_deg_q": [round(float(a), 5) for a in env.twin_stats["cam_err_deg"][::args.replan]]}),
                    "pos_err_mm_q": [round(float(a), 4) for a in env.twin_stats["pos_err_mm"][::args.replan]],
                    **({} if not env.twin_stats.get("root_off_mm") else {  # NUS-PATCH 2026-10-31: G_R root offset
                        "root_off_mm_q": [[round(float(x), 3) for x in v] for v in env.twin_stats["root_off_mm"][::args.replan]]}),
                },
            }
            # next: throughput per control step, physics breakdown detector D, rotation residual per query frame
            loop_s = (time.time() - t_loop0) if t_loop0 is not None else None
            rec["timing"] = {"sim_ms_step_med": round(float(np.median(sim_ms)), 2) if sim_ms else None,
                             "loop_ms_step_mean": round(1000 * loop_s / n_exec, 2) if (loop_s and n_exec) else None,
                             "infer_ms_med": rec["rtt_ms_med"], "server_infer_ms_med":
                             round(float(np.median(srv_ms)), 1) if srv_ms else None, "replan": args.replan}
            rec["breakdown_D"] = det.summary()
            if rec["twin"] is not None:
                _pe = env.twin_stats["pos_err_mm"][::args.replan]
                _re = env.twin_stats["rot_err_deg"][::args.replan]
                rec["twin"]["rot_err_deg_q"] = [round(float(a), 4) for a in _re]
                rec["twin"]["miss_q"] = int(sum(1 for a, b in zip(_pe, _re) if a > 2 or b > 1))
            if args.blind:  # next: nothing that is, or reveals, the outcome or the episode length
                for k_ in ("success", "steps", "queries", "wall_s"):
                    rec.pop(k_, None)
                if isinstance(rec.get("contacts"), dict):
                    rec["contacts"].pop("n_steps", None)
                if isinstance(rec.get("twin"), dict):
                    rec["twin"].pop("n_paint", None)  # = episode length
                rec["blind"] = True
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            # models1 7.5: no outcome in the log (only interface quantities); next --blind: no length proxies either
            if args.blind:
                print(f"EP task={task_id} ep={ep} init={init_id} error={err}", flush=True)
            else:
                print(f"EP task={task_id} ep={ep} init={init_id} queries={n_q} wall_s={rec['wall_s']} error={err}", flush=True)
        env.close()
    print(f"FINAL episodes {n_ep}", flush=True)  # models1 7.5: success count no longer printed


if __name__ == "__main__":
    main()
