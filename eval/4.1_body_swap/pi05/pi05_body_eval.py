"""pi0.5-LIBERO websocket client on the 10 ORIGINAL libero_spatial tasks.

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
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256          # openpi renders LIBERO at 256 (OFT also uses env_img_res=256)
MAX_STEPS = 220        # libero_spatial (same in OFT and openpi)
SUITE_MAX_STEPS = {"libero_spatial": 220, "libero_object": 280}  # NUS-PATCH 2026-10-27 (block 3: --suite)
NUM_STEPS_WAIT = 10


def quat2axisangle(quat):
    """robosuite transform_utils.quat2axisangle (xyzw), as copied in openpi's example."""
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


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
    p.add_argument("--replan", type=int, default=5)
    p.add_argument("--resize", type=int, default=224)
    p.add_argument("--seed", type=int, default=7, help="global seed, same as OFT set_seed_everywhere(7)")
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
    MAX_STEPS = SUITE_MAX_STEPS[args.suite]
    if args.suite == "libero_object":
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

    # The server compiles on its first inference (tens of seconds on A100) and blocks its event loop
    # meanwhile, so the client's 20 s keepalive ping times out and closes the connection; every later
    # episode then fails with ConnectionClosedError and is recorded as a policy failure. Localhost
    # needs no keepalive: disable it, and reconnect + retry the query if the connection still drops.
    import websockets.sync.client as _wsc
    import websockets.exceptions as _wse
    _connect = _wsc.connect
    _wsc.connect = lambda *a, **k: _connect(*a, **{"ping_interval": None, **k})
    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("server metadata:", client.get_server_metadata(), flush=True)

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
            try:
                while t < MAX_STEPS + NUM_STEPS_WAIT:
                    if t < NUM_STEPS_WAIT:
                        obs, _, done, _ = env.step(DUMMY)
                        t += 1
                        continue
                    if not plan:
                        el = make_element(obs, task.language, args.resize)
                        if args.dump_obs and n_q in dump_q:
                            np.savez_compressed(
                                os.path.join(args.dump_obs, f"t{task_id}_e{ep}_q{n_q}.npz"),
                                image=el["observation/image"], wrist_image=el["observation/wrist_image"],
                                state=el["observation/state"], prompt=el["prompt"],
                            )
                        tq = time.time()
                        res = infer(el)
                        infer_ms.append((time.time() - tq) * 1000)
                        chunk = res["actions"]
                        assert len(chunk) >= args.replan, chunk.shape
                        plan.extend(chunk[: args.replan])
                        n_q += 1
                    action = plan.popleft()
                    if clog is None and _contacts is not None and clog_err is None:
                        try:
                            clog = _contacts.make(env)
                        except Exception as ce:  # never let the contact log break an episode
                            clog_err = repr(ce)
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
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
                "task": task_id, "task_name": task.name, "episode": ep, "init": init_id, "success": success,
                "steps": t - NUM_STEPS_WAIT, "queries": n_q, "wall_s": round(time.time() - t0, 2),
                "suite": args.suite, "objq": objq, "host": host, "gpu": gpu,
                "align": getattr(env, "align_info", None),
                "contacts": clog.summary() if clog is not None else ({"error": clog_err} if clog_err else None),
                "body": args.body or "SRC", "hide_arm": os.environ.get("HIDE_ARM", "0"), "rtt_ms_mean": round(float(np.mean(infer_ms)), 1) if infer_ms else None, "error": err, "fix": fix,
                "twin": None if getattr(env, "twin_stats", None) is None else {
                    "pos_err_mm_max": float(np.max(env.twin_stats["pos_err_mm"])),
                    "rot_err_deg_max": float(np.max(env.twin_stats["rot_err_deg"])),
                    "n_paint": len(env.twin_stats["pos_err_mm"]),
                    "miss": int(sum(1 for a, b in zip(env.twin_stats["pos_err_mm"], env.twin_stats["rot_err_deg"]) if a > 2 or b > 1)),
                    "posture_deg_p95": float(np.percentile(env.twin_stats["posture_deg"], 95)) if env.twin_stats["posture_deg"] else None,
                    "stale": env.twin_stats["stale"], "t0": env.twin_stats["t0"],
                    "over_1mm": int(sum(1 for a in env.twin_stats["pos_err_mm"] if a > 1)),
                    "pos_err_mm_q": [round(float(a), 4) for a in env.twin_stats["pos_err_mm"][::args.replan]],
                    **({} if not env.twin_stats.get("root_off_mm") else {  # NUS-PATCH 2026-10-31: G_R root offset
                        "root_off_mm_q": [[round(float(x), 3) for x in v] for v in env.twin_stats["root_off_mm"][::args.replan]]}),
                },
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"EP {json.dumps(rec)}  running {n_succ}/{n_ep}", flush=True)
        env.close()
    print(f"FINAL success {n_succ}/{n_ep} = {n_succ / max(n_ep, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
