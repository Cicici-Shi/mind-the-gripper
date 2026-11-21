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

sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody"))
from embody.original import OriginalSpatialSuite  # noqa: E402
from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402
from openpi_client import image_tools  # noqa: E402
from openpi_client import websocket_client_policy  # noqa: E402

DUMMY = [0.0] * 6 + [-1.0]
ENV_RES = 256          # openpi renders LIBERO at 256 (OFT also uses env_img_res=256)
MAX_STEPS = 220        # libero_spatial (same in OFT and openpi)
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
    p.add_argument("--out", required=True, help="jsonl, one line per episode (appended)")
    p.add_argument("--dump-obs", default="", help="dir: save the policy input of selected queries as .npz")
    p.add_argument("--dump-queries", default="0,10", help="query indices per episode to dump")
    args = p.parse_args()

    np.random.seed(args.seed)
    random.seed(args.seed)
    suite = OriginalSpatialSuite()
    task_ids = range(suite.n_tasks) if args.tasks == "all" else [int(x) for x in args.tasks.split(",")]
    dump_q = {int(x) for x in args.dump_queries.split(",") if x != ""}
    if args.dump_obs:
        os.makedirs(args.dump_obs, exist_ok=True)

    client = websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    print("server metadata:", client.get_server_metadata(), flush=True)

    n_ep = n_succ = 0
    for task_id in task_ids:
        task = suite.get_task(task_id)
        init_states = suite.get_task_init_states(task_id)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=ENV_RES, camera_widths=ENV_RES)
        env.seed(0)
        perm = np.random.RandomState(5478 + task_id).permutation(len(init_states))
        for ep in range(args.ep_offset, args.ep_offset + args.trials):
            init_id = int(perm[ep])
            t0 = time.time()
            env.reset()
            obs = env.set_init_state(init_states[init_id])
            plan = collections.deque()
            t, n_q, infer_ms, success, err = 0, 0, [], False, None
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
                        res = client.infer(el)
                        infer_ms.append((time.time() - tq) * 1000)
                        chunk = res["actions"]
                        assert len(chunk) >= args.replan, chunk.shape
                        plan.extend(chunk[: args.replan])
                        n_q += 1
                    action = plan.popleft()
                    obs, _, done, _ = env.step(np.asarray(action).tolist())
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
                "rtt_ms_mean": round(float(np.mean(infer_ms)), 1) if infer_ms else None, "error": err,
            }
            with open(args.out, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(f"EP {json.dumps(rec)}  running {n_succ}/{n_ep}", flush=True)
        env.close()
    print(f"FINAL success {n_succ}/{n_ep} = {n_succ / max(n_ep, 1):.3f}", flush=True)


if __name__ == "__main__":
    main()
