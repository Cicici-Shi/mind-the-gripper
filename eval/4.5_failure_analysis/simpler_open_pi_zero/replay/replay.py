"""第 0 步回放：把 out_formal 记录里的 env_actions 原样送进环境，逐步记下真实状态。不加载策略。

用法：python replay.py --rec-root <out_formal> --out <输出根> --obs-mode {rgbd,state_dict} --list <任务清单.txt>
任务清单每行「块 格 键」。每条输出 <out>/<块>/<格>/<键>.npz，先写临时文件再改名，--skip-done 可续跑。
建环境与 reset 照 b2/eval_views.py（reset_key：seed=env_seed，Bridge/F2 另给 episode_id，reconfigure=True）。
可信性核对（逐条写进输出）：逐步 success 与记录的 success_steps 逐位相同；rgbd 模式下再核第 0 帧哈希。
"""
import argparse
import hashlib
import json
import os
import sys
import time
import traceback

import numpy as np

ENV_SRC_SHA = {  # 与 b2/eval_views.py 相同的 fork 源码哈希，改了就拒跑
    "grasp_single_in_scene.py": "c067aa46582a2ad9ced5a360b583221a9be635fab578368a0441685dced1965d",
    "move_near_in_scene.py": "49a23096ddf041457df7079e1898a56320772fa8414a296f71d11cbf6ace3832",
    "put_on_in_scene.py": "e8fa5aa31a82fa80699dc029388e81ab9c84f54c59fa881cb5825a0e21d31049",
    "base_env.py": "532cf7981ae3c13c292f3bac913e2d17baa1b5f8316d437da0c0674b07d9a77b",
}


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if hasattr(x, "p") and hasattr(x, "q"):
        return {"p": np.asarray(x.p).tolist(), "q": np.asarray(x.q).tolist()}
    return repr(x)


def pose7(p):
    return np.concatenate([np.asarray(p.p), np.asarray(p.q)]).astype(np.float64)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec-root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--obs-mode", default="state_dict", choices=["rgbd", "state_dict"])
    ap.add_argument("--list", required=True)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()

    import gymnasium as gym
    import simpler_env
    from simpler_env import ENVIRONMENT_MAP
    import mani_skill2_real2sim.envs.custom_scenes as cs
    d = os.path.dirname(cs.__file__)
    for f, want in ENV_SRC_SHA.items():
        got = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        assert got == want, f"fork 源码 {f} 与记录不符"

    todo = [ln.split() for ln in open(a.list) if ln.strip()]
    envs = {}
    for b, c, key in todo:
        odir = os.path.join(a.out, b, c)
        os.makedirs(odir, exist_ok=True)
        op = os.path.join(odir, key + ".npz")
        if a.skip_done and os.path.exists(op):
            continue
        te = time.time()
        try:
            z = np.load(os.path.join(a.rec_root, b, c, key + ".npz"))
            task = str(z["task"])
            if task not in envs:
                name, kw = ENVIRONMENT_MAP[task]
                kw = dict(kw, prepackaged_config=True)
                envs[task] = gym.make(name, obs_mode=a.obs_mode, **kw)
            env = envs[task]
            u = env.unwrapped
            reset = json.loads(str(z["reset_spec"]))
            obj = {}
            if "episode_id" in reset:
                obj["episode_id"] = int(reset["episode_id"])
            obs, info = env.reset(seed=int(z["env_seed"]), options={"obj_init_options": obj, "reconfigure": True})
            frame0_ok = None
            if a.obs_mode == "rgbd":
                cam = "3rd_view_camera" if "widowx" in u.robot_uid.lower() else "overhead_camera"
                rgb0 = np.ascontiguousarray(obs["image"][cam]["rgb"])
                frame0_ok = hashlib.sha256(rgb0.tobytes()).hexdigest() == str(z["frame0_sha256"])
            ag = u.agent
            # 任务物体：Bridge / move near 用 episode_source_obj（另记目标物），抓可乐罐用 obj
            src = getattr(u, "episode_source_obj", None) or getattr(u, "obj", None)
            tgt = getattr(u, "episode_target_obj", None)
            acts = u.get_actors()
            names = [x.name for x in acts]
            init_ok = bool(np.allclose(np.array([pose7(x.pose) for x in acts]), z["init_actor_poses"], atol=0, rtol=0)) \
                if np.array(z["init_actor_poses"]).shape == (len(acts), 7) else False
            ea = np.asarray(z["env_actions"], np.float64)

            def snap():
                pr = ag.get_proprioception()
                g = ag.gripper_closedness
                g = g() if callable(g) else g
                return dict(
                    tcp=pose7(ag.ee_pose), qpos=np.asarray(ag.robot.get_qpos(), np.float64),
                    grip_width=float(pr["eef_pos"][7]), grip_closed=float(g),
                    actors=np.array([pose7(x.pose) for x in acts]),
                    grasp_src=bool(ag.check_grasp(src)) if src is not None else False,
                    contact_src=jsonable(ag.check_contact_fingers(src)) if src is not None else None,
                )

            S = [snap()]
            succ, infos = [], []
            for t in range(len(ea)):
                obs, r, s, trunc, info = env.step(ea[t])
                succ.append(bool(s))
                infos.append(jsonable(info))
                S.append(snap())
            rec_succ = np.asarray(z["success_steps"], bool)
            succ = np.array(succ, bool)
            out = dict(
                block=b, cell=c, key=key, task=task, obs_mode=a.obs_mode, robot_uid=u.robot_uid,
                success_steps=succ, match_success=np.bool_(succ.shape == rec_succ.shape and np.array_equal(succ, rec_succ)),
                frame0_match=np.int8(-1 if frame0_ok is None else int(frame0_ok)), init_actor_match=np.bool_(init_ok),
                truncated_last=np.bool_(trunc), n_steps=np.int32(len(ea)),
                actor_names=json.dumps(names), src_name=str(src.name) if src is not None else "",
                tgt_name=str(tgt.name) if tgt is not None else "",
                robot_base=pose7(ag.robot.pose),
                tcp=np.array([x["tcp"] for x in S]), qpos=np.array([x["qpos"] for x in S]),
                grip_width=np.array([x["grip_width"] for x in S]), grip_closed=np.array([x["grip_closed"] for x in S]),
                actors=np.array([x["actors"] for x in S]),
                grasp_src=np.array([x["grasp_src"] for x in S]), contact_src=json.dumps([x["contact_src"] for x in S]),
                infos=json.dumps(infos), sec=np.float64(time.time() - te),
            )
            tmp = os.path.join(odir, f".{key}.tmp.npz")
            np.savez_compressed(tmp, **out)
            os.replace(tmp, op)
            print(f"[rp] {b} {c} {key} match={bool(out['match_success'])} f0={int(out['frame0_match'])} "
                  f"init={init_ok} {time.time() - te:.1f}s", flush=True)
        except Exception:
            print(f"[rp] ERROR {b} {c} {key}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
