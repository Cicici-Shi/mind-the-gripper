"""CogACT 第二阶段既有回合的回放：把 out_formal 记录里的 env_actions 原样送进官方环境，逐步记下真实状态。不加载策略。

用法：python replay_cogact.py --rec-root <b2c/out_formal> --out <输出根> --obs-mode {rgbd,state_dict} --list <清单.txt>
清单每行「块 格 键」。每条输出 <out>/<块>/<格>/<键>.npz，先写临时文件再改名，--skip-done 可续跑。
由第一阶段 bench2_simpler/step0_replay/replay.py 改写。差别：环境用官方 simpler_env.make（与 b2c/eval_views.py 的
make_env、reset_key 相同），官方 agent 没有 ee_pose 属性，这里按 ee_link 名取末端位姿，并把机器人全部连杆的位姿都存下来。
可信性核对（逐条写进输出，不在这里判）：逐步 success 与记录的 success_steps 逐位相同；初始 actor 名与位姿、机器人初始
qpos 与记录逐位相同；rgbd 模式下再核第 0 帧哈希。
本脚本只读 out_formal，不写 b2c 目录。
"""
import argparse
import hashlib
import json
import os
import time
import traceback
import warnings

import numpy as np

ENV_SRC_SHA = {  # 与 b2c/eval_views.py 相同的官方环境源码哈希，不同就拒跑
    "grasp_single_in_scene.py": "c067aa46582a2ad9ced5a360b583221a9be635fab578368a0441685dced1965d",
    "move_near_in_scene.py": "32f67ef26591b7b652ade2c583ec68645ab7a6de5e01739b584dec3c5a953d29",
    "put_on_in_scene.py": "e8fa5aa31a82fa80699dc029388e81ab9c84f54c59fa881cb5825a0e21d31049",
    "base_env.py": "532cf7981ae3c13c292f3bac913e2d17baa1b5f8316d437da0c0674b07d9a77b",
}
EE_LINK = {"widowx": "ee_gripper_link", "google": "link_gripper_tcp"}


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

    import simpler_env
    import mani_skill2_real2sim.envs.custom_scenes as cs
    d = os.path.dirname(cs.__file__)
    for f, want in ENV_SRC_SHA.items():
        got = hashlib.sha256(open(os.path.join(d, f), "rb").read()).hexdigest()
        assert got == want, f"官方环境源码 {f} 与记录不符：{got}"

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
            mes = int(z["max_episode_steps"])
            if (task, mes) not in envs:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    envs[(task, mes)] = simpler_env.make(task, obs_mode=a.obs_mode, max_episode_steps=mes)
            env = envs[(task, mes)]
            u = env.unwrapped
            freq_ok = bool(float(u._control_freq) == float(z["control_freq"]) and float(u._sim_freq) == float(z["sim_freq"]))
            reset = json.loads(str(z["reset_spec"]))
            obj = {}
            if "episode_id" in reset:
                obj["episode_id"] = int(reset["episode_id"])
            obs, info = env.reset(seed=int(z["env_seed"]), options={"obj_init_options": obj, "reconfigure": True})
            got_ep = info.get("episode_id", None)
            ep_ok = bool(int(-1 if got_ep is None else got_ep) == int(z["actual_episode_id"]))
            frame0_ok = None
            if a.obs_mode == "rgbd":
                cam = "3rd_view_camera" if "widowx" in u.robot_uid.lower() else "overhead_camera"
                rgb0 = np.ascontiguousarray(obs["image"][cam]["rgb"])
                frame0_ok = hashlib.sha256(rgb0.tobytes()).hexdigest() == str(z["frame0_sha256"])
            ag = u.agent
            links = ag.robot.get_links()
            link_names = [x.name for x in links]
            ee_name = EE_LINK["widowx" if "widowx" in u.robot_uid.lower() else "google"]
            ee = links[link_names.index(ee_name)]
            src = getattr(u, "episode_source_obj", None) or getattr(u, "obj", None)
            tgt = getattr(u, "episode_target_obj", None)
            acts = u.get_actors()
            names = [x.name for x in acts]
            rec_names = json.loads(str(z["init_actor_names"]))
            init_ok = bool(names == rec_names and np.array_equal(
                np.array([pose7(x.pose) for x in acts]).reshape(-1, 7), np.asarray(z["init_actor_poses"])))
            qpos0_ok = bool(np.array_equal(np.asarray(ag.robot.get_qpos(), np.float64), np.asarray(z["init_robot_qpos"])))
            ea = np.asarray(z["env_actions"], np.float64)

            def snap():
                return dict(
                    tcp=pose7(ee.get_pose()), qpos=np.asarray(ag.robot.get_qpos(), np.float64),
                    grip_closed=float(ag.get_gripper_closedness()),
                    actors=np.array([pose7(x.pose) for x in acts]),
                    links=np.array([pose7(x.get_pose()) for x in links]),
                    grasp_src=bool(ag.check_grasp(src)) if src is not None else False,
                    contact_src=jsonable(ag.check_contact_fingers(src)) if src is not None else None,
                )

            S = [snap()]
            succ, infos = [], []
            trunc = False
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
                init_qpos_match=np.bool_(qpos0_ok), episode_id_match=np.bool_(ep_ok), freq_match=np.bool_(freq_ok),
                truncated_last=np.bool_(trunc), n_steps=np.int32(len(ea)),
                actor_names=json.dumps(names), link_names=json.dumps(link_names), ee_link=ee_name,
                src_name=str(src.name) if src is not None else "", tgt_name=str(tgt.name) if tgt is not None else "",
                robot_base=pose7(ag.robot.pose),
                tcp=np.array([x["tcp"] for x in S]), qpos=np.array([x["qpos"] for x in S]),
                grip_closed=np.array([x["grip_closed"] for x in S]),
                actors=np.array([x["actors"] for x in S]), links=np.array([x["links"] for x in S]),
                grasp_src=np.array([x["grasp_src"] for x in S]), contact_src=json.dumps([x["contact_src"] for x in S]),
                env_actions=ea, rec_success=np.bool_(bool(z["success"])),
                infos=json.dumps(infos), sec=np.float64(time.time() - te),
            )
            tmp = os.path.join(odir, f".{key}.tmp.npz")
            np.savez_compressed(tmp, **out)
            os.replace(tmp, op)
            print(f"[rp] {b} {c} {key} match={bool(out['match_success'])} f0={int(out['frame0_match'])} "
                  f"init={init_ok} qpos0={qpos0_ok} ep={ep_ok} freq={freq_ok} steps={len(ea)} sec={time.time() - te:.1f}", flush=True)
        except Exception:
            print(f"[rp] ERROR {b} {c} {key}\n{traceback.format_exc()}", flush=True)


if __name__ == "__main__":
    main()
