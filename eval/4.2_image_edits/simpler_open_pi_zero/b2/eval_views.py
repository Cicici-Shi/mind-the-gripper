"""bench2 评测循环：一个进程跑一个（块，格）的一批键（协议 protocol_bench2.md v1）。

循环照 open-pi-zero/src/agent/eval.py 与探路的 baseline_pi0.py：每次查询推一个动作块，执行前 act_steps 步，
截断（max_episode_steps）时结束。bf16，不开 torch.compile（v1 §1 有意如此）。
与原循环的差别：
  1. 喂给策略的图换成 views.ViewMaker 的输出（本体状态、指令、物理不动）；
  2. 每次推理前按（块，场景配置，试次，查询序号）设 torch 种子；
  3. reset 用键指定的 seed（Bridge 另给 episode_id），并强制 reconfigure（每回合重建场景，初始状态与运行历史无关）；
  4. 逐步记录 success：主读数 = 最后一步（与作者评测一致），另存「任一步成功过」。
盲采集：日志不打印成败与步数。--blind-success 时 npz 里连成败字段都不写（校准键上 C0 以外的格用）。
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

B2 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2)
OPZ = os.path.expanduser("$PROJ_ROOT/bench2_simpler/open-pi-zero")

import numpy as np  # noqa: E402

import keys as K  # noqa: E402
import views as V  # noqa: E402

ENV_SRC_SHA = {  # fork（ManiSkill2_real2sim @ 91d154b）环境源码，启动时核对，改了就拒跑
    "grasp_single_in_scene.py": "c067aa46582a2ad9ced5a360b583221a9be635fab578368a0441685dced1965d",
    "move_near_in_scene.py": "49a23096ddf041457df7079e1898a56320772fa8414a296f71d11cbf6ace3832",
    "put_on_in_scene.py": "e8fa5aa31a82fa80699dc029388e81ab9c84f54c59fa881cb5825a0e21d31049",
    "base_env.py": "532cf7981ae3c13c292f3bac913e2d17baa1b5f8316d437da0c0674b07d9a77b",
}
# 本体状态探针（v1 §4 第 9 项）：在归一化后的本体状态上做的扰动
PROBE_POS_DELTA = 0.2     # 末端位置三维各加 0.2（归一化范围 [-1, 1] 的 10%）
PROBE_QUERIES = (0, 5, 10)  # 在这几次查询的缓存输入上做


def sha_file(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def check_env_src():
    import mani_skill2_real2sim.envs.custom_scenes as cs
    d = os.path.dirname(cs.__file__)
    for f, want in ENV_SRC_SHA.items():
        got = sha_file(os.path.join(d, f))
        assert got == want, f"fork 源码 {f} 与记录不符：{got}"


def reset_key(env, k):
    obj = {}
    if "episode_id" in k["reset"]:
        obj["episode_id"] = int(k["reset"]["episode_id"])
    return env.reset(seed=int(k["env_seed"]), options={"obj_init_options": obj, "reconfigure": True})


def check_reset_spec(env, info, k):
    """reset 后实际生效的 episode_id 与 urdf 版本；键里指定了的必须一致（协议 v1.1 的 F2 键）。"""
    got_ep = info.get("episode_id", None)
    got_ep = -1 if got_ep is None else int(got_ep)
    got_urdf = str(env.unwrapped.urdf_version)
    want = k["reset"]
    if "episode_id" in want:  # Bridge 与 F2 的 reset info 都带取模后的 episode_id（键核查实测）
        assert got_ep == int(want["episode_id"]), f"episode_id {got_ep} != 键 {want['episode_id']}"
    if "urdf_version" in want:
        assert got_urdf == want["urdf_version"], f"urdf {got_urdf!r} != 键 {want['urdf_version']!r}"
    return got_ep, got_urdf


def scene_state(env):
    """reset 后的机器人关节、机器人位姿、全部 actor 位姿、其他关节体关节角。"""
    u = env.unwrapped
    acts = u.get_actors()
    arts = [a for a in u._scene.get_all_articulations() if a is not u.agent.robot]
    return dict(
        init_robot_qpos=np.asarray(u.agent.robot.get_qpos(), np.float64),
        init_robot_pose=np.concatenate([u.agent.robot.pose.p, u.agent.robot.pose.q]).astype(np.float64),
        init_actor_names=json.dumps([a.name for a in acts]),
        init_actor_poses=np.array([np.concatenate([a.pose.p, a.pose.q]) for a in acts], np.float64).reshape(-1, 7),
        init_art_names=json.dumps([a.name for a in arts]),
        init_art_qpos=json.dumps([np.asarray(a.get_qpos(), np.float64).tolist() for a in arts]),
        init_urdf_version=str(u.urdf_version),
    )


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=sorted(K.BLOCKS))
    ap.add_argument("--cell", required=True, choices=V.CELLS)
    ap.add_argument("--keys", required=True, help="键清单 json（keys.py 生成）")
    ap.add_argument("--out", required=True, help="输出根目录；记录写到 <out>/<块>/<格>/<键>.npz")
    ap.add_argument("--lag", type=int, default=None, help="滞后 L（控制步）；默认按协议规则由控制频率算")
    ap.add_argument("--line-width", type=int, default=None, help="默认按协议公式由图宽算")
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--blind-success", action="store_true", help="npz 不写任何成败字段")
    ap.add_argument("--proprio-probe", action="store_true", help="在 PROBE_QUERIES 的缓存输入上做本体状态探针")
    ap.add_argument("--save-frames", default="", help="逗号分隔的查询序号，存该帧策略输入图 png（原尺寸与 224）")
    ap.add_argument("--save-video", action="store_true", help="存每回合策略输入图视频")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()

    blk = dict(K.BLOCKS[a.block])
    ckpt = os.path.expanduser(a.ckpt or blk["ckpt"])
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    for k in keys:
        assert k["task"] in blk["tasks"], k
    if a.limit:
        keys = keys[: a.limit]
    odir = os.path.join(a.out, a.block, a.cell)
    edir = os.path.join(odir, "_errors")
    os.makedirs(edir, exist_ok=True)

    def done(k):
        p = os.path.join(odir, k["key"] + ".npz")
        if not os.path.exists(p):
            return False
        try:
            with np.load(p) as z:
                return bool(z["complete"])
        except Exception:
            return False
    todo = [k for k in keys if not (a.skip_done and done(k))]
    print(f"[b2] block {a.block} cell {a.cell} keys {len(keys)} todo {len(todo)}", flush=True)
    if not todo:
        return

    import torch
    import hydra
    from omegaconf import OmegaConf
    import imageio
    import simpler_env
    check_env_src()
    cwd0 = os.getcwd()
    os.chdir(OPZ)
    sys.path.insert(0, OPZ)
    from src.model.vla.pizero import PiZeroInference
    cfg = OmegaConf.load(a.cfg or blk["cfg"])
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        smi = f"nvidia-smi failed: {e}"
    t0 = time.time()
    model = PiZeroInference(cfg, use_ddp=False)
    data = torch.load(ckpt, weights_only=True, map_location="cpu", mmap=True)  # mmap：同卡多进程时不各占一份 12 GB 内存
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}, strict=True)
    del data
    model.freeze_all_weights()
    model.to(dtype)
    model.to(dev)
    model.eval()
    print(f"[b2] model loaded {time.time() - t0:.1f}s gpu {gpu_name}", flush=True)
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    os.chdir(cwd0)
    act_steps = int(cfg.act_steps)
    envs = {}
    frames_q = {int(x) for x in a.save_frames.split(",") if x.strip() != ""}
    code_sha = {f: sha_file(os.path.join(B2, f)) for f in ("eval_views.py", "views.py", "keys.py")}

    def infer(inp, s):
        cm, vp, pp, apos = model.build_causal_mask_and_position_ids(inp["attention_mask"], dtype=dtype)
        itpm, am = model.split_full_mask_into_submasks(cm)
        x = {"input_ids": inp["input_ids"], "pixel_values": inp["pixel_values"].to(dtype),
             "image_text_proprio_mask": itpm, "action_mask": am, "vlm_position_ids": vp,
             "proprio_position_ids": pp, "action_position_ids": apos, "proprios": inp["proprios"].to(dtype)}
        x = {kk: v.to(dev) for kk, v in x.items()}
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        with torch.inference_mode():
            return model(**x)[0].float().cpu().numpy()

    for k in todo:
        te = time.time()
        try:
            if k["task"] not in envs:
                envs[k["task"]] = simpler_env.make(k["task"])
                u = envs[k["task"]].unwrapped
                print(f"[b2] env {k['task']} robot {u.robot_uid} ctrl_hz {u._control_freq} act_steps {act_steps} "
                      f"max_steps {envs[k['task']].spec.max_episode_steps}", flush=True)
            env = envs[k["task"]]
            u = env.unwrapped
            cam = V.cam_name(u.robot_uid)
            max_steps = int(env.spec.max_episode_steps)
            ctrl_hz = float(u._control_freq)
            L = a.lag or V.lag_for(act_steps, ctrl_hz)
            obs, info = reset_key(env, k)
            reset_info = json.dumps(jsonable(info))
            got_ep, got_urdf = check_reset_spec(env, info, k)
            adapter.reset()
            W = obs["image"][cam]["rgb"].shape[1]
            lw = a.line_width or V.line_width_for(W)
            vm = V.ViewMaker(env, a.cell, blk["body"], L, lw, act_steps, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
            vm.on_reset()
            if a.cell in V.HIDE_CELLS:
                vm.check_restore(obs)
                assert vm.restore_maxdiff == 0, f"隐藏后恢复与原图不同 maxdiff {vm.restore_maxdiff}"
            st0 = scene_state(env)
            # reset 后第 0 帧原图（干预之前）的哈希：门槛核对同一键五格、不同节点渲染是否逐位相同
            rgb0 = np.ascontiguousarray(obs["image"][cam]["rgb"])
            frame0_sha = hashlib.sha256(rgb0.tobytes()).hexdigest()
            instr0 = env.get_language_instruction()
            instr = instr0
            seeds, raw_actions, env_actions, qstats, video, succ_steps, probe = [], [], [], [], [], [], []
            step, q, truncated = 0, 0, False
            while True:
                img, stt = vm.make(obs, step)
                pobs = dict(obs)
                pobs["image"] = {cam: {"rgb": img}}
                s = K.noise_seed(a.block, k["cfg_id"], k["trial"], q)
                seeds.append(s)
                inp = adapter.preprocess(env, pobs, instr)
                ra = infer(inp, s)
                raw_actions.append(ra)
                if a.proprio_probe and q in PROBE_QUERIES:
                    pr = {"q": q, "proprio": inp["proprios"][0, 0].tolist()}
                    for name in ("same", "gripper_flip", "pos_shift"):
                        p2 = inp["proprios"].clone()
                        if name == "gripper_flip":
                            g = float(p2[0, 0, -1])
                            p2[0, 0, -1] = -1.0 if g > 0 else 1.0
                        elif name == "pos_shift":
                            p2[0, 0, :3] = (p2[0, 0, :3] + PROBE_POS_DELTA).clamp(-1, 1)
                        ra2 = infer(dict(inp, proprios=p2), s)
                        dd = np.abs(ra2 - ra)
                        pr[name] = {"bit_equal": bool(np.array_equal(ra2, ra)), "max_abs": float(dd.max()),
                                    "mean_abs": float(dd.mean()), "gripper_dim_max_abs": float(dd[:, -1].max())}
                    probe.append(pr)
                if q in frames_q:
                    tag = os.path.join(odir, "frames", f"{k['key']}_q{q:03d}")
                    os.makedirs(os.path.dirname(tag), exist_ok=True)
                    imageio.imwrite(tag + "_full.png", img)
                    imageio.imwrite(tag + "_224.png", V.to_model(img))
                if a.save_video:
                    video.append(img)
                qstats.append(stt)
                for ea in adapter.postprocess(ra)[:act_steps]:
                    obs, r, success, truncated, info = env.step(ea)
                    succ_steps.append(bool(success))
                    env_actions.append(np.asarray(ea, np.float64))
                    step += 1
                    if truncated:
                        break
                new = env.get_language_instruction()
                if new != instr:
                    instr = new
                q += 1
                if truncated:
                    break
            rec = dict(
                complete=np.bool_(True), block=a.block, cell=a.cell, key=k["key"], cfg_id=k["cfg_id"],
                cluster=k["cluster"], task=k["task"], trial=np.int32(k["trial"]), env_seed=np.int64(k["env_seed"]),
                reset_spec=json.dumps(k["reset"]), robot_uid=u.robot_uid,
                actual_episode_id=np.int32(got_ep), actual_urdf_version=got_urdf,
                n_queries=np.int32(q), seeds=np.array(seeds, np.uint64), raw_actions=np.array(raw_actions, np.float32),
                env_actions=np.array(env_actions, np.float64), instruction=instr0, instruction_final=instr,
                qstats=json.dumps(qstats), proprio_probe=json.dumps(probe), gpu_name=gpu_name, nvidia_smi=smi,
                host=socket.gethostname(), lag_steps=np.int32(L), line_width=np.int32(lw),
                query_every=np.int32(act_steps), max_episode_steps=np.int32(max_steps), control_freq=np.float64(ctrl_hz),
                img_hw=np.array(obs["image"][cam]["rgb"].shape[:2], np.int32),
                g_links=json.dumps([l.name for l in vm.g_links]), g_ids=vm.gids, p_names=json.dumps(vm.p_names),
                n_hidden_visual_bodies=np.int32(vm.n_vb),
                restore_maxdiff=np.int32(-1 if vm.restore_maxdiff is None else vm.restore_maxdiff),
                reset_info=reset_info, rgb_overlay_path=str(u.rgb_overlay_path), code_sha=json.dumps(code_sha),
                frame0_sha256=frame0_sha, frame0_shape=np.array(rgb0.shape, np.int32),
                vk_icd_filenames=os.environ.get("VK_ICD_FILENAMES", ""),
                ckpt=ckpt, dtype="bf16", torch_compile=np.bool_(False), blind_success=np.bool_(a.blind_success),
                sec=np.float64(time.time() - te), **st0)
            if not a.blind_success:
                # 成败字段：final = 最后一步（主读数），ever = 任一步，steps = 每步
                rec.update(success=np.bool_(succ_steps[-1]), success_ever=np.bool_(any(succ_steps)),
                           success_steps=np.array(succ_steps, np.bool_), steps=np.int32(step))
            tmp = os.path.join(odir, f".{k['key']}.tmp.npz")
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, os.path.join(odir, k["key"] + ".npz"))
            if a.save_video and video:
                vt = os.path.join(odir, "video", f".{k['key']}.tmp.mp4")
                os.makedirs(os.path.dirname(vt), exist_ok=True)
                imageio.mimsave(vt, video, fps=5)
                os.replace(vt, os.path.join(odir, "video", k["key"] + ".mp4"))
            ep = os.path.join(edir, k["key"] + ".txt")
            if os.path.exists(ep):
                os.replace(ep, ep + ".resolved")
            print(f"[b2] done {k['key']} {time.time() - te:.1f}s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            with open(os.path.join(edir, k["key"] + ".txt"), "w") as f:
                f.write(tb)
            print(f"[b2] ERROR {k['key']} (see _errors)", flush=True)
            print(tb, file=sys.stderr, flush=True)
    print(f"[b2] finished peak_vram_gb {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}", flush=True)


if __name__ == "__main__":
    main()
