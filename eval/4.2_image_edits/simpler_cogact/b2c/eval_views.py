"""bench2 第二阶段评测循环：CogACT 在五种画面下（补充协议 protocol_bench2_phase2_cogact.md；没写到的照 protocol_bench2.md v1.3）。

由第一阶段 b2/eval_views.py 改写。第一阶段是 open-pi-zero（每 4 / 2 步查询一次、读本体状态）；这里换成 CogACT 官方的
SimplerEnv 评测流程（$PROJ_ROOT/bench2_cogact/SimplerEnv/simpler_env/evaluation/maniskill2_evaluator.py 第 95-135 行）：
  * 策略：CogACT 官方 sim_cogact.CogACTInference，构造参数与 CogACT README 第 183-193 行（官方 main_inference 分支）相同：
    CogACT-Base、DiT-B、cfg_scale 1.5、action_scale 1.0；其余取类的默认值（DDIM 10 步、fp32、自适应集成 alpha 0.1，
    集成窗口 WidowX 7 / Google 2，Google 夹爪 sticky 10 次）。每回合开头 model.reset(指令)（evaluator 第 104 行）。
  * 每个控制步查询一次：image -> model.step(image, 指令) -> env.step(concat(world_vector, rot_axangle, gripper))，
    截断（步数上限）时结束；CogACT 从不预测终止（cogact_policy.py 第 196 行）。成功 = 最后一步的 done（evaluator 第 126 行）。
  * 环境：官方 SimplerEnv 的 simpler_env.make(任务名)（prepackaged visual matching 配置，与 CogACT 脚本的控制频率、
    仿真频率、机器人初始位姿、背景叠图相同，启动时断言），步数上限按 CogACT 脚本（keys.BLOCKS）。
    官方 make 里 obs_mode 写成了元组（simpler_env/__init__.py 第 79 行多了逗号），这里显式传 obs_mode="rgbd"。
  * 与官方循环的差别：
    1. 喂给策略的图换成 views.ViewMaker 的输出（指令、物理不动；CogACT 不读本体状态）；
    2. 每次查询前按（块，场景配置，试次，查询序号）设 torch 全局种子（CogACT 扩散头的初始噪声来自全局随机数，
       cogactvla.py 第 332 行）。公式与第一阶段相同（keys.noise_seed），与格无关；查询序号 = 控制步号；
    3. reset 用键指定的 seed（Bridge、F2 另给 episode_id），并强制 reconfigure；
    4. 逐步记录 success：主读数 = 最后一步，另存「任一步成功过」。
  * 分词器：LLM 骨干 llama2-7b-pure 只读 Llama-2 的 config 与分词器（权重全部来自 CogACT 检查点），用探路时拼的本地目录
    $PROJ_ROOT/bench2_cogact/llama2_stub（补充协议 §3 第 8 条），记录里存它的文件哈希。
盲采集：日志不打印成败与步数。--blind-success 时 npz 里连成败字段都不写。

一个进程只跑一个块（策略按身体只载入一次，约 3 到 4 分钟），可以跑这一块的多个（格，键）：
  --cells C0,C3 --keys k.json   跑 键 x 格（键在外层）；或 --plan plan.json（[{"cell":..., "key": 键 dict}, ...]，驱动脚本用）。
记录：<out>/<块>/<格>/<键>.npz；出错：<out>/<块>/<格>/_errors/<键>.txt。
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

B2C = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B2C)
ROOT = os.path.expanduser("$PROJ_ROOT/bench2_cogact")
COGACT = os.path.join(ROOT, "CogACT")
SIMPLER = os.path.join(ROOT, "SimplerEnv")
LLAMA_STUB = os.path.join(ROOT, "llama2_stub")

import numpy as np  # noqa: E402

import keys as K  # noqa: E402
import views as V  # noqa: E402

# 官方 ManiSkill2_real2sim（ef7a4d4）环境源码，启动时核对，改了就拒跑。move_near_in_scene.py 与第一阶段的 fork 不同
# （fork 在第 542 行多一行 episode_id 取模 60；本协议的 F2 键 episode_id 都在 0..59，不受影响）。
ENV_SRC_SHA = {
    "grasp_single_in_scene.py": "c067aa46582a2ad9ced5a360b583221a9be635fab578368a0441685dced1965d",
    "move_near_in_scene.py": "32f67ef26591b7b652ade2c583ec68645ab7a6de5e01739b584dec3c5a953d29",
    "put_on_in_scene.py": "e8fa5aa31a82fa80699dc029388e81ab9c84f54c59fa881cb5825a0e21d31049",
    "base_env.py": "532cf7981ae3c13c292f3bac913e2d17baa1b5f8316d437da0c0674b07d9a77b",
}
# 记进每条记录的上游文件（哈希由门槛与冻结清单比对）：CogACT 推理代码、prismatic 的视觉与 LLM 骨干、官方评测与环境入口、分词器
UPSTREAM_FILES = {
    "CogACT/sim_cogact/cogact_policy.py": os.path.join(COGACT, "sim_cogact/cogact_policy.py"),
    "CogACT/sim_cogact/adaptive_ensemble.py": os.path.join(COGACT, "sim_cogact/adaptive_ensemble.py"),
    "CogACT/vla/cogactvla.py": os.path.join(COGACT, "vla/cogactvla.py"),
    "CogACT/vla/load.py": os.path.join(COGACT, "vla/load.py"),
    "CogACT/action_model/action_model.py": os.path.join(COGACT, "action_model/action_model.py"),
    "CogACT/action_model/gaussian_diffusion.py": os.path.join(COGACT, "action_model/gaussian_diffusion.py"),
    "CogACT/action_model/models.py": os.path.join(COGACT, "action_model/models.py"),
    "CogACT/action_model/respace.py": os.path.join(COGACT, "action_model/respace.py"),
    "SimplerEnv/simpler_env/__init__.py": os.path.join(SIMPLER, "simpler_env/__init__.py"),
    "SimplerEnv/simpler_env/evaluation/maniskill2_evaluator.py": os.path.join(SIMPLER, "simpler_env/evaluation/maniskill2_evaluator.py"),
    "llama2_stub/tokenizer.model": os.path.join(LLAMA_STUB, "tokenizer.model"),
    "llama2_stub/tokenizer.json": os.path.join(LLAMA_STUB, "tokenizer.json"),
    "llama2_stub/tokenizer_config.json": os.path.join(LLAMA_STUB, "tokenizer_config.json"),
    "llama2_stub/special_tokens_map.json": os.path.join(LLAMA_STUB, "special_tokens_map.json"),
    "llama2_stub/added_tokens.json": os.path.join(LLAMA_STUB, "added_tokens.json"),
    "llama2_stub/config.json": os.path.join(LLAMA_STUB, "config.json"),
}
# CogACTInference 的构造参数（README 第 183-193 行）；其余用类默认值，记录里存实际生效的值，门槛与冻结清单比对
POLICY_KW = dict(action_model_type="DiT-B", cfg_scale=1.5, action_scale=1.0)
CODE_FILES_IN_RECORD = ("eval_views.py", "views.py", "keys.py")


def sha_file(p):
    try:
        return hashlib.sha256(open(p, "rb").read()).hexdigest()
    except OSError:
        return None


def upstream_sha():
    import prismatic
    pr = os.path.dirname(prismatic.__file__)
    d = {k: sha_file(p) for k, p in UPSTREAM_FILES.items()}
    for rel in ("models/backbones/vision/dinosiglip_vit.py", "models/backbones/llm/llama2.py",
                "models/backbones/llm/base_llm.py"):
        d["prismatic/" + rel] = sha_file(os.path.join(pr, rel))
    return d


def check_env_src():
    import mani_skill2_real2sim.envs.custom_scenes as cs
    d = os.path.dirname(cs.__file__)
    assert os.path.realpath(d).startswith(os.path.realpath(SIMPLER)), f"环境不是官方 SimplerEnv 副本：{d}"
    for f, want in ENV_SRC_SHA.items():
        got = sha_file(os.path.join(d, f))
        assert got == want, f"官方环境源码 {f} 与记录不符：{got}"


def make_env(task, blk):
    """官方 simpler_env.make（prepackaged visual matching），步数上限按 CogACT 脚本；断言控制、仿真频率与 CogACT 脚本相同。"""
    import simpler_env
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = simpler_env.make(task, obs_mode="rgbd", max_episode_steps=int(blk["max_episode_steps"]))
    u = env.unwrapped
    # obs_mode="rgbd" 在 ManiSkill2 内部记为 "image"，再由包装器转出 rgb / depth / Segmentation，所以这里不比 obs_mode
    got = dict(control_freq=float(u._control_freq), sim_freq=float(u._sim_freq), max_episode_steps=episode_limit(env))
    want = dict(control_freq=float(blk["control_freq"]), sim_freq=float(blk["sim_freq"]),
                max_episode_steps=int(blk["max_episode_steps"]))
    assert got == want, f"{task} 环境设置 {got} 与 CogACT 脚本 {want} 不符"
    return env


def episode_limit(env):
    """TimeLimit 包装器里实际生效的步数上限。"""
    e = env
    while e is not None:
        if hasattr(e, "_max_episode_steps"):
            return int(e._max_episode_steps)
        e = getattr(e, "env", None)
    raise AssertionError("找不到 TimeLimit")


def reset_key(env, k):
    obj = {}
    if "episode_id" in k["reset"]:
        obj["episode_id"] = int(k["reset"]["episode_id"])
    return env.reset(seed=int(k["env_seed"]), options={"obj_init_options": obj, "reconfigure": True})


def check_reset_spec(env, info, k):
    """reset 后实际生效的 episode_id 与 urdf 版本；键里指定了的必须一致（第一阶段协议 v1.1 的 F2 键）。"""
    got_ep = info.get("episode_id", None)
    got_ep = -1 if got_ep is None else int(got_ep)
    got_urdf = str(env.unwrapped.urdf_version)
    want = k["reset"]
    if "episode_id" in want:
        assert got_ep == int(want["episode_id"]), f"episode_id {got_ep} != 键 {want['episode_id']}"
    if "urdf_version" in want:
        assert got_urdf == want["urdf_version"], f"urdf {got_urdf!r} != 键 {want['urdf_version']!r}"
    return got_ep, got_urdf


def scene_state(env):
    """reset 后的机器人关节、机器人位姿、全部 actor 位姿、其他关节体关节角（与第一阶段逐字相同）。"""
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


def load_policy(blk):
    """载入 CogACT（官方 CogACTInference），LLM 骨干的 hf_hub_path 指到本地分词器目录。返回 (model, 实际配置 dict)。"""
    import torch
    sys.path.insert(0, COGACT)
    import prismatic.models.backbones.llm.llama2 as LL
    LL.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = LLAMA_STUB
    from sim_cogact import CogACTInference
    ckpt = os.path.expanduser(K.CKPT_COGACT)
    model = CogACTInference(saved_model_path=ckpt, policy_setup=blk["policy_setup"], **POLICY_KW)
    cfg = dict(policy_setup=model.policy_setup, unnorm_key=model.unnorm_key, cfg_scale=float(model.cfg_scale),
               use_ddim=bool(model.use_ddim), num_ddim_steps=int(model.num_ddim_steps),
               action_ensemble=bool(model.action_ensemble), action_ensemble_horizon=int(model.action_ensemble_horizon),
               adaptive_ensemble_alpha=float(model.adaptive_ensemble_alpha),
               sticky_gripper_num_repeat=int(model.sticky_gripper_num_repeat), action_scale=float(model.action_scale),
               horizon=int(model.horizon), future_action_window_size=int(model.vla.future_action_window_size),
               vlm_dtype=str(next(model.vla.vlm.parameters()).dtype),
               action_model_dtype=str(next(model.vla.action_model.parameters()).dtype),
               llm_hf_hub_path=LL.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"],
               tokenizer_class=type(model.vla.vlm.llm_backbone.tokenizer).__name__,
               ckpt=ckpt, ckpt_size=int(os.path.getsize(ckpt)), ckpt_mtime=int(os.path.getmtime(ckpt)),
               torch_version=torch.__version__)
    return model, cfg


def check_to_model(model):
    """views.to_model_all 必须与模型真实的图像变换（ToTensor 之前的几步）逐位相同；返回核对描述，不同就抛错。"""
    from torchvision.transforms import ToTensor
    tf = model.vla.vlm.vision_backbone.image_transform
    rng = np.random.RandomState(0)
    out = {}
    for shape in ((480, 640, 3), (512, 640, 3)):
        img = rng.randint(0, 256, size=shape, dtype=np.uint8)
        mine = V.to_model_all(img)
        from PIL import Image
        for name, t in (("dino", tf.dino_image_transform), ("siglip", tf.siglip_image_transform)):
            x = Image.fromarray(img)
            steps = []
            for s in t.transforms:
                if isinstance(s, ToTensor):
                    break
                x = s(x)
                steps.append(repr(s))
            ref = np.asarray(x)
            assert ref.shape == mine[name].shape and np.array_equal(ref, mine[name]), \
                f"to_model 与模型的 {name} 变换不同：{steps}"
            out[f"{name}_{shape[0]}x{shape[1]}"] = steps
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=sorted(K.BLOCKS))
    ap.add_argument("--cells", default=None, help="逗号分隔的格；与 --keys 一起用")
    ap.add_argument("--keys", default=None, help="键清单 json（keys.py 生成）")
    ap.add_argument("--plan", default=None, help="驱动脚本的切片：[{cell, key}]，与 --cells/--keys 二选一")
    ap.add_argument("--out", required=True, help="输出根目录；记录写到 <out>/<块>/<格>/<键>.npz")
    ap.add_argument("--lag", type=int, default=None, help="滞后 L（控制步）；默认 keys.BLOCKS 里的值（第一阶段的步数）")
    ap.add_argument("--line-width", type=int, default=None, help="默认按协议公式由图宽算")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--blind-success", action="store_true", help="npz 不写任何成败字段")
    ap.add_argument("--save-frames", default="", help="逗号分隔的查询序号，存该帧策略输入图 png（原尺寸与 224）")
    ap.add_argument("--frames-dir", default=None, help="存帧的目录（默认 <out>/<块>/<格>/frames）")
    a = ap.parse_args()

    blk = dict(K.BLOCKS[a.block])
    if a.plan:
        plan = json.load(open(a.plan))
    else:
        assert a.cells and a.keys, "--cells 与 --keys 要一起给"
        ks = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
        plan = [{"cell": c, "key": k} for k in ks for c in a.cells.split(",")]
    for p in plan:
        assert p["cell"] in V.CELLS, p["cell"]
        assert p["key"]["block"] == a.block and p["key"]["task"] in blk["tasks"], p["key"]

    def rec_path(c, k):
        return os.path.join(a.out, a.block, c, k["key"] + ".npz")

    def done(c, k):
        p = rec_path(c, k)
        if not os.path.exists(p):
            return False
        try:
            with np.load(p) as z:
                return bool(z["complete"])
        except Exception:
            return False
    todo = [p for p in plan if not (a.skip_done and done(p["cell"], p["key"]))]
    print(f"[b2c] block {a.block} plan {len(plan)} todo {len(todo)}", flush=True)
    if not todo:
        return

    import torch
    import imageio
    check_env_src()
    gpu_name = torch.cuda.get_device_name(0)
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        smi = f"nvidia-smi failed: {e}"
    t0 = time.time()
    model, pcfg = load_policy(blk)
    to_model_check = check_to_model(model)
    print(f"[b2c] model loaded {time.time() - t0:.1f}s gpu {gpu_name} policy {pcfg['policy_setup']} "
          f"unnorm {pcfg['unnorm_key']} vlm {pcfg['vlm_dtype']}", flush=True)
    # 每次查询的扩散输出（16 x 7 动作块，反归一化后）：包一层只做记录，不改输入输出
    last_chunk = {}
    _orig_predict = model.vla.predict_action

    def _recording_predict(*args, **kw):
        out = _orig_predict(*args, **kw)
        last_chunk["actions"] = np.array(out[0], np.float32)
        return out
    model.vla.predict_action = _recording_predict

    envs = {}
    frames_q = {int(x) for x in a.save_frames.split(",") if x.strip() != ""}
    code_sha = {f: sha_file(os.path.join(B2C, f)) for f in CODE_FILES_IN_RECORD}
    ups = upstream_sha()
    L = int(a.lag or blk["L"])
    qe = K.QUERY_EVERY

    for p in todo:
        c, k = p["cell"], p["key"]
        odir = os.path.join(a.out, a.block, c)
        edir = os.path.join(odir, "_errors")
        os.makedirs(edir, exist_ok=True)
        te = time.time()
        try:
            if k["task"] not in envs:
                envs[k["task"]] = make_env(k["task"], blk)
                u = envs[k["task"]].unwrapped
                print(f"[b2c] env {k['task']} robot {u.robot_uid} ctrl_hz {u._control_freq} sim_hz {u._sim_freq} "
                      f"max_steps {episode_limit(envs[k['task']])} overlay {os.path.basename(str(u.rgb_overlay_path))}",
                      flush=True)
            env = envs[k["task"]]
            u = env.unwrapped
            cam = V.cam_name(u.robot_uid)
            max_steps = episode_limit(env)
            ctrl_hz = float(u._control_freq)
            obs, info = reset_key(env, k)
            reset_info = json.dumps(jsonable(info))
            got_ep, got_urdf = check_reset_spec(env, info, k)
            W = obs["image"][cam]["rgb"].shape[1]
            lw = a.line_width or V.line_width_for(W)
            vm = V.ViewMaker(env, c, blk["body"], L, lw, qe, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
            vm.on_reset()
            if c in V.HIDE_CELLS:
                vm.check_restore(obs)
                assert vm.restore_maxdiff == 0, f"隐藏后恢复与原图不同 maxdiff {vm.restore_maxdiff}"
            st0 = scene_state(env)
            rgb0 = np.ascontiguousarray(obs["image"][cam]["rgb"])
            frame0_sha = hashlib.sha256(rgb0.tobytes()).hexdigest()
            instr0 = env.get_language_instruction()
            instr = instr0
            model.reset(instr)
            seeds, raw_actions, chunks, env_actions, qstats, succ_steps = [], [], [], [], [], []
            step, truncated = 0, False
            while not truncated:
                img, stt = vm.make(obs, step)
                s = K.noise_seed(a.block, k["cfg_id"], k["trial"], step)  # 查询序号 = 控制步号
                seeds.append(s)
                torch.manual_seed(s)
                torch.cuda.manual_seed_all(s)
                last_chunk.clear()
                raw, act = model.step(img, instr)
                assert "actions" in last_chunk, "这一步没有调用扩散头"
                chunks.append(last_chunk["actions"])
                raw_actions.append(np.concatenate([raw["world_vector"], raw["rotation_delta"], raw["open_gripper"]]))
                ea = np.concatenate([act["world_vector"], act["rot_axangle"], np.asarray(act["gripper"]).reshape(-1)])
                assert ea.shape == (7,), ea.shape
                if step in frames_q:
                    fdir = a.frames_dir or os.path.join(odir, "frames")
                    os.makedirs(fdir, exist_ok=True)
                    tag = os.path.join(fdir, f"{a.block}_{c}_{k['key']}_q{step:03d}")
                    imageio.imwrite(tag + "_full.png", img)
                    imageio.imwrite(tag + "_224.png", V.to_model(img))
                qstats.append(stt)
                obs, reward, success, truncated, info = env.step(ea)
                succ_steps.append(bool(success))
                env_actions.append(np.asarray(ea, np.float64))
                step += 1
                new = env.get_language_instruction()
                if new != instr:
                    instr = new
            rec = dict(
                complete=np.bool_(True), block=a.block, cell=c, key=k["key"], cfg_id=k["cfg_id"],
                cluster=k["cluster"], task=k["task"], trial=np.int32(k["trial"]), env_seed=np.int64(k["env_seed"]),
                reset_spec=json.dumps(k["reset"]), robot_uid=u.robot_uid,
                actual_episode_id=np.int32(got_ep), actual_urdf_version=got_urdf,
                n_queries=np.int32(step), seeds=np.array(seeds, np.uint64),
                raw_actions=np.array(raw_actions, np.float32), pred_chunks=np.array(chunks, np.float32),
                env_actions=np.array(env_actions, np.float64), instruction=instr0, instruction_final=instr,
                qstats=json.dumps(qstats), gpu_name=gpu_name, nvidia_smi=smi,
                host=socket.gethostname(), lag_steps=np.int32(L), line_width=np.int32(lw),
                query_every=np.int32(qe), max_episode_steps=np.int32(max_steps), control_freq=np.float64(ctrl_hz),
                sim_freq=np.float64(u._sim_freq),
                img_hw=np.array(obs["image"][cam]["rgb"].shape[:2], np.int32),
                g_links=json.dumps([l.name for l in vm.g_links]), g_ids=vm.gids, p_names=json.dumps(vm.p_names),
                n_hidden_visual_bodies=np.int32(vm.n_vb),
                restore_maxdiff=np.int32(-1 if vm.restore_maxdiff is None else vm.restore_maxdiff),
                reset_info=reset_info, rgb_overlay_path=str(u.rgb_overlay_path), code_sha=json.dumps(code_sha),
                upstream_sha=json.dumps(ups, sort_keys=True), policy_cfg=json.dumps(pcfg, sort_keys=True),
                to_model_check=json.dumps(to_model_check),
                frame0_sha256=frame0_sha, frame0_shape=np.array(rgb0.shape, np.int32),
                vk_icd_filenames=os.environ.get("VK_ICD_FILENAMES", ""),
                vk_driver_files=os.environ.get("VK_DRIVER_FILES", ""),
                blind_success=np.bool_(a.blind_success), sec=np.float64(time.time() - te), **st0)
            if not a.blind_success:
                rec.update(success=np.bool_(succ_steps[-1]), success_ever=np.bool_(any(succ_steps)),
                           success_steps=np.array(succ_steps, np.bool_), steps=np.int32(step))
            tmp = os.path.join(odir, f".{k['key']}.tmp.npz")
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, rec_path(c, k))
            ep = os.path.join(edir, k["key"] + ".txt")
            if os.path.exists(ep):
                os.replace(ep, ep + ".resolved")
            print(f"[b2c] done {c} {k['key']} {time.time() - te:.1f}s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            with open(os.path.join(edir, k["key"] + ".txt"), "w") as f:
                f.write(tb)
            print(f"[b2c] ERROR {c} {k['key']} (see _errors)", flush=True)
            print(tb, file=sys.stderr, flush=True)
    print(f"[b2c] finished peak_vram_gb {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}", flush=True)


if __name__ == "__main__":
    main()
