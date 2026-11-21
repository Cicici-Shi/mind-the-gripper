"""CogACT 原样基线（不改画面）。

参数逐字取自 CogACT 官方 sim_cogact/scripts/*.sh（各任务的第一组配置），用 SimplerEnv 自己的 get_args 解析，
回合循环调用 SimplerEnv 官方 run_maniskill2_eval_single_episode（未改动），策略用官方 CogACTInference（默认参数：
cfg 1.5、DDIM 10 步、adaptive ensemble，Google 窗口 2、WidowX 窗口 7，fp32）。
唯一的增加：每回合开头 torch.manual_seed(1000 + 回合序号)，让扩散初始噪声可复现；计时与显存峰值。
用法：python baseline_cogact.py <task1,task2,...（同一机器人）> <n_episodes> <out_dir>
没有 HF_TOKEN 时，把 llama2-7b-pure 的 hf_hub_path 指到 make_llama_stub.py 拼的本地目录（见该文件说明）。
"""
import os, sys, time, json
import numpy as np
import torch

R = os.path.expanduser("$PROJ_ROOT/bench2_cogact")
sys.path.insert(0, R + "/CogACT")
sys.path.insert(0, R + "/SimplerEnv")
CKPT = R + "/ckpt/CogACT-Base/checkpoints/CogACT-Base.pt"

GOOGLE = ["--robot", "google_robot_static", "--policy-setup", "google_robot", "--control-freq", "3", "--sim-freq", "513",
          "--max-episode-steps", "80", "--robot-init-rot-quat-center", "0", "0", "0", "1"]
WIDOWX = ["--policy-setup", "widowx_bridge", "--control-freq", "5", "--sim-freq", "500", "--max-episode-steps", "120",
          "--robot-init-rot-quat-center", "0", "0", "0", "1", "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
          "--obj-variation-mode", "episode", "--obj-episode-range", "0", "24"]
TASKS = {
    # cogact_pick_coke_can_visual_matching.sh，第一组：urdf_version=None，lr_switch=True（横放）
    "coke_can": GOOGLE + ["--env-name", "GraspSingleOpenedCokeCanInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
        "--rgb-overlay-path", "./ManiSkill2_real2sim/data/real_inpainting/google_coke_can_real_eval_1.png",
        "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.20", "0.20", "1",
        "--obj-init-x", "-0.35", "-0.12", "5", "--obj-init-y", "-0.02", "0.42", "5",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "0", "0", "1",
        "--additional-env-build-kwargs", "lr_switch=True", "urdf_version=None"],
    # cogact_move_near_visual_matching.sh，第一组：urdf_version=None
    "move_near": GOOGLE + ["--env-name", "MoveNearGoogleBakedTexInScene-v0", "--scene-name", "google_pick_coke_can_1_v4",
        "--rgb-overlay-path", "./ManiSkill2_real2sim/data/real_inpainting/google_move_near_real_eval_1.png",
        "--robot-init-x", "0.35", "0.35", "1", "--robot-init-y", "0.21", "0.21", "1",
        "--obj-variation-mode", "episode", "--obj-episode-range", "0", "60",
        "--robot-init-rot-rpy-range", "0", "0", "1", "0", "0", "1", "-0.09", "-0.09", "1",
        "--additional-env-build-kwargs", "urdf_version=None", "--additional-env-save-tags", "baked_except_bpb_orange"],
    # cogact_bridge.sh
    "spoon": WIDOWX + ["--robot", "widowx", "--env-name", "PutSpoonOnTableClothInScene-v0", "--scene-name", "bridge_table_1_v1",
        "--rgb-overlay-path", "ManiSkill2_real2sim/data/real_inpainting/bridge_real_eval_1.png",
        "--robot-init-x", "0.147", "0.147", "1", "--robot-init-y", "0.028", "0.028", "1"],
    "eggplant": WIDOWX + ["--robot", "widowx_sink_camera_setup", "--env-name", "PutEggplantInBasketScene-v0", "--scene-name", "bridge_table_1_v2",
        "--rgb-overlay-path", "ManiSkill2_real2sim/data/real_inpainting/bridge_sink.png",
        "--robot-init-x", "0.127", "0.127", "1", "--robot-init-y", "0.06", "0.06", "1"],
}

def parse(task, out):
    sys.argv = ["main_inference.py", "--policy-model", "cogact", "--ckpt-path", CKPT, "--logging-dir", out] + TASKS[task]
    from simpler_env.evaluation.argparse import get_args
    return get_args()

def main():
    tasks, n, out = sys.argv[1].split(","), int(sys.argv[2]), os.path.abspath(sys.argv[3])
    os.makedirs(out, exist_ok=True)
    os.chdir(R + "/SimplerEnv")  # 官方脚本在 SimplerEnv 根目录下跑，overlay 路径是相对路径
    from simpler_env.evaluation.maniskill2_evaluator import run_maniskill2_eval_single_episode
    from simpler_env.utils.env.env_builder import get_robot_control_mode
    os.environ["DISPLAY"] = ""
    if not os.environ.get("HF_TOKEN"):
        import prismatic.models.backbones.llm.llama2 as L
        L.LLAMA2_MODELS["llama2-7b-pure"]["hf_hub_path"] = R + "/llama2_stub"
        print("llama2-7b-pure hf_hub_path ->", R + "/llama2_stub", flush=True)
    from sim_cogact import CogACTInference
    args0 = parse(tasks[0], out)
    t0 = time.time()
    model = CogACTInference(saved_model_path=CKPT, policy_setup=args0.policy_setup, action_scale=args0.action_scale,
                            action_model_type="DiT-B", cfg_scale=1.5)
    print(f"LOAD sec {time.time()-t0:.1f} vram GB {torch.cuda.max_memory_allocated()/1e9:.2f}", flush=True)
    for task in tasks:
        args = parse(task, out)
        assert args.policy_setup == args0.policy_setup
        control_mode = get_robot_control_mode(args.robot, args.policy_model)
        common = dict(model=model, ckpt_path=args.ckpt_path, robot_name=args.robot, env_name=args.env_name,
                      scene_name=args.scene_name, robot_init_x=args.robot_init_xs[0], robot_init_y=args.robot_init_ys[0],
                      robot_init_quat=args.robot_init_quats[0], control_mode=control_mode,
                      additional_env_build_kwargs=args.additional_env_build_kwargs, rgb_overlay_path=args.rgb_overlay_path,
                      control_freq=args.control_freq, sim_freq=args.sim_freq, max_episode_steps=args.max_episode_steps,
                      enable_raytracing=args.enable_raytracing, additional_env_save_tags=args.additional_env_save_tags,
                      obs_camera_name=args.obs_camera_name, logging_dir=out)
        if args.obj_variation_mode == "xy":  # 与 maniskill2_evaluator 相同的顺序：x 外层、y 内层
            eps = [dict(obj_init_x=x, obj_init_y=y) for x in args.obj_init_xs for y in args.obj_init_ys]
        else:
            eps = [dict(obj_episode_id=i) for i in range(args.obj_episode_range[0], args.obj_episode_range[1])]
        eps = eps[:n]
        recs = []
        for k, e in enumerate(eps):
            torch.manual_seed(1000 + k); np.random.seed(1000 + k)
            torch.cuda.reset_peak_memory_stats()
            t = time.time()
            ok = run_maniskill2_eval_single_episode(**common, **e)
            rec = dict(task=task, env=args.env_name, k=k, **{a: float(b) for a, b in e.items()}, success=bool(ok),
                       sec=round(time.time()-t, 2), peak_vram_gb=round(torch.cuda.max_memory_allocated()/1e9, 2))
            recs.append(rec); print("EP", json.dumps(rec), flush=True)
            with open(os.path.join(out, "episodes.jsonl"), "a") as f: f.write(json.dumps(rec) + "\n")
        s = sum(r["success"] for r in recs)
        print(f"SUMMARY {task} {args.env_name} success {s} / {len(recs)} mean sec/ep {np.mean([r['sec'] for r in recs]):.2f} "
              f"peak VRAM GB {max(r['peak_vram_gb'] for r in recs):.2f} (torch allocated)", flush=True)

if __name__ == "__main__":
    main()
