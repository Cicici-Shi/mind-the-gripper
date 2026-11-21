"""问题 4：在 CogACT 用的官方 SimplerEnv main（eddb569）上，用 CogACT 评测脚本的环境参数建环境，
照 bench2_simpler/gripper_probe.py 的做法验：按连杆隐藏夹爪、取 actor 级分割掩膜、掩膜外零变化、恢复后与原图一致。
用法：python gripper_probe_cogact.py <out_dir> [task1,task2]（默认 spoon,coke_can）"""
import os, sys, re, traceback
import numpy as np, cv2, imageio
R = os.path.expanduser("$PROJ_ROOT/bench2_cogact")
sys.path.insert(0, R + "/SimplerEnv"); sys.path.insert(0, R)
out = os.path.abspath(sys.argv[1]); os.makedirs(out, exist_ok=True)
os.chdir(R + "/SimplerEnv")
from baseline_cogact import TASKS, CKPT
from simpler_env.evaluation.argparse import get_args
from simpler_env.utils.env.env_builder import build_maniskill2_env, get_robot_control_mode
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
PAT = re.compile(r"finger|gripper", re.I)

def raw_rgb(o, cam):
    return (np.clip(o["image"][cam]["Color"][..., :3], 0, 1) * 255).astype(np.uint8)

for task in (sys.argv[2].split(",") if len(sys.argv) > 2 else ["spoon", "coke_can"]):
    try:
        sys.argv = ["x", "--policy-model", "cogact", "--ckpt-path", CKPT] + TASKS[task]
        a = get_args()
        env = build_maniskill2_env(a.env_name, **(a.additional_env_build_kwargs or {}), obs_mode="rgbd", robot=a.robot,
                                   sim_freq=a.sim_freq, control_mode=get_robot_control_mode(a.robot, "cogact"),
                                   control_freq=a.control_freq, max_episode_steps=a.max_episode_steps,
                                   scene_name=a.scene_name, camera_cfgs={"add_segmentation": True},
                                   rgb_overlay_path=a.rgb_overlay_path)
        opts = {"robot_init_options": {"init_xy": np.array([a.robot_init_xs[0], a.robot_init_ys[0]]), "init_rot_quat": a.robot_init_quats[0]}}
        opts["obj_init_options"] = ({"init_xy": np.array([a.obj_init_xs[0], a.obj_init_ys[0]])} if a.obj_variation_mode == "xy"
                                    else {"episode_id": 0})
        obs, _ = env.reset(options=opts)
        u = env.unwrapped
        cam = "3rd_view_camera" if "widowx" in u.robot_uid.lower() else "overhead_camera"
        links = u.agent.robot.get_links()
        grip = [l for l in links if PAT.search(l.name)]
        gids = np.array([l.id for l in grip]); rids = np.array([l.id for l in links])
        print(task, a.env_name, "robot_uid", u.robot_uid, "camera", cam, "instruction:", env.get_language_instruction())
        print(task, "gripper links:", [l.name for l in grip])
        rgb0 = get_image_from_maniskill2_obs_dict(env, obs).copy()
        seg = obs["image"][cam]["Segmentation"][..., 1]
        gmask = np.isin(seg, gids); rmask = np.isin(seg, rids)
        print(task, "image shape", rgb0.shape, "pixels gripper", int(gmask.sum()), "robot", int(rmask.sum()))
        raw0 = raw_rgb(u.get_obs(), cam)
        print(task, "wrapper rgb vs raw Color max diff", int(np.abs(raw0.astype(int) - rgb0.astype(int)).max()))
        n_vb = 0
        for l in grip:
            for vb in l.get_visual_bodies(): vb.set_visibility(0.0); n_vb += 1
        obs2 = u.get_obs(); rgb_hide = raw_rgb(obs2, cam)
        seg2 = obs2["image"][cam]["Segmentation"][..., 1]
        print(task, "hidden visual bodies", n_vb, "gripper pixels after hide", int(np.isin(seg2, gids).sum()))
        diff = np.abs(rgb_hide.astype(int) - rgb0.astype(int)).sum(-1) > 30
        anyd = np.abs(rgb_hide.astype(int) - rgb0.astype(int)).max(-1) > 0
        print(task, "changed pixels after hide", int(diff.sum()), "changed outside gripper mask (>30)", int((diff & ~gmask).sum()),
              "(any)", int((anyd & ~gmask).sum()))
        black = rgb0.copy(); black[gmask] = 0
        cs, _ = cv2.findContours(gmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        contour = rgb_hide.copy(); cv2.drawContours(contour, cs, -1, (255, 255, 255), 2)
        imageio.imwrite(os.path.join(out, f"{task}_panel.png"), np.concatenate([rgb0, black, rgb_hide, contour], axis=1))
        for l in grip:
            for vb in l.get_visual_bodies(): vb.set_visibility(1.0)
        print(task, "restore max abs diff vs orig", int(np.abs(raw_rgb(u.get_obs(), cam).astype(int) - rgb0.astype(int)).max()))
        # reset 后可见性是否保留（接入 C3/C4/C5 时要不要每回合重设）
        for l in grip:
            for vb in l.get_visual_bodies(): vb.set_visibility(0.0)
        obs3, _ = env.reset(options=opts)
        print(task, "gripper pixels after hide then reset", int(np.isin(obs3["image"][cam]["Segmentation"][..., 1], [l.id for l in env.unwrapped.agent.robot.get_links() if PAT.search(l.name)]).sum()))
        env.close()
    except Exception:
        traceback.print_exc()
