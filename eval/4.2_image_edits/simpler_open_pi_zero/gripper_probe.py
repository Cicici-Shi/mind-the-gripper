# 问题 3：隐藏夹爪连杆、按连杆取分割掩膜。每个任务存原图、掩膜叠加、隐藏后、涂黑、擦掉再画轮廓五联图
import os, sys, re, traceback
import numpy as np, cv2, imageio
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
PAT = re.compile(r"finger|gripper", re.I)

def cam_name(env):
    u = env.unwrapped
    return "3rd_view_camera" if "widowx" in u.robot_uid.lower() else "overhead_camera"

def raw_rgb(o, cam):
    # u.get_obs() 返回未经包装器转换的观测：Color 为 0..1 浮点，已做过真实背景叠加
    return (np.clip(o["image"][cam]["Color"][..., :3], 0, 1) * 255).astype(np.uint8)

def label(img, s):
    img = img.copy(); cv2.putText(img, s, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2); return img

for task in ["widowx_spoon_on_towel", "google_robot_pick_coke_can"]:
    try:
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=0)
        u = env.unwrapped
        cam = cam_name(env)
        links = u.agent.robot.get_links()
        print(task, "robot_uid", u.robot_uid, "camera", cam)
        print(task, "links:", [(l.name, l.id) for l in links])
        grip = [l for l in links if PAT.search(l.name)]
        gids = np.array([l.id for l in grip]); rids = np.array([l.id for l in links])
        print(task, "gripper links:", [l.name for l in grip])
        rgb0 = get_image_from_maniskill2_obs_dict(env, obs).copy()
        seg = obs["image"][cam]["Segmentation"][..., 1]
        gmask = np.isin(seg, gids); rmask = np.isin(seg, rids)
        print(task, "per-link pixels:", {l.name: int((seg == l.id).sum()) for l in links if (seg == l.id).any()})
        print(task, "unique actor ids in view:", np.unique(seg).tolist())
        raw0 = raw_rgb(u.get_obs(), cam)
        print(task, "wrapper rgb vs raw Color max diff", int(np.abs(raw0.astype(int) - rgb0.astype(int)).max()))
        print(task, "pixels gripper", int(gmask.sum()), "robot", int(rmask.sum()), "of", gmask.size)
        vis = rgb0.copy(); vis[gmask] = (0.5 * vis[gmask] + 0.5 * np.array([255, 0, 255])).astype(np.uint8)
        vis[rmask & ~gmask] = (0.6 * vis[rmask & ~gmask] + 0.4 * np.array([0, 255, 255])).astype(np.uint8)
        black = rgb0.copy(); black[gmask] = 0
        # 隐藏夹爪可视体后重新取观测
        n_vb = 0
        for l in grip:
            for vb in l.get_visual_bodies():
                vb.set_visibility(0.0); n_vb += 1
        obs2 = u.get_obs()
        rgb_hide = raw_rgb(obs2, cam)
        seg2 = obs2["image"][cam]["Segmentation"][..., 1]
        print(task, "hidden visual bodies", n_vb, "gripper pixels after hide", int(np.isin(seg2, gids).sum()))
        diff = np.abs(rgb_hide.astype(int) - rgb0.astype(int)).sum(-1) > 30
        print(task, "changed pixels after hide", int(diff.sum()), "changed outside gripper mask", int((diff & ~gmask).sum()))
        contour = rgb_hide.copy()
        cs, _ = cv2.findContours(gmask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(contour, cs, -1, (255, 255, 255), 2)
        panel = np.concatenate([label(rgb0, "orig"), label(vis, "seg: gripper=magenta"), label(black, "blacken"),
                                label(rgb_hide, "erase(hide visual)"), label(contour, "erase+contour")], axis=1)
        imageio.imwrite(os.path.join(out, f"{task}_panel.png"), panel)
        imageio.imwrite(os.path.join(out, f"{task}_orig.png"), rgb0)
        imageio.imwrite(os.path.join(out, f"{task}_hide.png"), rgb_hide)
        # 恢复可见并确认回到原图
        for l in grip:
            for vb in l.get_visual_bodies(): vb.set_visibility(1.0)
        rgb_back = raw_rgb(u.get_obs(), cam)
        print(task, "restore max abs diff vs orig", int(np.abs(rgb_back.astype(int) - rgb0.astype(int)).max()))
        env.close()
    except Exception:
        traceback.print_exc()
