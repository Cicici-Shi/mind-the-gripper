# 无头渲染冒烟：WidowX 与 Google Robot 各渲染一帧，并打印渲染器/相机信息
import os, sys, time, traceback
import numpy as np
out = sys.argv[1] if len(sys.argv) > 1 else "."
os.makedirs(out, exist_ok=True)
import imageio
import simpler_env
from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict
for task in ["widowx_spoon_on_towel", "google_robot_pick_coke_can"]:
    t0 = time.time()
    try:
        env = simpler_env.make(task)
        obs, info = env.reset(seed=0)
        img = get_image_from_maniskill2_obs_dict(env, obs)
        print(task, "img", img.shape, img.dtype, "mean", img.mean(), "t", round(time.time()-t0, 2), flush=True)
        print(task, "instruction:", env.get_language_instruction(), flush=True)
        print(task, "cameras:", list(obs["image"].keys()), flush=True)
        imageio.imwrite(os.path.join(out, f"{task}_rgb.png"), img)
        env.close()
    except Exception:
        traceback.print_exc()
