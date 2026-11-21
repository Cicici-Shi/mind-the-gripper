"""硬门槛 4 的反向实验（第二阶段，在 bench2_cogact 环境与官方 SimplerEnv 上重做一遍；不跑模型）。
在当前 VK_ICD_FILENAMES 下建两个环境（WidowX 勺子、Google 横放可乐罐，用 eval_views.make_env，即 CogACT 的设置）、reset、
取第 0 帧，打印能否起来、报错原文、第 0 帧 sha256。由 jobs/check.sbatch 在默认、只给 lavapipe、只给 nvidia 三种设置下各起一个进程。
用法：python icd_probe.py <标签>
"""
import hashlib
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
tag = sys.argv[1]
print(f"[icd] {tag} VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '<unset>')} "
      f"VK_DRIVER_FILES={os.environ.get('VK_DRIVER_FILES', '<unset>')}", flush=True)
try:
    import numpy as np
    import eval_views as E
    import keys as K
    for blk, task, opts in (("B2", "widowx_spoon_on_towel", {"obj_init_options": {"episode_id": 0}, "reconfigure": True}),
                            ("F1", "google_robot_pick_horizontal_coke_can", {"reconfigure": True})):
        env = E.make_env(task, K.BLOCKS[blk])
        obs, _ = env.reset(seed=12345, options=opts)
        cam = "3rd_view_camera" if "widowx" in env.unwrapped.robot_uid else "overhead_camera"
        rgb = np.ascontiguousarray(obs["image"][cam]["rgb"])
        print(f"[icd] {tag} OK {task} shape {rgb.shape} mean {rgb.mean():.3f} sha256 "
              f"{hashlib.sha256(rgb.tobytes()).hexdigest()}", flush=True)
        env.close()
    print(f"[icd] {tag} RESULT OK", flush=True)
except BaseException:
    print(f"[icd] {tag} RESULT FAIL\n" + traceback.format_exc(), flush=True)
    sys.exit(3)
