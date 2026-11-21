"""硬门槛 4 的反向实验（协议 v1.1 冻结前，接手方）：在当前 VK_ICD_FILENAMES 下建一个 SimplerEnv 环境、reset、取第 0 帧。
打印能否起来、报错原文、第 0 帧 sha256。由 jobs/icd_check.sbatch 在三种设置下各起一个进程调用：
不设（默认）、只给 lavapipe（lvp_icd.json）、只给 nvidia（nvidia_icd.json）。不跑模型。
用法：python icd_probe.py <标签>
"""
import hashlib
import os
import sys
import traceback

tag = sys.argv[1]
print(f"[icd] {tag} VK_ICD_FILENAMES={os.environ.get('VK_ICD_FILENAMES', '<unset>')}", flush=True)
try:
    import numpy as np
    import simpler_env
    out = {}
    for task, opts in (("widowx_spoon_on_towel", {"obj_init_options": {"episode_id": 0}, "reconfigure": True}),
                       ("google_robot_pick_horizontal_coke_can", {"reconfigure": True})):
        env = simpler_env.make(task)
        obs, _ = env.reset(seed=12345, options=opts)
        cam = "3rd_view_camera" if "widowx" in env.unwrapped.robot_uid else "overhead_camera"
        rgb = np.ascontiguousarray(obs["image"][cam]["rgb"])
        out[task] = hashlib.sha256(rgb.tobytes()).hexdigest()
        print(f"[icd] {tag} OK {task} shape {rgb.shape} mean {rgb.mean():.3f} sha256 {out[task]}", flush=True)
        env.close()
    print(f"[icd] {tag} RESULT OK", flush=True)
except BaseException:
    print(f"[icd] {tag} RESULT FAIL\n" + traceback.format_exc(), flush=True)
    sys.exit(3)
