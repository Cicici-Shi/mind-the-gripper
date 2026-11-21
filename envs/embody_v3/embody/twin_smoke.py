"""Smoke test for TWIN_RENDER: replay the first SRC actions of one episode on each body, save the
twin images the policy would see next to the body's own images, and print twin placement error.

Usage (GPU node, EGL): TWIN_RENDER=1 python -m embody.twin_smoke SRC_DUMP_DIR OUT_DIR
"""
import glob
import os
import sys

import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from .body_env import BodyEnv
from .original import OriginalSpatialSuite


def main(src_dir, out_dir, task=0):
    os.makedirs(out_dir, exist_ok=True)
    suite = OriginalSpatialSuite()
    t = suite.get_task(task)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    f = sorted(glob.glob(os.path.join(src_dir, f"t{task}_e0_i*.npz")))[0]
    z = np.load(f, allow_pickle=True)
    init = int(z["init"])
    inits = suite.get_task_init_states(task)
    for body in ["Panda", "IIWA", "UR5e", "PandaR85"]:
        ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
        ref.seed(0)
        env = BodyEnv(ref, bddl, body, 256, 10, log=lambda m: print(m, flush=True))
        env.reset()
        obs = env.set_init_state(inits[init])
        for k, a in enumerate(z["action"][:41]):
            if k in (0, 20, 40):
                env.env.env._update_observables(force=True)
                nat = env.env.env._get_observations()
                row = [obs["agentview_image"][::-1, ::-1], nat["agentview_image"][::-1, ::-1],
                       obs["robot0_eye_in_hand_image"][::-1, ::-1], nat["robot0_eye_in_hand_image"][::-1, ::-1]]
                Image.fromarray(np.concatenate(row, axis=1)).save(os.path.join(out_dir, f"{body}_s{k}.png"))
            obs, _, done, _ = env.step(a.tolist())
        e = env.twin_err_mm
        print(f"{body}: twin err mm mean {np.mean(e):.3f} max {np.max(e):.3f} over {len(e)} renders", flush=True)
        env.close()
        env.twin.close()


if __name__ == "__main__":
    main(*sys.argv[1:3])
