"""Where do twin and reference differ at step 0? Saves ref | twin | 10x diff for both cameras.

Usage (GPU node): TWIN_RENDER=1 python -m embody.twin_t0diag OUT_DIR
"""
import os
import sys

import numpy as np
from PIL import Image

from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from .body_env import BodyEnv
from .original import OriginalSpatialSuite


def main(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    suite = OriginalSpatialSuite()
    t = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file)
    inits = suite.get_task_init_states(0)
    ref = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256)
    ref.seed(0)
    env = BodyEnv(ref, bddl, "Panda", 256, 10, log=lambda m: print(m, flush=True))
    held = {}
    orig = env._twin_t0
    env._twin_t0 = lambda r, o: (held.update(ref=r, twin=dict(o)), orig(r, o))
    env.reset()
    env.set_init_state(inits[42])
    for k in ("agentview_image", "robot0_eye_in_hand_image"):
        a = np.asarray(held["ref"][k]).astype(np.int16)
        b = np.asarray(held["twin"][k]).astype(np.int16)
        diff = np.abs(a - b).max(axis=2)
        ys, xs = np.nonzero(diff > 2)
        print(k, "pixels >2:", len(ys), "bbox y", (ys.min(), ys.max()) if len(ys) else None,
              "x", (xs.min(), xs.max()) if len(xs) else None, flush=True)
        vis = np.clip(diff * 10, 0, 255).astype(np.uint8)
        row = np.concatenate([a.astype(np.uint8), b.astype(np.uint8), np.stack([vis] * 3, axis=2)], axis=1)
        Image.fromarray(row[::-1, ::-1]).save(os.path.join(out_dir, f"t0_{k[:5]}.png"))
    env.close()


if __name__ == "__main__":
    main(sys.argv[1])
