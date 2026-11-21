"""GRP through BodyEnv: alignment log, mapped proprio, camera-in-site, images, and a short
open/close drive to check the opening-fraction mapping tracks the real linkage."""
import os, sys, numpy as np
sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody_dev"))
from embody import align, robots  # noqa
from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from PIL import Image
OUT = os.path.expanduser("$PROJ_ROOT/embody_dev/grp_check"); os.makedirs(OUT, exist_ok=True)
suite = OriginalSpatialSuite()
for tid in [0, 4]:
    task = suite.get_task(tid); inits = suite.get_task_init_states(tid)
    b = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    ref = OffScreenRenderEnv(bddl_file_name=b, camera_heights=256, camera_widths=256); ref.seed(0)
    for body in ["Panda", "PandaR85"]:
        env = BodyEnv(ref, b, body, 256, 10)
        env.reset(); obs = env.set_init_state(inits[0])
        f, fr = align.frames(env.env), align.frames(ref)
        print(f"task{tid} {body}: gripper_qpos(mapped)={obs['robot0_gripper_qpos'].round(4)} ref={ref._get_observations()['robot0_gripper_qpos'].round(4) if hasattr(ref,'_get_observations') else ''}"
              f" cam_in_site={f['wrist_cam_pos_in_site'].round(4)} ref_cam_in_site={fr['wrist_cam_pos_in_site'].round(4)}"
              f" eefq={obs['robot0_eef_quat'].round(3)}", flush=True)
        Image.fromarray(obs["agentview_image"][::-1]).save(f"{OUT}/t{tid}_{body}_agent.png")
        Image.fromarray(obs["robot0_eye_in_hand_image"][::-1]).save(f"{OUT}/t{tid}_{body}_wrist.png")
        if tid == 0:
            seq = []
            for k in range(60):
                a = [0, 0, 0, 0, 0, 0, 1 if k < 40 else -1]
                obs, *_ = env.step(a)
                seq.append(obs["robot0_gripper_qpos"][0])
            print(f"  {body} close 40 / open 20, gripper_qpos[0] every 5:", np.round(seq[::5], 4), flush=True)
        env.env.close()
