"""Gripper-swap perception probe: same SRC frames as probe.py, target = Panda arm + Robotiq 85.

Each frame is loaded into a Panda reference env (fixture poses + qpos), the reference gripper's
command state is rebuilt from the logged gripper commands, and the frame goes through the same
BodyEnv path as the GRP evaluation (wait steps = 0): IK on grip site, Robotiq opening matched,
wrist camera remounted, proprio mapped. 'Panda' through the same BodyEnv path is the noise floor.
Conditions: Panda, R85 (both views + proprio), R85-3rd-only, R85-wrist-only.

Run from $PROJ_ROOT/openvla-oft with EMBODY_DIR=$PROJ_ROOT/embody_dev:
  python probe_grp.py SRC_DUMP_DIR OUT_JSONL [EVERY]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.expanduser(os.environ.get("EMBODY_DIR", "$PROJ_ROOT/embody_dev")))
sys.path.append(os.getcwd())

import numpy as np

from experiments.robot.libero.run_libero_eval import GenerateConfig, initialize_model, prepare_observation, process_action
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from embody.body_env import BodyEnv
from embody.original import OriginalSpatialSuite
from embody.probe import load_frame_into_ref  # same frame loader as the arm probe

CKPT = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
STEP_PER_ACTION = 0.25  # gripper command change per control step (speed 0.01 x 25 substeps)


def grip_command(actions, k):
    c = -1.0  # open after the wait steps
    for a in actions[:k]:
        c = float(np.clip(c + STEP_PER_ACTION * np.sign(a[-1]), -1.0, 1.0))
    return c


def main(src_dir, out_path, every=16):
    every = int(every)
    cfg = GenerateConfig(pretrained_checkpoint=CKPT, task_suite_name="libero_spatial")
    set_seed_everywhere(cfg.seed)
    model, action_head, proprio_projector, noisy_action_projector, processor = initialize_model(cfg)
    resize_size = get_image_resize_size(cfg)

    def act(obs, desc):
        o, _ = prepare_observation(obs, resize_size)
        a = get_action(cfg, model, o, desc, processor=processor, action_head=action_head,
                       proprio_projector=proprio_projector, noisy_action_projector=noisy_action_projector,
                       use_film=cfg.use_film)
        return np.stack([process_action(np.array(x), cfg.model_family) for x in a])

    def mix(base, img_from=None, wrist_from=None):
        o = dict(base)
        if img_from is not None:
            o["agentview_image"] = img_from["agentview_image"]
        if wrist_from is not None:
            o["robot0_eye_in_hand_image"] = wrist_from["robot0_eye_in_hand_image"]
        return o

    def diff(a, b):
        return dict(dpos=float(np.linalg.norm(a[:, :3] - b[:, :3], axis=1).mean()),
                    drot=float(np.linalg.norm(a[:, 3:6] - b[:, 3:6], axis=1).mean()),
                    grip_flip=float((np.sign(a[:, 6]) != np.sign(b[:, 6])).mean()))

    suite = OriginalSpatialSuite()
    by_task = {}
    for f in sorted(glob.glob(os.path.join(src_dir, "t*_e*_i*.npz"))):
        by_task.setdefault(int(np.load(f, allow_pickle=True)["task"]), []).append(f)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["task"], r["ep"], r["step"]))
    out = open(out_path, "a")
    quiet = lambda m: None
    for tid in sorted(by_task):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        mk = lambda: OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, robots=["Panda"])
        ref = mk(); ref.seed(0); ref.reset()
        ref2 = mk(); ref2.seed(0); ref2.reset()
        envs = {"Panda": BodyEnv(ref2, bddl, "Panda", 256, 0, log=quiet), "R85": BodyEnv(ref2, bddl, "PandaR85", 256, 0, log=quiet)}
        for e in envs.values():
            e.env.reset()
        for f in by_task[tid]:
            z = np.load(f, allow_pickle=True)
            T = len(z["action"])
            for k in range(0, T, every):
                if (tid, int(z["ep"]), k) in done:
                    continue
                o_ref = load_frame_into_ref(ref, z, k)
                a_ref = act(o_ref, task.language)
                c = grip_command(z["action"], k)
                rec = {"task": tid, "ep": int(z["ep"]), "init": int(z["init"]), "step": k, "T": T,
                       "src_success": bool(z["success"]), "gripper_closed": bool(z["action"][k][6] > 0),
                       "grip_cmd": c, "a_ref_pos_norm": float(np.linalg.norm(a_ref[:, :3], axis=1).mean())}
                for name, e in envs.items():
                    load_frame_into_ref(ref2, z, k)
                    ref2.robots[0].gripper.current_action = np.array([-c, c])
                    state = ref2.sim.get_state().flatten()
                    o_t = e.set_init_state(state)
                    ref2.robots[0].gripper.current_action = np.array([-c, c])
                    rec[name] = diff(act(o_t, task.language), a_ref)
                    if name == "R85":
                        rec["R85-3rd-only"] = diff(act(mix(o_ref, img_from=o_t), task.language), a_ref)
                        rec["R85-wrist-only"] = diff(act(mix(o_ref, wrist_from=o_t), task.language), a_ref)
                        rec["R85-proprio-only"] = diff(act(dict(o_ref, robot0_gripper_qpos=o_t["robot0_gripper_qpos"]), task.language), a_ref)
                out.write(json.dumps(rec) + "\n")
                out.flush()
        for e in envs.values():
            e.env.close()
        ref.close(); ref2.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
