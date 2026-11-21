"""Ceiling-free perception probe: how much does the policy's action change when only the body changes?

For SRC frames (states dumped by run_libero_eval with DUMP_DIR), rebuild the exact Panda scene,
transfer it to each target body (same grip-site pose, objects, fixtures), render, and query the
frozen policy on: the Panda observation (reference) and each body's observation. Also swaps only
the third-person or only the wrist image, to see which view carries the change. OFT's L1 head
is deterministic, so any action difference comes from the input.

Run from $PROJ_ROOT/openvla-oft:  python $PROJ_ROOT/embody/embody/probe.py SRC_DUMP_DIR OUT_JSONL [EVERY]
"""
import glob
import json
import os
import sys

sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/embody"))
sys.path.append(os.getcwd())

import numpy as np

from experiments.robot.libero.run_libero_eval import (
    GenerateConfig, initialize_model, prepare_observation, process_action,
)
from experiments.robot.robot_utils import get_action, get_image_resize_size, set_seed_everywhere
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from embody import align, robots  # noqa: F401
from embody.original import OriginalSpatialSuite

CKPT = "moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10"
BODIES = ["Panda", "IIWA", "UR5e"]


def set_alpha(env, hidden, saved):
    m = env.sim.model
    for g, a in saved.items():
        m.geom_rgba[g][3] = 0.0 if hidden else a


def arm_geoms(env):
    m = env.sim.model
    return {g: float(m.geom_rgba[g][3]) for g in range(m.ngeom)
            if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("robot0") and m.geom_rgba[g][3] > 0}


def obs_of(env):
    env.env._update_observables(force=True)
    return env.env._get_observations()


def load_frame_into_ref(ref, z, k):
    m = ref.sim.model
    names = [str(x) for x in z["ref_body_names"]]
    for b, name in enumerate(names):
        if not name:
            continue
        try:
            bid = m.body_name2id(name)
        except (ValueError, KeyError):
            continue
        m.body_pos[bid] = z["ref_body_pos"][b]
        m.body_quat[bid] = z["ref_body_quat"][b]
    ref.sim.data.qpos[:] = z["qpos"][k]
    ref.sim.data.qvel[:] = 0.0
    ref.sim.forward()
    return obs_of(ref)


def transfer(target, ref, q_ref):
    f_ref = align.frames(ref)
    objs = align.object_state_by_name(ref)
    fixtures = align.static_body_pose(ref)
    q = align.ik_site(target, f_ref["site_pos"], f_ref["site_mat"], target.robots[0].init_qpos)
    align.apply_static_body_pose(target, fixtures)
    obs, _ = align.load_state_transfer(target, objs, q, f_ref["gripper_qpos"])
    sign = align.quat_sign(obs["robot0_eef_quat"], q_ref)
    obs["robot0_eef_quat"] = obs["robot0_eef_quat"] * sign
    f = align.frames(target)
    return obs, float(np.linalg.norm(f["site_pos"] - f_ref["site_pos"]) * 1000), sign


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
        return np.stack([process_action(np.array(x), cfg.model_family) for x in a])  # (8, 7)

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
    files = sorted(glob.glob(os.path.join(src_dir, "t*_e*_i*.npz")))
    by_task = {}
    for f in files:
        by_task.setdefault(int(np.load(f, allow_pickle=True)["task"]), []).append(f)
    done = set()
    if os.path.exists(out_path):
        for line in open(out_path):
            r = json.loads(line)
            done.add((r["task"], r["ep"], r["step"]))
    out = open(out_path, "a")
    for tid in sorted(by_task):
        task = suite.get_task(tid)
        bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
        mk = lambda body: OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=256, camera_widths=256, robots=[body])
        ref = mk("Panda"); ref.seed(0); ref.reset()
        targets = {}
        for b in BODIES:
            e = mk(b); e.seed(0); e.reset(); targets[b] = (e, arm_geoms(e))
        ref_alpha = arm_geoms(ref)
        for f in by_task[tid]:
            z = np.load(f, allow_pickle=True)
            T = len(z["action"])
            for k in range(0, T, every):  # chunk boundaries (8 open-loop steps) when every % 8 == 0
                if (tid, int(z["ep"]), k) in done:
                    continue
                o_ref = load_frame_into_ref(ref, z, k)
                a_ref = act(o_ref, task.language)
                rec = {"task": tid, "ep": int(z["ep"]), "init": int(z["init"]), "step": k, "T": T,
                       "src_success": bool(z["success"]), "gripper_closed": bool(z["action"][k][6] > 0),
                       "replay_check": float(np.abs(a_ref[0] - z["action"][k]).max()),
                       "a_ref_pos_norm": float(np.linalg.norm(a_ref[:, :3], axis=1).mean()),
                       "a_ref_rot_norm": float(np.linalg.norm(a_ref[:, 3:6], axis=1).mean())}
                rec["repeat"] = diff(act(o_ref, task.language), a_ref)  # same input twice: GPU determinism
                # Panda arm hidden on the reference scene itself (APP-strong)
                set_alpha(ref, True, ref_alpha)
                o_h = obs_of(ref)
                set_alpha(ref, False, ref_alpha)
                rec["Panda-hid"] = diff(act(o_h, task.language), a_ref)
                for b, (e, alpha) in targets.items():
                    set_alpha(e, False, alpha)
                    o_t, pos_mm, sign = transfer(e, ref, o_ref["robot0_eef_quat"])
                    rec[f"{b}"] = diff(act(o_t, task.language), a_ref)
                    rec[f"{b}_align_mm"] = pos_mm
                    if b != "Panda":
                        rec[f"{b}-3rd-only"] = diff(act(mix(o_ref, img_from=o_t), task.language), a_ref)
                        rec[f"{b}-wrist-only"] = diff(act(mix(o_ref, wrist_from=o_t), task.language), a_ref)
                        set_alpha(e, True, alpha)
                        o_th = obs_of(e)
                        o_th["robot0_eef_quat"] = o_th["robot0_eef_quat"] * sign
                        set_alpha(e, False, alpha)
                        rec[f"{b}-hid"] = diff(act(o_th, task.language), a_ref)
                out.write(json.dumps(rec) + "\n")
                out.flush()
        for e, _ in targets.values():
            e.close()
        ref.close()


if __name__ == "__main__":
    main(*sys.argv[1:])
