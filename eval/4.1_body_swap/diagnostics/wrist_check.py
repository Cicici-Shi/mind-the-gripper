"""How much of each camera is the arm itself?

UR5e's wrist image differed from Panda's by 2.5 grey levels in the G1 probe, which
suggests its arm reaches into the wrist view. That matters: the wrist camera is meant
to be body-independent (it rides on the gripper), so any arm pixels there leak the
arm-appearance factor into a channel we assumed was clean.

Per original task and body, this renders segmentation masks for both cameras and
reports the fraction of pixels belonging to arm links (vs gripper, vs scene), at the
first observation and over 10 steps of downward motion (the arm folds into view when
the gripper descends). Writes out/wrist_task{t}.json plus overlay images.
"""
import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embody import align, robots  # noqa: E402,F401  (robots registers classes)

from libero.libero import get_libero_path  # noqa: E402
from libero.libero.envs import OffScreenRenderEnv  # noqa: E402

DUMMY = np.array([0, 0, 0, 0, 0, 0, -1.0])
WAIT = 10
RES = 256
CAMS = ["agentview", "robot0_eye_in_hand"]


def make_env(task, robot):
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env = OffScreenRenderEnv(bddl_file_name=bddl, camera_heights=RES, camera_widths=RES, robots=[robot])
    env.seed(0)
    env.reset()
    return env


def geom_class(env):
    """Per geom id: 1 = arm link, 2 = gripper, 3 = mount, 0 = everything else.

    The mount is the same RethinkMount for every body, so it is kept apart from the
    arm: hiding the arm must not be confused with hiding shared hardware.
    """
    model = env.sim.model
    cls = np.zeros(model.ngeom, dtype=np.uint8)
    for g in range(model.ngeom):
        body = model.body_id2name(model.geom_bodyid[g])
        if body is None:
            continue
        if body.startswith("gripper0"):
            cls[g] = 2
        elif body.startswith("mount0"):
            cls[g] = 3
        elif body.startswith("robot0"):
            cls[g] = 1
    return cls


MJOBJ_GEOM = 5


def seg_fractions(env, cls, cam):
    """Fraction of camera pixels showing arm links and gripper.

    The segmentation buffer holds (objtype, objid). Sites are drawn too and carry
    their own id space, so only objtype == geom may be looked up in `cls`.
    """
    seg = env.sim.render(width=RES, height=RES, camera_name=cam, segmentation=True)
    otype, gid = seg[..., 0], seg[..., 1]
    valid = (otype == MJOBJ_GEOM) & (gid >= 0)
    out = np.zeros(gid.shape, dtype=np.uint8)
    out[valid] = cls[gid[valid]]
    n = out.size
    # flipped to match the orientation of the RGB renders used everywhere else
    return float((out == 1).sum() / n), float((out == 2).sum() / n), out[::-1]


def overlay(env, cam, mask):
    """RGB with arm pixels tinted red and gripper pixels tinted blue."""
    img = env.sim.render(width=RES, height=RES, camera_name=cam)[::-1].astype(np.int32)
    img[mask == 1] = (img[mask == 1] * 0.4 + np.array([255, 0, 0]) * 0.6).astype(np.int32)
    img[mask == 2] = (img[mask == 2] * 0.4 + np.array([0, 0, 255]) * 0.6).astype(np.int32)
    return img.clip(0, 255).astype(np.uint8)


def run(task_id, suite, outdir, save_images):
    task = suite.get_task(task_id)
    init = suite.get_task_init_states(task_id)[0]

    env = make_env(task, "Panda")
    obs = env.set_init_state(init)
    for _ in range(WAIT):
        obs, _, _, _ = env.step(DUMMY)
    f_ref = align.frames(env)
    objs = align.object_state_by_name(env)
    q_ref = obs["robot0_eef_quat"].copy()
    env.close()

    rec = {"task_id": task_id, "task": task.name, "bodies": {}}
    tiles = []
    ref_rgb, ref_mask = {}, {}
    for arm in ["Panda", "IIWA", "UR5e"]:
        env = make_env(task, arm)
        q = align.ik_site(env, f_ref["site_pos"], f_ref["site_mat"], env.robots[0].init_qpos)
        align.load_state_transfer(env, objs, q, f_ref["gripper_qpos"])
        cls = geom_class(env)

        per_cam = {}
        for cam in CAMS:
            a0, g0, mask0 = seg_fractions(env, cls, cam)
            per_cam[cam] = {"arm_frac_t0": a0, "gripper_frac_t0": g0}
            rgb = env.sim.render(width=RES, height=RES, camera_name=cam)[::-1].astype(np.int32)
            if arm == "Panda":
                ref_rgb[cam], ref_mask[cam] = rgb, mask0
            else:
                # Where does the pixel difference against Panda actually live?
                d = np.abs(rgb - ref_rgb[cam]).mean(axis=2)
                arm_pix = (mask0 == 1) | (ref_mask[cam] == 1)
                total = d.sum()
                per_cam[cam]["rgb_mean_abs_diff"] = float(d.mean())
                per_cam[cam]["diff_share_on_arm_pixels"] = float(d[arm_pix].sum() / total) if total else 0.0
                per_cam[cam]["mean_abs_diff_off_arm"] = float(d[~arm_pix].mean())
            if cam == "robot0_eye_in_hand" and save_images:
                tiles.append(overlay(env, cam, mask0))

        # descend for 10 steps: the arm swings into the wrist view as the gripper drops
        traj = {cam: [] for cam in CAMS}
        a = DUMMY.copy()
        a[2] = -0.5
        for _ in range(10):
            env.step(a)
            for cam in CAMS:
                traj[cam].append(seg_fractions(env, cls, cam)[0])
        for cam in CAMS:
            per_cam[cam]["arm_frac_max_descend"] = float(max(traj[cam]))
            per_cam[cam]["arm_frac_end_descend"] = float(traj[cam][-1])

        rec["bodies"][arm] = per_cam
        env.close()

    if save_images and tiles:
        Image.fromarray(np.concatenate(tiles, axis=1)).save(os.path.join(outdir, f"wrist_task{task_id}.png"))
    with open(os.path.join(outdir, f"wrist_task{task_id}.json"), "w") as f:
        json.dump(rec, f, indent=1)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", type=int, default=10)
    ap.add_argument("--out", default="out")
    ap.add_argument("--images", type=int, default=2, help="save wrist overlays for the first N tasks")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from embody.original import OriginalSpatialSuite
    suite = OriginalSpatialSuite()
    print("original tasks:", suite.n_tasks, flush=True)

    agg = {}
    for t in range(min(args.tasks, suite.n_tasks)):
        rec = run(t, suite, args.out, t < args.images)
        for arm, cams in rec["bodies"].items():
            w, a = cams["robot0_eye_in_hand"], cams["agentview"]
            agg.setdefault(arm, []).append((w["arm_frac_t0"], w["arm_frac_max_descend"], a["arm_frac_t0"],
                                            w.get("rgb_mean_abs_diff", 0.0), w.get("diff_share_on_arm_pixels", 0.0),
                                            a.get("rgb_mean_abs_diff", 0.0), a.get("diff_share_on_arm_pixels", 0.0)))
            print(f"task{t} {arm:5s} wrist arm% t0={w['arm_frac_t0']*100:.2f} "
                  f"max={w['arm_frac_max_descend']*100:.2f} diff={w.get('rgb_mean_abs_diff', 0):.2f} "
                  f"(on-arm {w.get('diff_share_on_arm_pixels', 0)*100:.0f}%) | "
                  f"agentview arm% t0={a['arm_frac_t0']*100:.2f} diff={a.get('rgb_mean_abs_diff', 0):.2f} "
                  f"(on-arm {a.get('diff_share_on_arm_pixels', 0)*100:.0f}%)", flush=True)

    print("\nmean over tasks:", flush=True)
    for arm, rows in agg.items():
        r = np.array(rows)
        print(f"  {arm:5s} wrist arm% t0 {r[:,0].mean()*100:.2f}  max-descend {r[:,1].mean()*100:.2f}  "
              f"diff {r[:,3].mean():.2f} on-arm {r[:,4].mean()*100:.0f}% | "
              f"agentview arm% {r[:,2].mean()*100:.2f}  diff {r[:,5].mean():.2f} on-arm {r[:,6].mean()*100:.0f}%",
              flush=True)


if __name__ == "__main__":
    main()
