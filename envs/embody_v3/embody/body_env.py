"""Evaluation wrapper that runs a LIBERO episode on another body.

embody_v3 (2026-10-27): merge of $PROJ_ROOT/embody (published block-1 code, RECOLOR_ARM) and
$PROJ_ROOT/embody_dev (block-2 code: Robotiq, TWIN_RENDER, GRP_KP, seed library), plus block-3 options.
With the environment variables of an existing run, the code path and numbers of that run are
unchanged (regression: tools/regress.py, WORKLOG section 7, 10/27).

For each episode:
  1. A Panda reference env runs the original pipeline: set_init_state + wait steps.
  2. The settled scene (objects by joint name, gripper qpos) and Panda's grip_site
     pose are transferred to the target body via IK.
  3. The policy starts immediately on the target body (skip_wait=True), so its first
     observation matches what it would have seen on Panda, up to the arm's appearance.

Body names: "Panda" (SRC-sham), "IIWA", "UR5e", "Kinova3" (K3), "PandaL3" (Panda with joint3 removed,
P6j3). Arms keep the Panda gripper.
Bodies ending in "R85" keep the Panda arm but use the Robotiq 85 gripper (GRP); "PandaRT" uses the
Rethink gripper (RT). For these foreign grippers: the wrist camera is moved so its pose in the
grip-site frame equals Panda's; the gripper starts at the same command state as the settled Panda
and settles under it; and the proprio gripper_qpos is mapped to the Panda-equivalent 2-vector by
opening fraction (endpoints calibrated in free air, tools/calib_grippers.py).
TWIN_RENDER=1 ("paint Panda back", as Mirage cross-painting but exact): physics runs on the target
body, while the policy's camera images come from a Panda twin env that is synced every step: same
fixtures, objects copied by joint name, Panda arm placed by IK on the target's grip-site pose
(warm-started, nullspace pull toward the episode-start joints), Panda fingers at the target's
opening. Proprio is unchanged. TWIN_BODY=UR5e paints a UR5e instead (reverse cell: Panda physics,
UR5e pixels). For a Robotiq target the Panda pads are placed where the Robotiq pads are (6.4 mm
back along the site z axis). Per-episode checks are kept in self.twin_stats (see _twin_*).
  TWIN_BODY=PandaR85 on Panda physics (G_R, block 3): the twin's arm IK is seeded every step with the
  physical Panda's joints (same arm), its site is placed so the Robotiq pads land on the Panda pads
  (+6.4 mm along site z, mirror of G_T), its six linkage joints come from the free-air calibration
  table (r85_table.npz) at the Panda's opening fraction, and its wrist camera is put at the physical
  Panda's wrist-camera world pose every step.
  TWIN_CAMS=both|wrist|agent (default both): which camera images are replaced by the twin's
  (G_T^wrist / G_T^agent). The other camera stays the physical body's own render.
COPY_GRIP_CMD (Panda-gripper bodies only): 1 copies the reference gripper's command state
  (current_action) to the target after the transfer, 0 leaves the target's fresh zero command.
  The two published code versions differ here: $PROJ_ROOT/embody never copied (block 1: sham, rec, iiwa,
  ur5e, *hid), $PROJ_ROOT/embody_dev copied by default (block 2 twin cells). Default "auto" = copy iff
  TWIN_RENDER=1, which reproduces every published run. Foreign grippers always copy (their init).
HIDE_ARM=1 sets alpha=0 on the arm's visual geoms (bodies named robot0_*): the gripper
stays visible and collision geoms are untouched, so physics is unchanged.
RECOLOR_ARM=1 paints the same visible arm geoms one flat color (material removed): appearance
only, gripper and physics unchanged.
"""
import os

import numpy as np

from . import align, robots  # noqa: F401  robots registers MountedIIWA / MountedUR5e / block-3 bodies
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.env_wrapper import ControlEnv

DUMMY = [0, 0, 0, 0, 0, 0, -1]
_HERE = os.path.dirname(os.path.abspath(__file__))


def gripper_kind(body):
    if body.endswith("R85"):
        return "r85"
    if body.endswith("RT"):
        return "rethink"
    if body.endswith("RT2"):
        return "rethink2"
    return "panda"


def panda_armed(body):
    """Bodies whose arm is the 7-joint Panda arm (same kinematics as the reference)."""
    return body in ("Panda", "PandaR85", "PandaRT", "PandaRT2")


class BodyEnv:
    skip_wait = True

    def __init__(self, ref_env, bddl_file, body, resolution, num_steps_wait, log=print, render=True):
        self.ref = ref_env
        self.body = body
        self.num_steps_wait = num_steps_wait
        self.log = log
        if render:
            self.env = OffScreenRenderEnv(
                bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution, robots=[body]
            )
        else:  # physics only (action replay): no GL context, no camera observations
            self.env = ControlEnv(
                bddl_file_name=bddl_file, robots=[body],
                has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False,
            )
        self.env.seed(0)
        self.sign = 1.0
        self.hide_arm = os.environ.get("HIDE_ARM") == "1"
        self.recolor_arm = os.environ.get("RECOLOR_ARM") == "1"
        self.gk = gripper_kind(body)
        self.grp = self.gk != "panda"  # foreign gripper on the physics body
        twin_on = os.environ.get("TWIN_RENDER") == "1"
        cg = os.environ.get("COPY_GRIP_CMD", "auto")
        assert cg in ("auto", "0", "1"), f"COPY_GRIP_CMD={cg}"
        self.copy_grip_cmd = twin_on if cg == "auto" else cg == "1"
        self.twin = None
        self.nc_geoms = None
        if twin_on and render:
            state = np.random.get_state()  # building the twin must not move the reference's RNG stream
            self.twin_body = os.environ.get("TWIN_BODY", "Panda")
            self.twin = OffScreenRenderEnv(
                bddl_file_name=bddl_file, camera_heights=resolution, camera_widths=resolution, robots=[self.twin_body]
            )
            self.twin.seed(0)
            np.random.set_state(state)
            self._twin_q = None
            self.twin_stats = None
            self.twin_gk = gripper_kind(self.twin_body)
            self.twin_cams = os.environ.get("TWIN_CAMS", "both")
            assert self.twin_cams in ("both", "wrist", "agent"), f"TWIN_CAMS={self.twin_cams}"
            # G_R-type twin: foreign-gripper twin on a Panda-armed body -> seed IK with the physical joints
            self.twin_seed_phys = self.twin_gk != "panda" and panda_armed(body) and panda_armed(self.twin_body)
            self.twin_dz = self._pad_shift(self.gk, self.twin_gk)
            # TWIN_MOUNT=sunk (G_R, protocol_block3 1.4): no IK, twin joints = physical joints, the twin's gripper
            # root body is sunk along the hand so the Robotiq pad centre sits on the Panda pad centre
            self.twin_sunk = os.environ.get("TWIN_MOUNT", "ik") == "sunk"
            if self.twin_sunk:
                assert self.twin_seed_phys and self.twin_gk == "r85", "TWIN_MOUNT=sunk is for TWIN_BODY=PandaR85 on Panda physics"
        # GRP_PAD_ALIGN=1: sensitivity variant, Robotiq pad centre placed where Panda's pad centre is
        # (site-frame z: Panda pad -3.6 mm, Robotiq pad -10.0 mm; measured by pad_offset.py)
        self.pad_dz = 0.0064 if (os.environ.get("GRP_PAD_ALIGN") == "1" and self.gk == "r85") else 0.0

    def reset(self):
        # LIBERO samples fixture placements from the GLOBAL np.random stream at every reset.
        # Restore the stream after the target's reset so the reference env draws exactly the
        # sequence it draws in a plain SRC run: same (task, episode) -> same scene across conditions.
        self.ref.reset()
        state = np.random.get_state()
        obs = self.env.reset()
        if self.twin is not None:
            self.twin.reset()  # also inside the saved RNG window: the reference draws stay unchanged
        np.random.set_state(state)
        return obs

    def set_init_state(self, init_state):
        obs = self.ref.set_init_state(init_state)
        for _ in range(self.num_steps_wait):
            obs, _, _, _ = self.ref.step(DUMMY)
        ref_obs = obs
        f_ref = align.frames(self.ref)
        objs = align.object_state_by_name(self.ref)
        fixtures = align.static_body_pose(self.ref)
        q_ref = obs["robot0_eef_quat"].copy()

        if self.hide_arm:
            n_hidden = self._hide_arm()  # reset may rebuild the model, so redo every episode
        if self.recolor_arm:
            n_recolored = self._recolor_arm()
        if os.environ.get("NO_FOREARM_CONTACT") == "1":
            self.nc_geoms = self._no_forearm_contact()
        p_target = f_ref["site_pos"]
        if self.grp and self.pad_dz:
            # align finger-pad centres instead of grip sites: Robotiq pads sit 6.4 mm further back
            p_target = p_target + f_ref["site_mat"] @ np.array([0.0, 0.0, self.pad_dz])
        q = align.ik_site(self.env, p_target, f_ref["site_mat"], self.env.robots[0].init_qpos)
        _, fix_mm = align.apply_static_body_pose(self.env, fixtures)
        if self.grp:
            obs, missing = align.load_state_transfer(self.env, objs, q, None)
            self._grp_init(f_ref)
            obs = self.env.env._get_observations()
        else:
            obs, missing = align.load_state_transfer(self.env, objs, q, f_ref["gripper_qpos"])
            if self.copy_grip_cmd:
                # same gripper: also copy its command state (the Panda reference has been sending
                # "open" for num_steps_wait steps; a fresh target gripper would start from 0)
                self.env.robots[0].gripper.current_action = np.array(self.ref.robots[0].gripper.current_action, copy=True)
        f = align.frames(self.env)
        self.sign = align.quat_sign(obs["robot0_eef_quat"], q_ref)
        self.align_info = {
            "pos_mm": float(np.linalg.norm(f["site_pos"] - f_ref["site_pos"]) * 1000),
            "rot_deg": float(align.rot_err_deg(f["site_mat"], f_ref["site_mat"])),
            "fixture_mm": float(fix_mm), "missing": list(missing), "quat_sign": float(self.sign),
            "wrist_cam_rot_deg": float(align.rot_err_deg(f["wrist_cam_mat_in_site"], f_ref["wrist_cam_mat_in_site"])),
            "wrist_cam_pos_mm": float(np.linalg.norm(f["wrist_cam_pos_in_site"] - f_ref["wrist_cam_pos_in_site"]) * 1000),
            "hand_in_site_deg": float(align.rot_err_deg(f["hand_in_site"], f_ref["hand_in_site"])),
            "copy_grip_cmd": int(self.copy_grip_cmd) if not self.grp else 1,  # 2026-10-31: same value as the BODY_ALIGN log
            "forearm_geoms_off": len(self.nc_geoms) if self.nc_geoms is not None else None,
        }
        self.log(
            f"BODY_ALIGN: body={self.body} pos_mm={np.linalg.norm(f['site_pos'] - f_ref['site_pos']) * 1000:.4f} "
            f"rot_deg={align.rot_err_deg(f['site_mat'], f_ref['site_mat']):.4f} quat_sign={self.sign:+.0f} "
            f"fixture_mm={fix_mm:.3f} missing={missing} "
            f"wrist_cam_rot_deg={align.rot_err_deg(f['wrist_cam_mat_in_site'], f_ref['wrist_cam_mat_in_site']):.3f}"
            + (f" hidden_geoms={n_hidden}" if self.hide_arm else "")
            + (f" recolored_geoms={n_recolored}" if self.recolor_arm else "")
            + f" copy_grip_cmd={int(self.copy_grip_cmd) if not self.grp else 1}"
            + (f" forearm_geoms_off={len(self.nc_geoms)}" if self.nc_geoms is not None else "")
        )
        if self.twin is not None:
            self._twin_start(fixtures)
            obs = self._twin_paint(obs)
            self._twin_t0(ref_obs, obs)
        return self._fix(obs)

    def _hide_arm(self):
        m = self.env.sim.model
        n = 0
        for g in range(m.ngeom):
            if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("robot0") and m.geom_rgba[g][3] > 0:
                m.geom_rgba[g][3] = 0.0
                n += 1
        return n

    def _no_forearm_contact(self):
        """UR5e_nc: contype = conaffinity = 0 on every collision geom of robot0_forearm_link.
        Returns [(geom, old contype, old conaffinity)] for the manipulation check (embody.contacts)."""
        m = self.env.sim.model
        out = []
        for g in range(m.ngeom):
            if (m.body_id2name(m.geom_bodyid[g]) or "") == "robot0_forearm_link" and (m.geom_contype[g] or m.geom_conaffinity[g]):
                out.append((g, int(m.geom_contype[g]), int(m.geom_conaffinity[g])))
                m.geom_contype[g] = 0
                m.geom_conaffinity[g] = 0
        assert out, "NO_FOREARM_CONTACT=1 but the body has no robot0_forearm_link collision geoms"
        return out

    RECOLOR_RGBA = (0.80, 0.25, 0.10, 1.0)

    def _recolor_arm(self):
        m = self.env.sim.model
        n = 0
        for g in range(m.ngeom):
            if (m.body_id2name(m.geom_bodyid[g]) or "").startswith("robot0") and m.geom_rgba[g][3] > 0:
                m.geom_matid[g] = -1
                m.geom_rgba[g] = self.RECOLOR_RGBA
                n += 1
        return n

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if self.twin is not None:
            obs = self._twin_paint(obs)
        return self._fix(obs), reward, done, info

    # ---- TWIN_RENDER: Panda pixels on target physics ----
    R85_PAD_BACK = 0.0064  # Robotiq pads sit 6.4 mm further back along site z than Panda pads
    # finger-pad centre z in the grip-site frame (pad_offset.py; Rethink by tools/calib_grippers.py)
    PAD_Z = {"panda": -0.0036, "r85": -0.0100, "rethink": None}

    def _pad_shift(self, phys, twin):
        """Site-z shift that puts the twin's pads onto the physical pads."""
        if phys == twin:
            return 0.0
        if (phys, twin) == ("r85", "panda"):
            return -self.R85_PAD_BACK  # block-2 value, kept bit-identical
        if (phys, twin) == ("panda", "r85"):
            return self.R85_PAD_BACK
        pz = dict(self.PAD_Z)
        pz["rethink"] = pz["rethink2"] = float(self._calib()["rt_pad_z"])
        return pz[phys] - pz[twin]

    def _twin_start(self, fixtures):
        align.apply_static_body_pose(self.twin, fixtures)
        tr = self.twin.robots[0]
        if self.twin_body == "Panda":
            # first guess and nullspace rest: the reference Panda's joints (the posture the policy saw)
            q0 = self.ref.sim.data.qpos[self.ref.robots[0]._ref_joint_pos_indexes].copy()
        else:
            q0 = np.asarray(tr.init_qpos, dtype=float)
        self._twin_q, self._twin_rest = q0.copy(), q0.copy()
        self._twin_last = None  # (eef_pos, agentview) at the previous paint
        if getattr(self, "twin_sunk", False):
            mt = self.twin.sim.model
            hand = mt.body_name2id(tr.robot_model.eef_name)
            roots = [b for b in range(mt.nbody) if (mt.body_id2name(b) or "").startswith("gripper0")
                     and not (mt.body_id2name(mt.body_parentid[b]) or "").startswith("gripper0")]
            assert len(roots) == 1 and mt.body_parentid[roots[0]] == hand, roots
            self._sunk_root, self._sunk_pos0 = roots[0], mt.body_pos[roots[0]].copy()
        self.twin_stats = {"pos_err_mm": [], "rot_err_deg": [], "posture_deg": [], "stale": 0, "t0": None}
        if getattr(self, "twin_sunk", False):
            # amendment 1, 3.1 (2026-10-31): per painted frame, the twin gripper root's offset from its nominal mount,
            # in the hand frame, mm [dx, dy, dz] (lateral = hypot(dx, dy)); index 0 = initial frame, k = after step k-1
            self.twin_stats["root_off_mm"] = []

    def _twin_paint(self, obs):
        tw, tr = self.twin, self.twin.robots[0]
        align.apply_object_state(tw, align.object_state_by_name(self.env))
        if getattr(self, "twin_sunk", False):
            return self._twin_paint_sunk(obs)
        p, R = align.site_pose(self.env.sim, self.env.robots[0].eef_site_id)
        if self.twin_dz:
            p = p + R @ np.array([0.0, 0.0, self.twin_dz])  # twin pads onto the physical pads
        gain = float(os.environ.get("TWIN_NS_GAIN", "0"))  # frozen at 0 by the CPU gain sweep, 2026-10-27
        extra = []
        if self.twin_seed_phys:
            # G_R: the Robotiq site must sit 48 - 6.4 = 41.6 mm behind where the Panda hand puts its own
            # site, so the twin hand is pulled back that far: warm start from the previous twin solution
            # (continuity), with the physical joints as rest posture and as the first retry seed
            q_phys = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy()
            if self._twin_q is None or self._twin_last is None:
                self._twin_q = q_phys
            self._twin_rest = q_phys
            extra = [q_phys]
        q, ep, er = align.ik_site_ns(tw, p, R, self._twin_q, self._twin_rest, gain=gain)
        if ep > 1e-3:  # warm start stuck (joint limits / local minimum): retry from other seeds
            seeds = extra + [self._twin_rest, np.asarray(tr.init_qpos, dtype=float)] + self._seed_postures(p)
            for seed in seeds:
                q2, ep2, er2 = align.ik_site_ns(tw, p, R, seed, self._twin_rest, gain=0.0)
                if ep2 < ep:
                    q, ep, er = q2, ep2, er2
                if ep < 1e-5:
                    break
            self.twin_stats["retries"] = self.twin_stats.get("retries", 0) + 1
        self._twin_q = q
        d = tw.sim.data
        d.qpos[tr._ref_joint_pos_indexes] = q
        g = np.asarray(obs["robot0_gripper_qpos"])
        d.qpos[tr._ref_gripper_joint_pos_indexes] = self._twin_fingers(g)
        d.qvel[:] = 0.0
        tw.sim.forward()
        if self.twin_seed_phys:
            self._twin_wrist_cam_to_physical()
        p_tw, R_tw = align.site_pose(tw.sim, tr.eef_site_id)
        st = self.twin_stats
        st["pos_err_mm"].append(float(np.linalg.norm(p_tw - p)) * 1000)
        st["rot_err_deg"].append(float(align.rot_err_deg(R_tw, R)))
        if self.body == "Panda" and panda_armed(self.twin_body):  # posture fidelity (same arm)
            q_phys = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes]
            st["posture_deg"].append(float(np.degrees(np.max(np.abs(q - q_phys)))))
        tw.env._update_observables(force=True)
        o = tw.env._get_observations()
        img = o["agentview_image"]
        if self._twin_last is not None:  # stale-frame guard
            moved = np.linalg.norm(obs["robot0_eef_pos"] - self._twin_last[0]) > 0.005
            if moved and np.array_equal(img, self._twin_last[1]):
                st["stale"] += 1
        self._twin_last = (np.array(obs["robot0_eef_pos"], copy=True), img.copy())
        if self.twin_cams in ("both", "agent"):
            obs["agentview_image"] = img
        if self.twin_cams in ("both", "wrist"):
            obs["robot0_eye_in_hand_image"] = o["robot0_eye_in_hand_image"]
        return obs

    def _pad_centre(self, sim, names):
        m, d = sim.model, sim.data
        return np.mean([d.geom_xpos[m.geom_name2id("gripper0_" + n)] for n in names], axis=0)

    def _twin_paint_sunk(self, obs):
        """G_R without IK: copy the physical Panda joints, Robotiq linkage from the kp100 table at the Panda
        opening fraction, gripper root shifted inside the hand so the pad centres coincide (shift from the
        calibrated pad-centre table, 42.2 mm open to 47.3 mm closed, plus the physical pads' lateral offset when the
        Panda fingers are asymmetric), wrist camera at the physical camera pose.
        twin_stats pos_err_mm = pad-centre residual (gate: > 1 mm on < 1% of query frames), rot_err_deg = hand
        orientation residual."""
        tw, tr = self.twin, self.twin.robots[0]
        c = self._calib()
        q_phys = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy()
        f = self._open_fraction_panda(self._panda_equiv(np.asarray(obs["robot0_gripper_qpos"])))
        fj = c["r85_fj"][0] + (1.0 - f) * (c["r85_fj"][-1] - c["r85_fj"][0])  # kp100 free-air open..closed
        fingers = np.array([np.interp(fj, c["r85_fj"], c["r85_q"][:, k]) for k in range(c["r85_q"].shape[1])])
        pad_r85 = np.array([np.interp(fj, c["r85_fj"], c["r85_pad_hand"][:, k]) for k in range(3)])
        mt, d = tw.sim.model, tw.sim.data
        # target: the physical Panda pad centre in the physical hand frame (the Panda fingers are not always
        # symmetric about the hand axis when one finger is blocked, so the free-air value is not enough)
        hb_p = self.env.sim.model.body_name2id(self.env.robots[0].robot_model.eef_name)
        R_hp = self.env.sim.data.xmat[hb_p].reshape(3, 3)
        p_ph = self._pad_centre(self.env.sim, ("finger1_pad_collision", "finger2_pad_collision"))
        pad_panda = R_hp.T @ (p_ph - self.env.sim.data.xpos[hb_p])
        mt.body_pos[self._sunk_root] = self._sunk_pos0 + (pad_panda - pad_r85)  # parent is the hand
        self.twin_stats["root_off_mm"].append([float(x) * 1000 for x in (pad_panda - pad_r85)])
        d.qpos[tr._ref_joint_pos_indexes] = q_phys
        d.qpos[tr._ref_gripper_joint_pos_indexes] = fingers
        d.qvel[:] = 0.0
        tw.sim.forward()
        self._twin_wrist_cam_to_physical()
        p_tw = self._pad_centre(tw.sim, ("left_fingerpad_collision", "right_fingerpad_collision"))
        hb_t = mt.body_name2id(tr.robot_model.eef_name)
        st = self.twin_stats
        st["pos_err_mm"].append(float(np.linalg.norm(p_tw - p_ph)) * 1000)
        st["rot_err_deg"].append(float(align.rot_err_deg(d.xmat[hb_t].reshape(3, 3), self.env.sim.data.xmat[hb_p].reshape(3, 3))))
        st["posture_deg"].append(0.0)
        tw.env._update_observables(force=True)
        o = tw.env._get_observations()
        img = o["agentview_image"]
        if self._twin_last is not None:
            moved = np.linalg.norm(obs["robot0_eef_pos"] - self._twin_last[0]) > 0.005
            if moved and np.array_equal(img, self._twin_last[1]):
                st["stale"] += 1
        self._twin_last = (np.array(obs["robot0_eef_pos"], copy=True), img.copy())
        if self.twin_cams in ("both", "agent"):
            obs["agentview_image"] = img
        if self.twin_cams in ("both", "wrist"):
            obs["robot0_eye_in_hand_image"] = o["robot0_eye_in_hand_image"]
        return obs

    def _twin_fingers(self, g):
        """Twin gripper joint qpos for the physical gripper's raw proprio g."""
        if self.twin_gk == "panda":
            return self._panda_equiv(g)  # identical to block 2 for Panda / Robotiq physics
        f = self._open_fraction_panda(self._panda_equiv(g))
        if self.twin_gk == "r85":
            c = self._calib()
            fj = self.R85_CLOSED * (1.0 - f)  # inverse of _grp_proprio
            return np.array([np.interp(fj, c["r85_fj"], c["r85_q"][:, k]) for k in range(c["r85_q"].shape[1])])
        if self.twin_gk in ("rethink", "rethink2"):
            c = self._calib()
            k = "rt2" if self.twin_gk == "rethink2" else "rt"
            return c[k + "_closed"] + f * (c[k + "_open"] - c[k + "_closed"])
        raise ValueError(self.twin_gk)

    def _twin_wrist_cam_to_physical(self):
        """Put the twin's wrist camera at the physical body's wrist-camera world pose (G_R)."""
        import mujoco
        import robosuite.utils.transform_utils as T
        tw = self.twin
        mt, dt = tw.sim.model, tw.sim.data
        cam_t = mt.camera_name2id("robot0_eye_in_hand")
        cam_p = self.env.sim.model.camera_name2id("robot0_eye_in_hand")
        p_w = self.env.sim.data.cam_xpos[cam_p].copy()
        R_w = self.env.sim.data.cam_xmat[cam_p].reshape(3, 3).copy()
        b = mt.cam_bodyid[cam_t]
        R_b = dt.xmat[b].reshape(3, 3)
        mt.cam_pos[cam_t] = R_b.T @ (p_w - dt.xpos[b])
        q = T.mat2quat(R_b.T @ R_w)  # xyzw
        mt.cam_quat[cam_t] = np.array([q[3], q[0], q[1], q[2]])
        mujoco.mj_camlight(mt._model, dt._data)

    _SEED_LIB = None

    def _seed_postures(self, p, k=8):
        """Panda postures that the SRC runs actually reached, nearest grip-site position first.
        Only a Panda twin uses them (TWIN_SEED_DIR, default $PROJ_ROOT/runs/src)."""
        if not (self.twin_body == "Panda" or self.twin_seed_phys):
            return []
        if BodyEnv._SEED_LIB is None:
            import glob
            d = os.path.expanduser(os.environ.get("TWIN_SEED_DIR", "$PROJ_ROOT/runs/src"))
            P, Q = [], []
            for f in sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz"))):
                z = np.load(f, allow_pickle=True)
                names = [str(x) for x in z["joint_names"]]
                if names[:7] != [f"robot0_joint{i}" for i in range(1, 8)]:
                    continue
                n = min(len(z["eef_pos"]), len(z["qpos"]))
                P.append(z["eef_pos"][:n])
                Q.append(z["qpos"][:n, :7])
            BodyEnv._SEED_LIB = (np.concatenate(P), np.concatenate(Q)) if P else (np.zeros((0, 3)), np.zeros((0, 7)))
            self.log(f"TWIN_SEED_LIB: {len(BodyEnv._SEED_LIB[0])} postures from {d}")
        P, Q = BodyEnv._SEED_LIB
        if not len(P):
            return []
        idx = np.argsort(np.linalg.norm(P - p, axis=1))[:k]
        return [Q[i].astype(float) for i in idx]

    def _twin_t0(self, ref_obs, obs):
        """Step-0 identity: the twin's first frame against the reference Panda's own frame."""
        out = {}
        for k in ("agentview_image", "robot0_eye_in_hand_image"):
            a = np.asarray(ref_obs[k], dtype=np.int16)
            b = np.asarray(obs[k], dtype=np.int16)
            out[k] = (int(np.abs(a - b).max()), float(np.abs(a - b).mean()))
        self.twin_stats["t0"] = out
        self.log(f"TWIN_T0: twin={self.twin_body} agent_max={out['agentview_image'][0]} "
                 f"agent_mean={out['agentview_image'][1]:.4f} wrist_max={out['robot0_eye_in_hand_image'][0]} "
                 f"wrist_mean={out['robot0_eye_in_hand_image'][1]:.4f}")

    def _fix(self, obs):
        obs["robot0_eef_quat"] = obs["robot0_eef_quat"] * self.sign
        if self.grp:
            obs["robot0_gripper_qpos"] = self._panda_equiv(obs["robot0_gripper_qpos"])
            if self.pad_dz:  # report the Panda-equivalent grip-site position
                _, R = align.site_pose(self.env.sim, self.env.robots[0].eef_site_id)
                obs["robot0_eef_pos"] = obs["robot0_eef_pos"] - R @ np.array([0.0, 0.0, self.pad_dz])
        return obs

    # ---- foreign grippers ----
    # Endpoint-matched linear map, calibrated in free air (grp_check2, 2026-10-26):
    # Panda fingers reach 0.039 open and ~0.0005 closed; Robotiq finger_joint goes from 0 (open)
    # to a free-air plateau of ~0.50 rad (fingertips meet), not the 0.8 joint limit.
    PANDA_OPEN = 0.039
    R85_CLOSED = 0.50

    def _panda_equiv(self, g):
        if self.gk == "r85":
            return self._grp_proprio(g)
        if self.gk in ("rethink", "rethink2"):
            return self._rt_proprio(g)
        return g[:2]

    def _open_fraction_panda(self, p2):
        return float(np.clip((p2[0] - p2[1]) / (2 * self.PANDA_OPEN), 0.0, 1.0))

    def _grp_proprio(self, q6):
        f = float(np.clip(1.0 - q6[0] / self.R85_CLOSED, 0.0, 1.0))  # opening fraction
        return np.array([self.PANDA_OPEN * f, -self.PANDA_OPEN * f])

    def _rt_proprio(self, q2):
        """Rethink [l, r] slide joints -> Panda-equivalent by opening fraction of the l finger
        (free-air endpoints from tools/calib_grippers.py; the fingers are mirrored)."""
        c = self._calib()
        k = "rt2" if self.gk == "rethink2" else "rt"
        lo, hi = float(c[k + "_closed"][0]), float(c[k + "_open"][0])
        f = float(np.clip((q2[0] - lo) / (hi - lo), 0.0, 1.0))
        return np.array([self.PANDA_OPEN * f, -self.PANDA_OPEN * f])

    _CALIB = None

    @classmethod
    def _calib(cls):
        if cls._CALIB is None:
            path = os.environ.get("GRIPPER_CALIB", os.path.join(_HERE, "gripper_calib.npz"))
            cls._CALIB = dict(np.load(path))
        return cls._CALIB

    def _grp_init(self, f_ref):
        """Match the reference gripper's command state, let the foreign gripper settle under it,
        then move the wrist camera. Arm, objects and fixtures are restored after the gripper settles."""
        env, robot = self.env, self.env.robots[0]
        sim, m, d = env.sim, env.sim.model, env.sim.data
        ref_g = self.ref.robots[0].gripper
        if self.gk == "r85":
            closedness = float(np.asarray(ref_g.current_action).reshape(-1)[-1])
            robot.gripper.current_action = np.array([closedness])
        else:  # Rethink: same 2-vector command, opposite sign convention to Panda ([1,-1] closes)
            robot.gripper.current_action = -np.array(ref_g.current_action, dtype=float)
        cmd = np.asarray(robot.gripper.current_action, dtype=float)
        # settle the gripper joints under their own actuators at the matching command
        act = [i for i in range(m.nu) if (m.actuator_id2name(i) or "").startswith("gripper0")]
        kp = os.environ.get("GRP_KP")
        if kp and self.gk == "r85":  # sensitivity: stiffer Robotiq position actuators (stock kp=20; Panda fingers use 1000)
            for i in act:
                m.actuator_gainprm[i, 0] = float(kp)
                m.actuator_biasprm[i, 1] = -float(kp)
        lo, hi = m.actuator_ctrlrange[act, 0], m.actuator_ctrlrange[act, 1]
        target = (lo + hi) / 2 + (hi - lo) / 2 * (cmd[-1] if self.gk == "r85" else cmd)
        keep_q = d.qpos.copy()
        gidx = np.array(robot._ref_gripper_joint_pos_indexes)
        gvel = np.array(robot._ref_gripper_joint_vel_indexes)
        for _ in range(300):
            d.ctrl[:] = 0.0
            d.ctrl[act] = target
            sim.step()
            g = d.qpos[gidx].copy()
            d.qpos[:] = keep_q
            d.qpos[gidx] = g
            nz = np.setdiff1d(np.arange(m.nv), gvel)
            d.qvel[nz] = 0.0
        d.qvel[:] = 0.0
        sim.forward()
        # wrist camera: same pose in the grip-site frame as on Panda
        cam = m.camera_name2id("robot0_eye_in_hand")
        body = m.cam_bodyid[cam]
        p_site, R_site = align.site_pose(sim, robot.eef_site_id)
        p_cam = p_site + R_site @ f_ref["wrist_cam_pos_in_site"]
        R_b = d.xmat[body].reshape(3, 3)
        m.cam_pos[cam] = R_b.T @ (p_cam - d.xpos[body])
        sim.forward()
        env.check_success()
        env._post_process()
        env._update_observables(force=True)

    def close(self):
        self.env.close()
        self.ref.close()
        if self.twin is not None:
            self.twin.close()

    def __getattr__(self, name):
        return getattr(self.env, name)
