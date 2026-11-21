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
TOY1 options (2026-11-03, $PROJ_ROOT/embody_toy only; every one is off by default and then the code path is the
embody_v3 one). Physical body PandaR85 (Robotiq), twin Panda (TWIN_RENDER=1):
  TOY_PARTS=hand|fingers|palm: instead of the whole twin Panda, composite only these Panda gripper parts onto the
    PHYSICAL render (physical arm, physical Robotiq per TOY_PHYS_GRIPPER). palm = gripper bodies with no joint between
    them and the gripper root (Panda gripper0_right_gripper; Robotiq adapter link), fingers = the jointed ones (Panda
    fingers; Robotiq knuckles and fingers), hand = both. Compositing is per pixel: twin render with every robot geom
    hidden except the chosen parts (scene kept for occlusion), twin segmentation gives the part pixels, a part pixel
    replaces the physical pixel only where its metric depth is nearer than the physical depth. Hidden geoms are moved
    to geom group 3 (not drawn, no shadow, physics untouched) only for the duration of these renders.
  TOY_PHYS_GRIPPER=hide_same (default) | keep | hide_all: which physical Robotiq parts are hidden in the composite
    (hide_same = the same category as the painted Panda parts).
  TOY_TWIN_POSE=auto (default) | ik | flange: ik = the G_T pose (twin pads on the Robotiq pads, Panda hand 41.6 mm
    in front of the physical hand); flange = twin joints copied from the physical Panda arm (Panda hand on the
    physical flange, Panda pads 41.6 mm short of the Robotiq pads). auto = flange for palm, ik otherwise.
  TOY_WRIST_CAM= (default) | phys | back: phys = the twin's wrist camera is put at the physical wrist-camera world
    pose (the G pipeline pose, = where a Panda whose grip site sits on the Robotiq grip site has its camera), i.e. it
    does not follow the twin 6.4 mm back; back = the physical wrist camera itself is moved TOY_CAM_BACK_MM (6.4) back
    along the grip-site z axis (on G without twin: original Robotiq look, camera as in published G_T). With TOY_PARTS
    the twin camera always follows the physical camera (both renders must share it); default there = phys.
  TOY_TIP_SHIFT_MM="x,y,z": render-only translation of the twin Panda finger bodies in the hand frame (y = finger
    opening axis); also settable per frame with set_tip_shift() (tools/tip_probe.py).
  TOY_PIXSTATS=1: per painted frame, visible gripper pixel counts (self.toy_stats["px"], rows
    [agent_panda, agent_robotiq, wrist_panda, wrist_robotiq]); extra segmentation renders, images unchanged.
protocol_toy1.md section 9 names (2026-11-03, embody_toy only; all off by default):
  TWIN_CAM=twin (default, published) | phys: phys puts the twin wrist camera at the physical wrist-camera world pose
    every paint, for any TWIN_BODY (G_T^fc). Per paint twin_stats["cam_err_mm"/"cam_err_deg"] = twin vs physical wrist
    camera world pose (recorded for every twin cell).
  HIDE_GRIPPER=all|fingers|base: alpha 0 on the visible geoms of that gripper body group of the PHYSICAL env, every
    episode (G_hid, S_hid). Groups (printed, align_info["hidden_gripper_geoms"]): base = gripper bodies with no joint
    between them and the gripper root (Panda hand; Robotiq adapter/base), fingers = the jointed ones (Panda fingers;
    Robotiq knuckles, fingers, pads), all = both. Collision flags are untouched.
  GRP_MAP=kp20 (default, published: finger_joint 0 -> open, 0.50 -> closed) | kp100 (0 -> open, kp100 free-air
    plateau 0.697 from gripper_calib.npz r85_fj[-1] -> closed); Robotiq proprio and the Panda twin finger opening.
  WRIST_CAM_DZ_MM=d: after the published re-mount in _grp_init, the physical wrist camera is translated d mm along the
    grip-site z axis (G_cam: -6.4 = back). TOY_WRIST_CAM=back is the same with d = -TOY_CAM_BACK_MM.
  TWIN_MOUNT=copy (probes): twin body == physical body, twin arm AND gripper joints copied from the physical env (no
    IK); render-only offsets via set_probe(): robot_shift (whole twin robot, world m), grip_shift (twin gripper root,
    world m), cam_shift (wrist camera, world m, on top of TWIN_CAM=phys).
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
            self.twin_copy = os.environ.get("TWIN_MOUNT", "ik") == "copy"  # toy1 probes
            if self.twin_copy:
                assert self.twin_body == body, "TWIN_MOUNT=copy needs TWIN_BODY == BODY"
            if self.twin_sunk:
                assert self.twin_seed_phys and self.twin_gk == "r85", "TWIN_MOUNT=sunk is for TWIN_BODY=PandaR85 on Panda physics"
        # GRP_PAD_ALIGN=1: sensitivity variant, Robotiq pad centre placed where Panda's pad centre is
        # (site-frame z: Panda pad -3.6 mm, Robotiq pad -10.0 mm; measured by pad_offset.py)
        self.pad_dz = 0.0064 if (os.environ.get("GRP_PAD_ALIGN") == "1" and self.gk == "r85") else 0.0
        self.toy_res = resolution
        self._toy_init()
        self._pg_init()

    # ---- phasegate (exploration, 2026-11-10): image intervention inside one task window, see phasegate.py ----
    PG_IMG = ("agentview_image", "robot0_eye_in_hand_image")

    def _pg_init(self):
        self.pg_mode = os.environ.get("PG_MODE", "")
        self.pg = None
        if not self.pg_mode:
            return
        from . import phasegate
        assert self.pg_mode in phasegate.MODES, self.pg_mode
        assert self.body == "Panda" and not self.hide_gripper and not self.hide_arm, "phasegate runs on the stock Panda"
        if self.pg_mode == "look":
            assert self.twin is not None and self.twin_sunk and self.twin_cams == "both", \
                "PG_MODE=look needs TWIN_RENDER=1 TWIN_BODY=PandaR85 TWIN_MOUNT=sunk (both cameras)"
        else:
            assert self.twin is None, "PG_MODE=erase runs without a twin"
        self.pg = phasegate.PhaseGate(os.environ["PG_WINDOW"], float(os.environ.get("PG_LIFT_M", "0.03")),
                                      yoke=bool(os.environ.get("PG_YOKE_DIR")))
        self._pg_z = 0.0
        self.log(f"PHASEGATE: mode={self.pg_mode} window={self.pg.window} lift_m={self.pg.lift_m}"
                 + (f" yoke_dir={os.environ['PG_YOKE_DIR']}" if self.pg.yoke else ""))

    def pg_set_yoke(self, t_close, t_lift, src=""):
        """phasegate_yoke: fixed window for the next episode (source event times); call before set_init_state."""
        assert self.pg is not None and self.pg.yoke, "pg_set_yoke needs PG_MODE and PG_YOKE_DIR"
        self.pg.set_yoke(t_close, t_lift, src)

    def _pg_alpha(self, values=None):
        """values None: set every gripper geom to alpha 0 and return the old values; else restore them."""
        m = self.env.sim.model
        palm, fingers = self._grip_parts(m)
        sel = palm + fingers
        if values is None:
            values = [float(m.geom_rgba[g][3]) for g in sel]
            for g in sel:
                m.geom_rgba[g][3] = 0.0
            return values
        for g, a in zip(sel, values):
            m.geom_rgba[g][3] = a

    def _pg_first(self, obs):
        """Initial observation (static state). erase: forced re-render with the gripper at alpha 0; look: twin."""
        gate = self.pg.active()
        if self.pg_mode == "look":
            painted = self._twin_paint(dict(obs))
            if gate:
                for k in self.PG_IMG:
                    obs[k] = painted[k]
        elif gate:
            # direct renders of the two cameras: a forced observable update here would reset the cameras' sampling
            # phase and shift every later frame of the episode by a substep (found by the smoke, 2026-11-10)
            old = self._pg_alpha()
            try:
                sim, res = self.env.sim, self.toy_res
                for k, cam in zip(self.PG_IMG, ("agentview", "robot0_eye_in_hand")):
                    img = np.asarray(sim.render(camera_name=cam, width=res, height=res, depth=False))
                    assert img.shape == np.asarray(obs[k]).shape, (img.shape, np.asarray(obs[k]).shape)
                    obs[k] = np.array(img, copy=True)
            finally:
                self._pg_alpha(old)
        return obs

    def _pg_step(self, action):
        """One env step with the gate decided BEFORE the step (from the action and the hand height of the previous
        observation), so the erased image is rendered by the normal observation pipeline at the normal instant
        (same timing as HIDE_GRIPPER=all); a forced re-render after the step would show a later substep."""
        gate = self.pg.update(action, self._pg_z)
        old = self._pg_alpha() if (gate and self.pg_mode == "erase") else None
        try:
            obs, reward, done, info = self.env.step(action)
        finally:
            if old is not None:
                self._pg_alpha(old)
        self._pg_z = float(obs["robot0_eef_pos"][2])
        if self.pg_mode == "look":  # painted every step (twin statistics stay per-step); used only when the gate is open
            painted = self._twin_paint(dict(obs))
            if gate:
                for k in self.PG_IMG:
                    obs[k] = painted[k]
        return self._fix(obs), reward, done, info

    def pg_fields(self):
        if self.pg is None:
            return {}
        return self.pg.fields(self.pg_mode)

    # ---- TOY1 (2026-11-03) ----
    def _toy_init(self):
        e = os.environ
        self.twin_cam = e.get("TWIN_CAM", "twin")
        assert self.twin_cam in ("twin", "phys"), self.twin_cam
        self.grp_map = e.get("GRP_MAP", "kp20")
        assert self.grp_map in ("kp20", "kp100"), self.grp_map
        self.hide_gripper = e.get("HIDE_GRIPPER", "")
        assert self.hide_gripper in ("", "all", "fingers", "base"), self.hide_gripper
        self.wrist_dz = float(e["WRIST_CAM_DZ_MM"]) / 1000.0 if e.get("WRIST_CAM_DZ_MM") else 0.0
        if self.grp_map != "kp20" or self.wrist_dz:
            assert self.gk == "r85", "GRP_MAP / WRIST_CAM_DZ_MM are for the Robotiq body"
        if self.twin_cam == "phys":
            assert self.twin is not None, "TWIN_CAM=phys needs TWIN_RENDER=1"
        self.probe = {"robot_shift": None, "grip_shift": None, "cam_shift": None}
        self._probe_pos0 = {}
        if self.gk == "r85":
            self.log(self.grp_map_selfcheck())
        self.toy_parts = e.get("TOY_PARTS", "")
        self.toy_phys = e.get("TOY_PHYS_GRIPPER", "hide_same")
        self.toy_pose = e.get("TOY_TWIN_POSE", "auto")
        self.toy_cam = e.get("TOY_WRIST_CAM", "")
        self.toy_back = float(e.get("TOY_CAM_BACK_MM", "6.4")) / 1000.0
        self.toy_pixstats = e.get("TOY_PIXSTATS") == "1"
        sh = e.get("TOY_TIP_SHIFT_MM", "")
        self.toy_tip_shift = np.array([float(x) for x in sh.split(",")]) / 1000.0 if sh else None
        assert self.toy_parts in ("", "hand", "fingers", "palm"), self.toy_parts
        assert self.toy_phys in ("hide_same", "keep", "hide_all"), self.toy_phys
        assert self.toy_pose in ("auto", "ik", "flange"), self.toy_pose
        assert self.toy_cam in ("", "phys", "back"), self.toy_cam
        assert self.toy_tip_shift is None or self.toy_tip_shift.shape == (3,), sh
        if self.toy_pose == "auto":
            self.toy_pose = "flange" if self.toy_parts == "palm" else "ik"
        if self.toy_parts or self.toy_pose == "flange" or self.toy_tip_shift is not None:
            assert self.twin is not None and self.twin_gk == "panda" and not self.twin_sunk, "TOY_* twin options need TWIN_RENDER=1 with a Panda twin"
        if self.toy_parts or self.toy_cam:
            assert self.gk == "r85", "TOY_PARTS / TOY_WRIST_CAM are written for the Robotiq (PandaR85) physical body"
        if self.toy_pose == "flange":
            assert panda_armed(self.body), "TOY_TWIN_POSE=flange copies Panda-arm joints"
        if self.toy_cam == "back":
            self.wrist_dz = -self.toy_back
        if self.toy_cam == "phys":
            self.twin_cam = "phys"
        self.toy_on = bool(self.toy_parts or self.toy_cam or self.toy_pixstats or self.toy_tip_shift is not None
                           or self.toy_pose == "flange" or self.twin_cam != "twin" or self.grp_map != "kp20"
                           or self.hide_gripper or self.wrist_dz or (self.twin is not None and self.twin_copy))
        self._tip_pos0 = None
        self.toy_stats = None
        self.toy_info = None if not self.toy_on else {
            "parts": self.toy_parts, "phys_gripper": self.toy_phys if self.toy_parts else "", "twin_pose": self.toy_pose,
            "wrist_cam": self.toy_cam or ("phys" if self.toy_parts else "pipeline"), "cam_back_mm": self.toy_back * 1000,
            "tip_shift_mm": None if self.toy_tip_shift is None else (self.toy_tip_shift * 1000).tolist(),
            "pixstats": self.toy_pixstats, "twin_cam": self.twin_cam, "grp_map": self.grp_map if self.gk == "r85" else None,
            "hide_gripper": self.hide_gripper, "wrist_cam_dz_mm": self.wrist_dz * 1000,
            "twin_mount": os.environ.get("TWIN_MOUNT", "ik") if self.twin is not None else None}

    def grp_map_selfcheck(self):
        """protocol_toy1 8.5: proprio at finger_joint 0, 0.35, 0.70 under both maps."""
        out = []
        for q in (0.0, 0.35, 0.70):
            g = np.zeros(6)
            g[0] = q
            out.append(f"q={q:.2f} kp20={np.round(self._grp_proprio(g, 'kp20'), 5).tolist()} "
                       f"kp100={np.round(self._grp_proprio(g, 'kp100'), 5).tolist()}")
        return f"GRP_MAP selfcheck (active={self.grp_map}, kp100 closed={self._kp100_closed():.4f}): " + "; ".join(out)

    def _kp100_closed(self):
        return float(self._calib()["r85_fj"][-1])

    def set_probe(self, robot_shift=None, grip_shift=None, cam_shift=None, grip_shift_hand=None, obj_shift=None, cams="both"):
        """Render-only probe offsets for the twin; None = off. Applied at the next paint. robot_shift, grip_shift,
        cam_shift: world vectors (m), converted to the parent frame at each paint (world direction fixed);
        grip_shift_hand: offset of the twin gripper root in its parent (hand) frame (moves with the hand; toy2 forks);
        obj_shift: (object joint name, world vector) render-only translation of one free object in the twin (toy2
        positive control). cams = "both" | "agent" | "wrist" (toy2 amendment): the offsets appear only in that camera;
        the other camera gets the same-appearance twin rendered WITHOUT the offsets (never the physical image)."""
        assert cams in ("both", "agent", "wrist"), cams
        f = lambda v: None if v is None else np.asarray(v, float)
        self.probe = {"robot_shift": f(robot_shift), "grip_shift": f(grip_shift), "cam_shift": f(cam_shift),
                      "grip_shift_hand": f(grip_shift_hand), "obj_shift": obj_shift, "cams": cams}

    def _hide_gripper(self):
        """HIDE_GRIPPER: alpha 0 on the visible geoms of the chosen gripper group of the physical env."""
        m = self.env.sim.model
        base, fingers = self._grip_parts(m)
        sel = {"all": base + fingers, "fingers": fingers, "base": base}[self.hide_gripper]
        names = []
        for g in sel:
            if m.geom_rgba[g][3] > 0:
                m.geom_rgba[g][3] = 0.0
                names.append(m.geom_id2name(g) or f"geom{g}")
        return names

    def set_tip_shift(self, shift_m):
        """Render-only translation (m, hand frame) of the twin Panda fingers from the next paint on; None = off."""
        self.toy_tip_shift = None if shift_m is None else np.asarray(shift_m, dtype=float).reshape(3)

    @staticmethod
    def _grip_parts(m):
        """gripper0 geoms split into palm (no joint between the body and the gripper root) and fingers."""
        palm, fingers = [], []
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if not (m.body_id2name(b) or "").startswith("gripper0"):
                continue
            moving, bb = False, b
            while (m.body_id2name(bb) or "").startswith("gripper0"):
                if m.body_jntnum[bb] > 0:
                    moving = True
                    break
                bb = int(m.body_parentid[bb])
            (fingers if moving else palm).append(g)
        return palm, fingers

    @staticmethod
    def _robot_geoms(m):
        return [g for g in range(m.ngeom)
                if (m.body_id2name(m.geom_bodyid[g]) or "").startswith(("robot0", "gripper0", "mount0"))]

    def _toy_sets(self):
        """(twin geoms to show, twin geoms to hide, physical geoms to hide), recomputed per episode (reset may rebuild)."""
        mt, mp = self.twin.sim.model, self.env.sim.model
        tp, tf = self._grip_parts(mt)
        show = {"hand": tp + tf, "fingers": tf, "palm": tp}[self.toy_parts]
        hide_t = sorted(set(self._robot_geoms(mt)) - set(show))
        pp, pf = self._grip_parts(mp)
        hide_p = {"keep": [], "hide_all": pp + pf,
                  "hide_same": {"hand": pp + pf, "fingers": pf, "palm": pp}[self.toy_parts]}[self.toy_phys]
        return sorted(show), hide_t, sorted(hide_p)

    def _render(self, sim, cam, hide, depth=True, seg=False, how="group"):
        """RGB (+ metric depth) (+ per-pixel geom id, -1 = none) of one camera with the geoms in `hide` moved to
        geom group 3 (how="group") or made alpha 0 (how="alpha", the protocol_toy1 / HIDE_GRIPPER way) for this
        render only."""
        from robosuite.utils.camera_utils import get_real_depth_map
        m = sim.model
        ctx = sim._render_context_offscreen
        assert ctx.vopt.geomgroup[3] == 0, "geom group 3 must be invisible"
        hide = list(hide)
        old = (m.geom_group[hide].copy() if how == "group" else m.geom_rgba[hide, 3].copy()) if len(hide) else None
        if len(hide):
            if how == "group":
                m.geom_group[hide] = 3
            else:
                m.geom_rgba[hide, 3] = 0.0
        try:
            W = H = self.toy_res
            rgb, d = sim.render(width=W, height=H, camera_name=cam, depth=True)
            out = [rgb, get_real_depth_map(sim, d) if depth else None]
            if seg:
                ctx.render(width=W, height=H, camera_id=m.camera_name2id(cam), segmentation=True)
                raw = ctx.read_pixels(W, H, segmentation=False).astype(np.int64)
                sid = raw[:, :, 0] + raw[:, :, 1] * 256 + raw[:, :, 2] * 65536  # segid + 1, 0 = background
                lut = np.full(ctx.scn.ngeom + 2, -1, dtype=np.int64)
                for i in range(ctx.scn.ngeom):
                    gg = ctx.scn.geoms[i]
                    if gg.segid != -1 and int(gg.objtype) == 5:  # mjOBJ_GEOM
                        lut[gg.segid + 1] = gg.objid
                sid[(sid < 0) | (sid >= len(lut))] = 0
                out.append(lut[sid])
            return out
        finally:
            if old is not None:
                if how == "group":
                    m.geom_group[hide] = old
                else:
                    m.geom_rgba[hide, 3] = old

    CAMS = (("agentview_image", "agentview"), ("robot0_eye_in_hand_image", "robot0_eye_in_hand"))

    def _toy_composite(self, obs):
        """TOY_PARTS: chosen Panda parts (twin) over the physical render, by metric depth."""
        show, hide_t, hide_p = self._toy_sets()
        px = []
        for ok, cam in self.CAMS:
            use = self.twin_cams in ("both", "agent") if cam == "agentview" else self.twin_cams in ("both", "wrist")
            rgb_p, dep_p, gid_p = self._render(self.env.sim, cam, hide_p, seg=True)
            rgb_t, dep_t, gid_t = self._render(self.twin.sim, cam, hide_t, seg=True)
            mask = np.isin(gid_t, show) & (dep_t < dep_p)
            img = np.where(mask[..., None], rgb_t, rgb_p)
            if use:
                obs[ok] = img
                pp, pf = self._grip_parts(self.env.sim.model)
                grip_p = np.isin(gid_p, pp + pf)  # physical Robotiq geoms still drawn (not hidden)
                px += [int(mask.sum()), int((grip_p & ~mask).sum())]
            else:  # this camera keeps the plain physical image (not counted here)
                px += [-1, -1]
        if self.toy_pixstats:
            self.toy_stats["px"].append(px)
        return obs

    def _toy_pix_plain(self, obs):
        """TOY_PIXSTATS without TOY_PARTS: gripper pixel counts of the images the policy gets (segmentation of the
        sim that produced each camera image)."""
        px = []
        for ok, cam in self.CAMS:
            twin_cam = self.twin is not None and (self.twin_cams in ("both", "agent") if cam == "agentview" else self.twin_cams in ("both", "wrist"))
            sim = self.twin.sim if twin_cam else self.env.sim
            p, f = self._grip_parts(sim.model)
            gid = self._render(sim, cam, [], depth=False, seg=True)[2]
            n = int(np.isin(gid, p + f).sum())
            panda = (self.twin_gk if twin_cam else self.gk) == "panda"
            px += [n, 0] if panda else [0, n]
        self.toy_stats["px"].append(px)

    def _toy_apply_tip_shift(self):
        if self.toy_tip_shift is None and self._tip_pos0 is None:
            return
        m = self.twin.sim.model
        names = ("gripper0_leftfinger", "gripper0_rightfinger")
        if self._tip_pos0 is None:
            self._tip_pos0 = {n: m.body_pos[m.body_name2id(n)].copy() for n in names}
        sh = np.zeros(3) if self.toy_tip_shift is None else self.toy_tip_shift
        for n in names:
            b = m.body_name2id(n)
            assert m.body_id2name(m.body_parentid[b]) == "gripper0_right_gripper"
            m.body_pos[b] = self._tip_pos0[n] + sh

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
        hidden_grip = self._hide_gripper() if self.hide_gripper else None
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
        if self.toy_on:
            self.align_info["toy"] = self.toy_info
            self.align_info["grp_map"] = self.grp_map if self.gk == "r85" else None
            self.align_info["twin_cam"] = self.twin_cam if self.twin is not None else None
            self.align_info["hide_gripper"] = self.hide_gripper or None
            self.align_info["hidden_gripper_geoms"] = hidden_grip
            self.align_info["wrist_cam_dz_mm"] = self.wrist_dz * 1000
            if self.gk == "r85":
                cp = self.env.sim.model.camera_name2id("robot0_eye_in_hand")
                self.align_info["wrist_cam_world"] = self.env.sim.data.cam_xpos[cp].tolist() + self.env.sim.data.cam_xmat[cp].tolist()
            self.log(f"TOY1: {self.toy_info} grp_map={self.align_info['grp_map']} hidden_gripper_geoms={hidden_grip} "
                     f"wrist_cam_world={self.align_info.get('wrist_cam_world')}")
            self.toy_stats = {"px": []}
        if self.pg is not None:
            if self.twin is not None:
                self._twin_start(fixtures)
            self._pg_z = float(obs["robot0_eef_pos"][2])
            self.pg.reset(self._pg_z)
            obs = self._pg_first(obs)
            if self.twin is not None:
                self._twin_t0(ref_obs, obs)
            return self._fix(obs)
        if self.twin is not None:
            self._twin_start(fixtures)
            obs = self._twin_paint(obs)
            self._twin_t0(ref_obs, obs)
        elif self.toy_pixstats:
            self._toy_pix_plain(obs)
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
        if self.pg is not None:
            return self._pg_step(action)
        obs, reward, done, info = self.env.step(action)
        if self.twin is not None:
            obs = self._twin_paint(obs)
        elif self.toy_pixstats:
            self._toy_pix_plain(obs)
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
        """TWIN_CAM_SYNC=1 (toy2): after mj_step the physical xpos / site_xpos / cam_xpos lag qpos by one substep, while
        the twin is placed from physical qpos (objects, copy mount) and forwarded; the paint then mixes two instants.
        With the flag the physical data are forwarded for the duration of the paint (so the IK target, pad centres,
        camera pose and object states all refer to the qpos instant) and restored bit for bit afterwards; physics is
        untouched. Off (default): identical to toy1 / earlier batches."""
        if os.environ.get("TWIN_CAM_SYNC") != "1":
            return self._twin_paint_inner(obs)
        import copy as _copy
        from . import snapshot as _snap
        d = self.env.sim.data
        keep = _copy.copy(d._data)
        self.env.sim.forward()
        try:
            return self._twin_paint_inner(obs)
        finally:
            _snap._copy_mjdata(d._data, keep)

    def _twin_paint_inner(self, obs):
        tw, tr = self.twin, self.twin.robots[0]
        align.apply_object_state(tw, align.object_state_by_name(self.env))
        if self.probe.get("obj_shift") is not None:  # toy2 positive control: render-only object translation
            jn, vec = self.probe["obj_shift"]
            a = tw.sim.model.get_joint_qpos_addr(jn)
            tw.sim.data.qpos[a[0]:a[0] + 3] += np.asarray(vec, float)
        if getattr(self, "twin_sunk", False):
            return self._twin_paint_sunk(obs)
        if getattr(self, "twin_copy", False):
            return self._twin_paint_copy(obs)
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
        flange = self.toy_pose == "flange"
        if self._probe_pos0:  # toy1 probe: IK must see the un-shifted gripper (offsets are re-applied after IK)
            for b_, p0_ in self._probe_pos0.items():
                tw.sim.model.body_pos[b_] = p0_
        if getattr(self, "probe_fix_q", None) is not None:  # toy1 probe: same twin arm for every condition of a frame
            q, ep, er = np.array(self.probe_fix_q, copy=True), 0.0, 0.0
        elif flange:  # TOY1: twin joints = physical Panda-arm joints (Panda hand on the physical flange)
            q, ep, er = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes].copy(), 0.0, 0.0
        else:
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
        if self.toy_on:
            self._toy_apply_tip_shift()
        tw.sim.forward()
        if self.probe["grip_shift"] is not None or self.probe.get("grip_shift_hand") is not None or self._probe_pos0:  # toy1 probe: move only the twin gripper root
            mt = tw.sim.model
            root = [b for b in range(mt.nbody) if (mt.body_id2name(b) or "").startswith("gripper0")
                    and not (mt.body_id2name(mt.body_parentid[b]) or "").startswith("gripper0")]
            self._probe_body_pos(mt.body_id2name(root[0]), self.probe["grip_shift"], self.probe.get("grip_shift_hand"))
            tw.sim.forward()
        if self.twin_seed_phys or self.toy_parts or self.twin_cam == "phys":
            self._twin_wrist_cam_to_physical(self.probe["cam_shift"])
        self._twin_cam_err()
        p_tw, R_tw = align.site_pose(tw.sim, tr.eef_site_id)
        st = self.twin_stats
        if flange:  # residual of the twin hand against the physical hand
            hb_t = tw.sim.model.body_name2id(tr.robot_model.eef_name)
            hb_p = self.env.sim.model.body_name2id(self.env.robots[0].robot_model.eef_name)
            p_tw, p = tw.sim.data.xpos[hb_t], self.env.sim.data.xpos[hb_p]
            R_tw, R = tw.sim.data.xmat[hb_t].reshape(3, 3), self.env.sim.data.xmat[hb_p].reshape(3, 3)
        st["pos_err_mm"].append(float(np.linalg.norm(p_tw - p)) * 1000)
        st["rot_err_deg"].append(float(align.rot_err_deg(R_tw, R)))
        if self.body == "Panda" and panda_armed(self.twin_body):  # posture fidelity (same arm)
            q_phys = self.env.sim.data.qpos[self.env.robots[0]._ref_joint_pos_indexes]
            st["posture_deg"].append(float(np.degrees(np.max(np.abs(q - q_phys)))))
        if self.toy_parts:
            return self._toy_paint_composite(obs)
        o = self._twin_obs_probe()
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
        if self.toy_pixstats:
            self._toy_pix_plain(obs)
        return obs

    def _phys_cam_pose(self):
        """Physical wrist-camera world pose as stored in the data (see _twin_paint for TWIN_CAM_SYNC)."""
        cp = self.env.sim.model.camera_name2id("robot0_eye_in_hand")
        d = self.env.sim.data
        return d.cam_xpos[cp].copy(), d.cam_xmat[cp].reshape(3, 3).copy()

    def _twin_cam_err(self):
        """Twin vs physical wrist-camera world pose at this paint (mm, deg)."""
        ct = self.twin.sim.model.camera_name2id("robot0_eye_in_hand")
        dt = self.twin.sim.data
        p_w, R_w = self._phys_cam_pose()
        st = self.twin_stats
        st.setdefault("cam_err_mm", []).append(float(np.linalg.norm(dt.cam_xpos[ct] - p_w) * 1000))
        st.setdefault("cam_err_deg", []).append(float(align.rot_err_deg(dt.cam_xmat[ct].reshape(3, 3), R_w)))

    def _probe_body_pos(self, name, shift_world, shift_parent=None):
        """Set a twin body's position = original + shift (world vector expressed in the parent frame)."""
        m, d = self.twin.sim.model, self.twin.sim.data
        b = m.body_name2id(name)
        if b not in self._probe_pos0:
            self._probe_pos0[b] = m.body_pos[b].copy()
        off = np.zeros(3)
        if shift_world is not None:
            Rp = d.xmat[m.body_parentid[b]].reshape(3, 3)
            off = Rp.T @ shift_world
        if shift_parent is not None:
            off = off + shift_parent
        m.body_pos[b] = self._probe_pos0[b] + off

    def _twin_paint_copy(self, obs):
        """TWIN_MOUNT=copy: same body, arm and gripper joints copied, probe offsets, camera on the physical camera
        (+ cam_shift). twin_stats pos_err_mm = twin vs physical grip-site position (0 without offsets)."""
        tw, tr = self.twin, self.twin.robots[0]
        pr, d = self.env.robots[0], tw.sim.data
        d.qpos[tr._ref_joint_pos_indexes] = self.env.sim.data.qpos[pr._ref_joint_pos_indexes]
        d.qpos[tr._ref_gripper_joint_pos_indexes] = self.env.sim.data.qpos[pr._ref_gripper_joint_pos_indexes]
        d.qvel[:] = 0.0
        tw.sim.forward()  # parent frames for the offsets
        mt = tw.sim.model
        base = [b for b in range(mt.nbody) if mt.body_parentid[b] == 0 and (mt.body_id2name(b) or "").startswith("robot0")]
        assert len(base) == 1, base
        self._probe_body_pos(mt.body_id2name(base[0]), self.probe["robot_shift"])
        roots = [b for b in range(mt.nbody) if (mt.body_id2name(b) or "").startswith("gripper0")
                 and not (mt.body_id2name(mt.body_parentid[b]) or "").startswith("gripper0")]
        assert len(roots) == 1, roots
        self._probe_body_pos(mt.body_id2name(roots[0]), self.probe["grip_shift"], self.probe.get("grip_shift_hand"))
        tw.sim.forward()
        self._twin_wrist_cam_to_physical(self.probe["cam_shift"])
        self._twin_cam_err()
        p_tw, _ = align.site_pose(tw.sim, tr.eef_site_id)
        p_ph, _ = align.site_pose(self.env.sim, pr.eef_site_id)
        st = self.twin_stats
        st["pos_err_mm"].append(float(np.linalg.norm(p_tw - p_ph)) * 1000)
        st["rot_err_deg"].append(0.0)
        st["posture_deg"].append(0.0)
        o = self._twin_obs_probe()
        if self.twin_cams in ("both", "agent"):
            obs["agentview_image"] = o["agentview_image"]
        if self.twin_cams in ("both", "wrist"):
            obs["robot0_eye_in_hand_image"] = o["robot0_eye_in_hand_image"]
        return obs

    def _twin_obs_probe(self):
        """Twin observation with the probe offsets as set; with probe cams = agent | wrist, the other camera is re-rendered
        after removing every render-only offset (same twin state otherwise), through the same observation path."""
        tw = self.twin
        tw.env._update_observables(force=True)
        o = tw.env._get_observations()
        cams = self.probe.get("cams", "both")
        if cams == "both":
            return o
        o = {k: (np.array(v, copy=True) if isinstance(v, np.ndarray) else v) for k, v in o.items()}
        m, d = tw.sim.model, tw.sim.data
        saved_pos = {b: m.body_pos[b].copy() for b in self._probe_pos0}
        for b, p0 in self._probe_pos0.items():
            m.body_pos[b] = p0
        obj = self.probe.get("obj_shift")
        if obj is not None:
            a = m.get_joint_qpos_addr(obj[0])[0]
            d.qpos[a:a + 3] -= np.asarray(obj[1], float)
        tw.sim.forward()
        if self.twin_seed_phys or self.toy_parts or self.twin_cam == "phys":
            self._twin_wrist_cam_to_physical(self.probe["cam_shift"] if self.probe.get("cam_shift") is not None and cams == "agent" else None)
        tw.env._update_observables(force=True)
        o2 = tw.env._get_observations()
        other = "robot0_eye_in_hand_image" if cams == "agent" else "agentview_image"
        o[other] = o2[other]
        for b, p in saved_pos.items():  # put the offsets back so the twin state is as painted
            m.body_pos[b] = p
        if obj is not None:
            d.qpos[a:a + 3] += np.asarray(obj[1], float)
        tw.sim.forward()
        return o

    def _toy_paint_composite(self, obs):
        st = self.twin_stats
        obs = self._toy_composite(obs)
        img = obs["agentview_image"]
        if self._twin_last is not None:  # stale-frame guard, on the composite
            moved = np.linalg.norm(obs["robot0_eef_pos"] - self._twin_last[0]) > 0.005
            if moved and np.array_equal(img, self._twin_last[1]):
                st["stale"] += 1
        self._twin_last = (np.array(obs["robot0_eef_pos"], copy=True), np.array(img, copy=True))
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
        self._twin_cam_err()
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

    def _twin_wrist_cam_to_physical(self, shift_world=None):
        """Put the twin's wrist camera at the physical body's wrist-camera world pose (G_R); toy1 probes may add a
        world translation."""
        import mujoco
        import robosuite.utils.transform_utils as T
        tw = self.twin
        mt, dt = tw.sim.model, tw.sim.data
        cam_t = mt.camera_name2id("robot0_eye_in_hand")
        cam_p = self.env.sim.model.camera_name2id("robot0_eye_in_hand")
        p_w, R_w = self._phys_cam_pose()
        if shift_world is not None:
            p_w = p_w + shift_world
        b = mt.cam_bodyid[cam_t]
        R_b = dt.xmat[b].reshape(3, 3)
        mt.cam_pos[cam_t] = R_b.T @ (p_w - dt.xpos[b])
        if getattr(self, "toy_on", False):
            # toy1: float64 conversion (robosuite mat2quat works in float32 -> up to ~0.03 deg camera rotation error,
            # seen in t0check <job>); published G_R path below is kept unchanged
            qw = np.zeros(4)
            mujoco.mju_mat2Quat(qw, np.ascontiguousarray(R_b.T @ R_w).reshape(9))
            mt.cam_quat[cam_t] = qw
        else:
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

    def _grp_proprio(self, q6, grp_map=None):
        closed = self.R85_CLOSED if (grp_map or getattr(self, "grp_map", "kp20")) == "kp20" else self._kp100_closed()
        f = float(np.clip(1.0 - q6[0] / closed, 0.0, 1.0))  # opening fraction
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
        if self.wrist_dz:  # TOY1 G_cam (WRIST_CAM_DZ_MM / TOY_WRIST_CAM=back): translate along the grip-site z axis
            p_cam = p_cam + R_site @ np.array([0.0, 0.0, self.wrist_dz])
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
