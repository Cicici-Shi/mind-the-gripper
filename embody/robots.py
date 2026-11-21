"""Mounted arm classes for LIBERO so that robots=["IIWA"] / ["UR5e"] resolve.

LIBERO rewrites robots=[name] into f"Mounted{name}" and only defines MountedPanda.
These classes reuse robosuite's IIWA / UR5e models but take Panda's gripper,
mount, and table base offsets, so the base frame coincides with the Panda setup.

Variants with a specific initial joint configuration (from IK alignment) are
created with make_variant(); robosuite's RobotModelMeta registers every class
by __name__ on creation.

embody_v3 (block 3) adds, all opt-in by body name, nothing changes for the old names:
  Kinova3   MountedKinova3: robosuite Kinova3 arm + Panda gripper / mount / base offset (K3).
  PandaRT   MountedPandaRT: Panda arm + RethinkGripper (RT).
  PandaL3   MountedPandaL3: Panda with joint3 removed from the MJCF and its child frame rotated by the
            lock angle, so the arm has 6 joints and OSC sees a 6-column Jacobian (P6j3).
"""
import numpy as np

from robosuite.models.robots.manipulators.iiwa_robot import IIWA
from robosuite.models.robots.manipulators.kinova3_robot import Kinova3
from robosuite.models.robots.manipulators.ur5e_robot import UR5e
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.single_arm import SingleArm
from robosuite.utils.mjcf_utils import array_to_string, string_to_array

import libero.libero.envs.robots  # noqa: F401  registers MountedPanda*
from libero.libero.envs.robots.mounted_panda import MountedPanda

_PANDA_BASE_OFFSETS = MountedPanda.base_xpos_offset.fget(None)


class _MountedMixin:
    @property
    def default_mount(self):
        return "RethinkMount"

    @property
    def default_gripper(self):
        return "PandaGripper"

    @property
    def base_xpos_offset(self):
        return _PANDA_BASE_OFFSETS


class MountedIIWA(_MountedMixin, IIWA):
    pass


class MountedUR5e(_MountedMixin, UR5e):
    pass


ROBOT_CLASS_MAPPING.update({"MountedIIWA": SingleArm, "MountedUR5e": SingleArm})

_BASES = {"Panda": MountedPanda, "IIWA": MountedIIWA, "UR5e": MountedUR5e}


def make_variant(arm: str, tag: str, init_qpos) -> str:
    """Register Mounted{arm}{tag} with a fixed init_qpos. Returns the name to pass
    as robots=[...] (without the 'Mounted' prefix LIBERO adds)."""
    q = np.asarray(init_qpos, dtype=float).copy()
    name = f"Mounted{arm}{tag}"
    type(name, (_BASES[arm],), {"init_qpos": property(lambda self, q=q: q.copy())})
    ROBOT_CLASS_MAPPING[name] = SingleArm
    return f"{arm}{tag}"


# GRP condition: Panda arm, Robotiq 85 gripper instead of the Panda hand.
class MountedPandaR85(MountedPanda):
    @property
    def default_gripper(self):
        return "Robotiq85Gripper"


ROBOT_CLASS_MAPPING["MountedPandaR85"] = SingleArm
_BASES["PandaR85"] = MountedPandaR85


# ---------------- embody_v3 / block 3 ----------------

# K3: second foreign 7-joint arm. Torque limits are Kinova3's own (+-32 / +-13 Nm), on purpose.
class MountedKinova3(_MountedMixin, Kinova3):
    pass


# RT: second parallel gripper. Rethink fingers use kp 1000 and forcerange +-20 N, the same as the
# Panda fingers (rethink_gripper.xml vs panda_gripper.xml), so there is no stiffness confound.
class MountedPandaRT(MountedPanda):
    @property
    def default_gripper(self):
        return "RethinkGripper"


ROBOT_CLASS_MAPPING.update({"MountedKinova3": SingleArm, "MountedPandaRT": SingleArm})
_BASES.update({"Kinova3": MountedKinova3, "PandaRT": MountedPandaRT})


def lock_joint(model, joint, angle):
    """Remove `joint` (unprefixed name, e.g. "joint3") from a robosuite RobotModel's MJCF and freeze
    its child body at `angle`: the body's quat is post-multiplied by the joint rotation, which is
    exactly the kinematics the hinge had at that angle. The joint's motor is removed too and the
    model's cached joint / actuator lists are updated, so SingleArm builds a 6-joint arm (OSC
    Jacobian, nullspace and torque vectors all 6 wide). Only joints at the body origin are allowed.
    """
    full = model.naming_prefix + joint
    parent = None
    for body in model.root.iter("body"):
        for j in body.findall("joint"):
            if j.get("name") == full:
                parent, jel = body, j
    assert parent is not None, f"{full} not found"
    assert np.allclose(string_to_array(jel.get("pos", "0 0 0")), 0.0), "lock_joint: joint must sit at the body origin"
    axis = string_to_array(jel.get("axis", "0 0 1"))
    axis = axis / np.linalg.norm(axis)
    q_body = string_to_array(parent.get("quat", "1 0 0 0"))  # wxyz
    q_rot = np.concatenate([[np.cos(angle / 2)], np.sin(angle / 2) * axis])  # wxyz
    w1, x1, y1, z1 = q_body
    w2, x2, y2, z2 = q_rot
    q_new = np.array([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ])
    parent.set("quat", array_to_string(q_new))
    parent.remove(jel)
    acts = [a for a in model.actuator if a.get("joint") == full]
    assert len(acts) == 1, f"expected one actuator on {full}, found {len(acts)}"
    model.actuator.remove(acts[0])
    model._elements["joints"] = [e for e in model._elements["joints"] if e is not jel]
    model._elements["actuators"] = [e for e in model._elements["actuators"] if e is not acts[0]]
    model._joints = [n for n in model._joints if n != joint]
    model._actuators = [n for n in model._actuators if n != acts[0].get("name")[len(model.naming_prefix):]]
    assert len(model._joints) == len(model._elements["joints"]) and len(model._actuators) == len(model._elements["actuators"])


# P6j3 lock angle: median of robot0_joint3 at the first policy step over the 200 SRC episodes
# ($PROJ_ROOT/runs/src/*.npz qpos[0]); computed 2026-10-27: median 0.001397, range [-0.0388, 0.0449].
PANDA_L3_ANGLE = 0.0013974246779987272


def _make_locked_panda(j, angle):
    idx = j - 1

    class _Locked(MountedPanda):
        def __init__(self, idn=0):
            super().__init__(idn=idn)
            lock_joint(self, f"joint{j}", angle)

        @property
        def init_qpos(self):
            return np.delete(MountedPanda.init_qpos.fget(self), idx)

    name = f"MountedPandaL{j}"
    cls = type(name, (_Locked,), {"LOCKED_JOINT": j, "LOCK_ANGLE": angle})
    ROBOT_CLASS_MAPPING[name] = SingleArm
    _BASES[f"PandaL{j}"] = cls
    return cls


MountedPandaL3 = _make_locked_panda(3, PANDA_L3_ANGLE)


# RT2 (protocol_block3.md 1.1 / 1.5): Rethink gripper with a longer closing stroke. Stock Rethink fingers stop
# 13.4 mm apart (inner pad faces) and cannot hold the LIBERO bowl rims. Only the slide-joint limits and the
# actuator ctrlrange are extended inward; kp 1000, force range +-20 N, opening width, geometry and colours are
# unchanged. Joint limits: +6.5 mm per finger (inner gap 13.4 - 13.0 = 0.4 mm when closed, pads never overlap;
# pad geoms do not collide with each other, so 7 mm would make them interpenetrate by 0.6 mm). Ctrlrange closing
# end: +7.0 mm per finger, so the closed command still overshoots the joint limit (stock 9.3 mm, now 9.8 mm).
from robosuite.models.grippers import GRIPPER_MAPPING  # noqa: E402
from robosuite.models.grippers.rethink_gripper import RethinkGripper  # noqa: E402

RT2_JOINT_EXT = 0.0065
RT2_CTRL_EXT = 0.0070


class RethinkGripper2(RethinkGripper):
    def __init__(self, idn=0):
        super().__init__(idn=idn)
        pre = self.naming_prefix
        for j in self.worldbody.iter("joint"):
            n = j.get("name")
            lo, hi = string_to_array(j.get("range"))
            if n == pre + "l_finger_joint":  # closes toward negative
                j.set("range", array_to_string(np.array([lo - RT2_JOINT_EXT, hi])))
            elif n == pre + "r_finger_joint":  # closes toward positive
                j.set("range", array_to_string(np.array([lo, hi + RT2_JOINT_EXT])))
        for a in self.actuator.iter("position"):
            lo, hi = string_to_array(a.get("ctrlrange"))
            if a.get("joint") == pre + "l_finger_joint":
                a.set("ctrlrange", array_to_string(np.array([lo - RT2_CTRL_EXT, hi])))
            elif a.get("joint") == pre + "r_finger_joint":
                a.set("ctrlrange", array_to_string(np.array([lo, hi + RT2_CTRL_EXT])))


GRIPPER_MAPPING["RethinkGripper2"] = RethinkGripper2


class MountedPandaRT2(MountedPanda):
    @property
    def default_gripper(self):
        return "RethinkGripper2"


ROBOT_CLASS_MAPPING["MountedPandaRT2"] = SingleArm
_BASES["PandaRT2"] = MountedPandaRT2


# libero_object (block 3, protocol section 6) uses the floor scene: LIBERO rewrites robots=[name] into
# f"OnTheGround{name}" (no mount, its own base offsets). Every body above gets an OnTheGround twin class
# with OnTheGroundPanda's mount (None) and base offsets; arm, gripper and joints are those of the Mounted class.
from libero.libero.envs.robots.on_the_ground_panda import OnTheGroundPanda  # noqa: E402

_GROUND_OFFSETS = OnTheGroundPanda.base_xpos_offset.fget(None)


def _make_ground(name, base):
    cname = f"OnTheGround{name}"
    cls = type(cname, (base,), {"default_mount": property(lambda self: None),
                                "base_xpos_offset": property(lambda self: _GROUND_OFFSETS)})
    ROBOT_CLASS_MAPPING[cname] = SingleArm
    return cls


for _n, _b in list(_BASES.items()):
    if _n != "Panda":  # LIBERO defines OnTheGroundPanda itself
        _make_ground(_n, _b)
