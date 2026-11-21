"""Online arm-environment contact log (protocol_block3.md section 4.1), same rule as $PROJ_ROOT/embody_dev/_contacts.py.

Per control step (after env.step), scan sim.data.contact: a contact counts when one side's body name starts
with "robot0" and the other side's body does not start with "robot0", the gripper prefix or "mount0".
Accumulates arm_env_contact_steps, first_contact (step index, -1 if none), pairs {"robot_body|other_body": steps},
first_grip_obj (block 3 amendment 1, 3.1: first step at which a gripper-prefix geom touches a movable object, i.e. a body
whose root below the world carries a free joint; -1 if none; read-only, added 2026-10-31),
stall_steps (k >= 10, mean |action[:3]| over the last 10 actions > 0.5 and the grip site moved < 5 mm over 10 steps).

forearm_check=True (UR5e_nc manipulation check): the forearm collision geoms have contype=conaffinity=0 in the
physics model, so they never make contacts. Every step a shadow MjData gets the same qpos, the forearm geoms'
contype/conaffinity are restored on the model only for the duration of mj_kinematics + mj_collision on the
shadow data, then set back to 0. The main simulation data is never touched. A step "penetrates" when a shadow
contact between a robot0_forearm_link geom and a non-robot geom has dist < -1 mm; recorded per step as
forearm_pen_steps, forearm_pen_max_mm (deepest), forearm_pen_pairs.
"""
import json

import mujoco
import numpy as np

PEN_MM = 1.0


class ContactLog:
    def __init__(self, sim, robot, forearm_geoms=None):
        self.sim = sim
        self.m = sim.model
        self.site = robot.eef_site_id
        self.gp = robot.gripper.naming_prefix
        self.k = 0
        self.steps = 0
        self.first = -1
        self.pairs = {}
        self.stall = 0
        self.acts = []
        self.pos = []
        self.first_grip = -1
        self._obj_body = self._movable_bodies()
        self.fc = bool(forearm_geoms)
        if self.fc:
            m = self.m
            self.fa = [int(g) for g in forearm_geoms]  # (geom id, original contype, original conaffinity)
            self.fa_orig = {g: (1, 1) for g in self.fa}
            self.shadow = mujoco.MjData(m._model)
            self.pen_steps = 0
            self.pen_max = 0.0
            self.pen_pairs = {}

    def _movable_bodies(self):
        """Body ids that belong to a free-joint object (the object's root body and all its descendants)."""
        m = self.m
        free_roots = {int(m.jnt_bodyid[j]) for j in range(m.njnt) if int(m.jnt_type[j]) == 0}  # mjJNT_FREE
        out = set()
        for b in range(m.nbody):
            a = b
            while a > 0:
                if a in free_roots:
                    out.add(b)
                    break
                a = int(m.body_parentid[a])
        return out

    def _bn(self, g):
        return self.m.body_id2name(self.m.geom_bodyid[g]) or ""

    def update(self, action):
        d = self.sim.data
        hit = False
        for i in range(d.ncon):
            c = d.contact[i]
            b1, b2 = self._bn(c.geom1), self._bn(c.geom2)
            if self.first_grip < 0:
                o1, o2 = int(self.m.geom_bodyid[c.geom1]), int(self.m.geom_bodyid[c.geom2])
                if (b1.startswith(self.gp) and o2 in self._obj_body) or (b2.startswith(self.gp) and o1 in self._obj_body):
                    self.first_grip = self.k
            for x, y in ((b1, b2), (b2, b1)):
                if x.startswith("robot0") and not y.startswith(("robot0", self.gp, "mount0")):
                    hit = True
                    key = f"{x}|{y}"
                    self.pairs[key] = self.pairs.get(key, 0) + 1
        if hit:
            self.steps += 1
            if self.first < 0:
                self.first = self.k
        self.acts.append(np.asarray(action, dtype=float)[:3])
        self.pos.append(d.site_xpos[self.site].copy())
        if self.k >= 10 and np.linalg.norm(np.asarray(self.acts[-11:-1]), axis=1).mean() > 0.5 \
                and np.linalg.norm(self.pos[-1] - self.pos[-11]) < 0.005:
            self.stall += 1
        if self.fc:
            self._forearm_pen()
        self.k += 1

    def _forearm_pen(self):
        m, mm = self.m, self.m._model
        s = self.shadow
        s.qpos[:] = self.sim.data.qpos
        mujoco.mj_kinematics(mm, s)
        for g in self.fa:
            mm.geom_contype[g], mm.geom_conaffinity[g] = self.fa_orig[g]
        try:
            mujoco.mj_collision(mm, s)
        finally:
            for g in self.fa:
                mm.geom_contype[g] = 0
                mm.geom_conaffinity[g] = 0
        fa = set(self.fa)
        deepest = 0.0
        for i in range(s.ncon):
            c = s.contact[i]
            if c.geom1 in fa or c.geom2 in fa:
                other = c.geom2 if c.geom1 in fa else c.geom1
                ob = self._bn(other)
                if ob.startswith(("robot0", self.gp, "mount0")):
                    continue
                depth = -float(c.dist) * 1000
                if depth > PEN_MM:
                    deepest = max(deepest, depth)
                    self.pen_pairs[ob] = self.pen_pairs.get(ob, 0) + 1
        if deepest > 0:
            self.pen_steps += 1
            self.pen_max = max(self.pen_max, deepest)

    def summary(self):
        out = {"arm_env_contact_steps": int(self.steps), "first_contact": int(self.first),
               "stall_steps": int(self.stall), "pairs": dict(self.pairs), "n_steps": int(self.k),
               "first_grip_obj": int(self.first_grip)}
        if self.fc:
            out.update(forearm_geoms_off=len(self.fa), forearm_pen_steps=int(self.pen_steps),
                       forearm_pen_max_mm=float(self.pen_max), forearm_pen_pairs=dict(self.pen_pairs))
        return out

    def npz_fields(self):
        s = self.summary()
        return {"contact_" + k: (json.dumps(v) if isinstance(v, dict) else v) for k, v in s.items()}


def make(env):
    """ContactLog on the physics env behind a BodyEnv or a plain LIBERO env (call after set_init_state)."""
    nc = getattr(env, "nc_geoms", None)  # BodyEnv with NO_FOREARM_CONTACT=1: [(geom, contype, conaffinity)]
    log = ContactLog(env.sim, env.robots[0], forearm_geoms=[g for g, _, _ in nc] if nc else None)
    if nc:
        log.fa_orig = {g: (ct, ca) for g, ct, ca in nc}
    return log
