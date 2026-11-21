"""physfix (protocol_physfix.md 1, 5.3, 7.4), 2026-11-06: run the byte-copied OFT eval script
$PROJ_ROOT/vla_zoo/oft/run_libero_eval_m1.py (= embody_toy/oft/run_libero_eval_toy.py, sha256 a65fc6d0...; unchanged)
with EMBODY_DIR=$PROJ_ROOT/embody_next and these runtime patches (the script file itself is not modified):
  1. M1_FASTSYNC=1: m1_fastsync.patch() (bitwise-verified fast TWIN_CAM_SYNC snapshot, WORKLOG_vlazoo 8.3).
  2. BLIND: the script's log_message drops every line that carries an outcome ("Success:", "# successes",
     "success rate", "Total successes", "Overall success rate"); robosuite/OFT save_rollout_video is replaced by one that
     writes NO video at all (a video's length reveals the episode length, hence success). The dump npz still holds
     `success` (needed by the readout); nothing prints it.
  3. Online BADQACC (5.3): BodyEnv.step reads MjData.warning[mjWARN_BADQACC].number before/after every policy step;
     per episode the increase and the first policy step with an increase go into the npz.
  4. npz extras (via a numpy proxy around the script's _dump_episode only): pf_qacc_count, pf_qacc_first_step (-1 none),
     pf_r85_armature (float, 0 off), pf_grip_armature (model.dof_armature of the 6 Robotiq finger dofs, empty for
     Panda), pf_blind=1, pf_cell_env (json of the cell switches), pf_launch_sha.
Usage: python launch_pf.py <args of run_libero_eval_m1.py>
"""
import hashlib, json, os, runpy, sys
import numpy as _np

assert os.path.realpath(os.path.expanduser(os.environ["EMBODY_DIR"])) == os.path.realpath(os.path.expanduser("$PROJ_ROOT/embody_next")), os.environ["EMBODY_DIR"]
sys.path.insert(0, os.path.expanduser(os.environ["EMBODY_DIR"]))
if os.environ.get("M1_FASTSYNC") == "1":
    sys.path.insert(0, os.path.expanduser("$PROJ_ROOT/vla_zoo/common"))
    import m1_fastsync
    m1_fastsync.patch()
    print("M1_FASTSYNC patched", flush=True)
import mujoco  # noqa: E402
from embody import body_env as _be  # noqa: E402
assert os.path.realpath(_be.__file__).startswith(os.path.realpath(os.path.expanduser("$PROJ_ROOT/embody_next"))), _be.__file__

_W = int(mujoco.mjtWarning.mjWARN_BADQACC)
_GJ = ("finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint", "right_outer_knuckle_joint",
       "right_inner_finger_joint", "right_inner_knuckle_joint")
_ST = {"n": 0, "count": 0, "first": -1}
_orig_reset, _orig_sis, _orig_step = _be.BodyEnv.reset, _be.BodyEnv.set_init_state, _be.BodyEnv.step


def _qn(self):
    return int(self.env.sim.data._data.warning[_W].number)


def _reset(self, *a, **k):
    _ST.update(n=0, count=0, first=-1, con=[], con_err="")
    return _orig_reset(self, *a, **k)


def _sis(self, *a, **k):
    out = _orig_sis(self, *a, **k)
    _ST.update(n=0, count=0, first=-1, con=[], con_err="")
    return out


_F6 = _np.zeros(6)


def _grip_sets(self):
    """(left-finger geoms, right-finger geoms, all gripper geoms, robot geoms) of the physical sim, cached per model."""
    m = self.env.sim.model
    key = id(m._model)
    c = _ST.get("gs")
    if c is not None and c[0] == key:
        return c[1]
    L, R, Gr, Rb = set(), set(), set(), set()
    for g in range(m.ngeom):
        bn = m.body_id2name(m.geom_bodyid[g]) or ""
        if bn.startswith(("robot0", "gripper0", "mount0")):
            Rb.add(g)
        if bn.startswith("gripper0"):
            Gr.add(g)
            low = bn.lower()
            if "left" in low and ("finger" in low or "pad" in low):
                L.add(g)
            elif "right" in low and ("finger" in low or "pad" in low):
                R.add(g)
    hb = m.body_name2id(self.env.robots[0].robot_model.eef_name)
    v = (L, R, Gr, Rb, hb)
    _ST["gs"] = (key, v)
    return v


def _contacts(self):
    """Per policy step (protocol_physfix 7.1 grasp plausibility, CPU-side later): gripper-vs-non-robot contacts of the
    physical sim: [n, min dist (m, <0 = penetration), sum normal force (N), n left-finger, n right-finger,
    left mean contact pos in hand frame (3, m), right (3), body id of the largest-force contacted body]."""
    sim = self.env.sim
    m, d = sim.model._model, sim.data._data
    L, R, Gr, Rb, hb = _grip_sets(self)
    Rh = d.xmat[hb].reshape(3, 3)
    ph = d.xpos[hb]
    n, dmin, fs, pl, pr, best, bf = 0, _np.nan, 0.0, [], [], -1, -1.0
    for i in range(d.ncon):
        c = d.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if (g1 in Gr) == (g2 in Gr):
            continue
        other = g2 if g1 in Gr else g1
        mine = g1 if g1 in Gr else g2
        if other in Rb:
            continue
        mujoco.mj_contactForce(m, d, i, _F6)
        fn = float(abs(_F6[0]))
        n += 1
        dmin = float(c.dist) if _np.isnan(dmin) else min(dmin, float(c.dist))
        fs += fn
        loc = Rh.T @ (_np.asarray(c.pos) - ph)
        if mine in L:
            pl.append(loc)
        elif mine in R:
            pr.append(loc)
        if fn > bf:
            bf, best = fn, int(m.geom_bodyid[other])
    ml = _np.mean(pl, 0) if pl else _np.full(3, _np.nan)
    mr = _np.mean(pr, 0) if pr else _np.full(3, _np.nan)
    return [n, dmin, fs, len(pl), len(pr), *ml, *mr, best]


def _step(self, *a, **k):
    n0 = _qn(self)
    out = _orig_step(self, *a, **k)
    n1 = _qn(self)
    if n1 > n0:
        _ST["count"] += n1 - n0
        if _ST["first"] < 0:
            _ST["first"] = _ST["n"]
    _ST["n"] += 1
    try:
        _ST.setdefault("con", []).append(_contacts(self))
    except Exception as e:  # never break an episode for a descriptive quantity
        _ST["con_err"] = repr(e)
    return out


_be.BodyEnv.reset, _be.BodyEnv.set_init_state, _be.BodyEnv.step = _reset, _sis, _step
_ENV = {k: os.environ.get(k) for k in ("BODY", "GRP_KP", "GRP_MAP", "R85_ARMATURE", "COPY_GRIP_CMD", "TWIN_RENDER", "TWIN_BODY",
                                       "TWIN_CAMS", "TWIN_CAM", "TWIN_CAM_SYNC", "TOY_PARTS", "TOY_PHYS_GRIPPER", "TOY_TWIN_POSE",
                                       "M1_FASTSYNC", "EMBODY_DIR") if os.environ.get(k) is not None}
_LSHA = hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest()
_CUR = {"env": None}

import experiments.robot.libero.libero_utils as _lu  # noqa: E402


def _no_video(rollout_images, idx, success, task_description, log_file=None):
    return None


_lu.save_rollout_video = _no_video
_BAD = ("Success:", "# successes", "success rate", "Total successes", "Overall success rate", "success=")


class _NpProxy:
    def __getattr__(self, n):
        return getattr(_np, n)

    def savez_compressed(self, path, **kw):
        e = _CUR["env"]
        ga = []
        if e is not None and getattr(e, "gk", "") == "r85":
            m = e.env.sim.model
            ga = [float(m._model.dof_armature[m.jnt_dofadr[m.joint_name2id("gripper0_" + j)]]) for j in _GJ]
        kw.update(pf_qacc_count=_ST["count"], pf_qacc_first_step=_ST["first"],
                  pf_r85_armature=float(os.environ.get("R85_ARMATURE") or 0.0), pf_grip_armature=_np.asarray(ga, _np.float64),
                  pf_blind=1, pf_cell_env=json.dumps(_ENV, sort_keys=True), pf_launch_sha=_LSHA,
                  pf_grip_contact=_np.asarray(_ST.get("con") or _np.zeros((0, 12)), _np.float32),
                  pf_grip_contact_cols="n,min_dist_m,sum_fn_N,n_left,n_right,left_xyz_hand(3),right_xyz_hand(3),max_force_body",
                  pf_grip_contact_err=_ST.get("con_err", ""),
                  pf_body_names=_np.asarray([e.env.sim.model.body_id2name(b) or "" for b in range(e.env.sim.model.nbody)]) if e is not None else _np.asarray([]))
        return _np.savez_compressed(path, **kw)


here = os.path.expanduser("$PROJ_ROOT/vla_zoo/oft")
script = os.path.join(here, "run_libero_eval_m1.py")
assert hashlib.sha256(open(script, "rb").read()).hexdigest().startswith("a65fc6d08c48"), "eval script changed"
sys.argv = [script] + sys.argv[1:]
_g0 = runpy.run_path(script, run_name="pf_lib")  # defines everything, does not run main (returns a COPY)
g = _g0["run_task"].__globals__  # the live module namespace the script's functions resolve names in
assert g["run_episode"].__globals__ is g and g["_dump_episode"].__globals__ is g
_orig_log = g["log_message"]


def _log(message, log_file=None):
    if any(b in str(message) for b in _BAD):
        return
    return _orig_log(message, log_file)


g["log_message"] = _log
_orig_dump = g["_dump_episode"]


def _dump(*a, **k):
    g["np"] = _NpProxy()
    try:
        return _orig_dump(*a, **k)
    finally:
        g["np"] = _np


g["_dump_episode"] = _dump
_orig_run_episode = g["run_episode"]


def _run_episode(cfg, env, *a, **k):
    _CUR["env"] = env
    return _orig_run_episode(cfg, env, *a, **k)


g["run_episode"] = _run_episode
g["save_rollout_video"] = _no_video
print(f"PF_LAUNCH blind=1 qacc=1 env={json.dumps(_ENV, sort_keys=True)} launch_sha={_LSHA[:16]}", flush=True)
g["eval_libero"]()
