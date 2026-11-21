"""TOY2 (2026-11-03): in-process snapshot / restore of a closed-loop OFT episode at a policy-query time.

What is saved (everything that the continuation depends on):
  - the physical MjData, whole (mujoco.MjData copy: qpos, qvel, act, ctrl, qacc_warmstart, time, contacts, ...),
  - the robosuite robot / controller / gripper python state (numeric attributes, deques, interpolators deep-copied;
    attributes that reference the simulator are skipped),
  - the robosuite / LIBERO env counters (timestep, cur_time, done, ...), the observation dict at the query,
  - BodyEnv bookkeeping (quaternion sign, twin warm-start q, stale-frame memory),
  - the policy side: the open-loop action queue and the RNG states (numpy, random, torch CPU / CUDA).
restore() may target the same BodyEnv or another BodyEnv built on the same physical body (identical MjModel sizes):
the fork driver restores one baseline state into envs with different twin renderers.
"""
import copy
import random

import numpy as np

import collections

_SIMPLE = (int, float, bool, str, type(None), np.ndarray, np.generic)


def _ok(v, depth=0):
    """Plain data only: numbers, arrays, containers of them, robosuite *Buffer objects (recent_qpos etc.)."""
    if isinstance(v, _SIMPLE):
        return True
    if depth > 3:
        return False
    if isinstance(v, (list, tuple, collections.deque)):
        return all(_ok(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return all(isinstance(k, (str, int)) and _ok(x, depth + 1) for k, x in v.items())
    if "Buffer" in type(v).__name__ or "Interpolator" in type(v).__name__:
        return True
    return False


def _plain_state(obj, skip=()):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if k not in skip and _ok(v)}


def _set_plain(obj, st):
    for k, v in st.items():
        setattr(obj, k, copy.deepcopy(v))


def _copy_mjdata(dst, src):
    """Copy every array field and the time of MjData src into dst (same model). The python bindings have no
    mj_copyData; the contact list is derived and recomputed by the next mj_step / mj_forward."""
    n = 0
    for name in dir(src):
        if name.startswith("_"):
            continue
        try:
            v = getattr(src, name)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            try:
                d = getattr(dst, name)
                if isinstance(d, np.ndarray) and d.shape == v.shape and d.flags.writeable:
                    d[...] = v
                    n += 1
            except Exception:
                pass
    dst.time = src.time
    return n


def _rob(be):
    return be.env.env  # BodyEnv -> OffScreenRenderEnv -> robosuite / LIBERO env


def snapshot(be, obs, action_queue):
    import mujoco
    import torch
    env = _rob(be)
    r = env.robots[0]
    snap = {
        "mjdata": copy.copy(env.sim.data._data),
        "robot": _plain_state(r), "controller": _plain_state(r.controller),
        "gripper_action": copy.deepcopy(getattr(r.gripper, "current_action", None)),
        "env": _plain_state(env, skip=("robots", "model", "objects_dict", "fixtures_dict", "_observables")),
        "obs": copy.deepcopy(obs), "queue": list(copy.deepcopy(list(action_queue))),
        "be": {"sign": be.sign, "_twin_q": copy.deepcopy(getattr(be, "_twin_q", None)),
               "_twin_rest": copy.deepcopy(getattr(be, "_twin_rest", None))},
        "rng": (np.random.get_state(), random.getstate(), torch.get_rng_state(),
                torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None),
    }
    snap["mjdata_nbytes"] = int(snap["mjdata"].qpos.nbytes)
    return snap


def restore(be, snap):
    import mujoco
    import torch
    env = _rob(be)
    r = env.robots[0]
    _copy_mjdata(env.sim.data._data, snap["mjdata"])
    _set_plain(r, snap["robot"])
    _set_plain(r.controller, snap["controller"])
    if snap["gripper_action"] is not None:
        r.gripper.current_action = copy.deepcopy(snap["gripper_action"])
    _set_plain(env, snap["env"])
    be.sign = snap["be"]["sign"]
    np.random.set_state(snap["rng"][0])
    random.setstate(snap["rng"][1])
    torch.set_rng_state(snap["rng"][2])
    if snap["rng"][3] is not None:
        torch.cuda.set_rng_state_all(snap["rng"][3])
    return copy.deepcopy(snap["obs"]), list(copy.deepcopy(snap["queue"]))
