"""physfix (Astra ruling 3, 2026-11-06): check the Robotiq proprio / twin-finger mapping over the WHOLE open-close
cycle (not only the closed endpoint) under the stabilised configuration. Physics only, no GL, no policy, no outcome.
Free air: from a LIBERO init state (gripper above the table), arm action 0, gripper command sequence
  open x15, close x40, open x40, close x40   (OFT binary command: -1 open, +1 close)
Variants: panda (BODY=Panda, reference), a_kp100 (PandaR85 GRP_KP=100 R85_ARMATURE=1e-3), kp100 (no armature, old),
          kp20 (stock Robotiq kp 20, GRP_KP unset).
Per step: finger_joint (raw), the 6 linkage joints, opening fraction under GRP_MAP kp20 and kp100 (BodyEnv._grp_proprio,
the proprio the policy sees and the twin finger opening), and for panda the Panda opening fraction.
Summary per variant: closed / open plateaus of finger_joint, opening fraction at the plateaus under both maps, steps to
reach 10 / 50 / 90 % of each transition, and per-step |fraction - Panda fraction| over each transition.
Usage: python grip_sweep_pf.py --code $PROJ_ROOT/embody_next --out FILE.json [--suite libero_spatial] [--tasks 0,5]
"""
import argparse, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.path.expanduser("$PROJ_ROOT/embody_next"))
ap.add_argument("--out", required=True)
ap.add_argument("--suites", default="libero_spatial,libero_object")
ap.add_argument("--tasks", default="0,5")
args = ap.parse_args()
for k in ("R85_ARMATURE", "GRP_KP", "GRP_MAP", "TWIN_RENDER", "BODY"):
    os.environ.pop(k, None)
os.environ.update(COPY_GRIP_CMD="1", ORIGINAL_ONLY="1")
sys.path.insert(0, args.code)
import embody  # noqa
assert os.path.abspath(embody.__file__).startswith(os.path.abspath(args.code))
from embody.body_env import BodyEnv  # noqa
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs.env_wrapper import ControlEnv  # noqa

# runtime-patch variant (WORKLOG_vlazoo 14 / phys_replay.py style: model.dof_armature set on the compiled model, compiled
# constants NOT recomputed), applied before the gripper settle of every reset; test-only monkeypatch, embody_next untouched
_orig_gi = BodyEnv._grp_init


def _gi(self, *a, **k):
    v = os.environ.get("PF_RT_ARMATURE")
    if v and self.gk == "r85":
        m = self.env.sim.model
        for j in GJ:
            m.dof_armature[m.jnt_dofadr[m.joint_name2id("gripper0_" + j)]] = float(v)
    return _orig_gi(self, *a, **k)


BodyEnv._grp_init = _gi
SEQ = [-1.0] * 15 + [1.0] * 40 + [-1.0] * 40 + [1.0] * 40
SEG = {"close1": (15, 55), "open1": (55, 95), "close2": (95, 135)}
VAR = {"panda": ("Panda", {}), "a_kp100": ("PandaR85", {"GRP_KP": "100", "R85_ARMATURE": "1e-3"}),
       "rt_kp100": ("PandaR85", {"GRP_KP": "100", "PF_RT_ARMATURE": "1e-3"}),
       "kp100": ("PandaR85", {"GRP_KP": "100"}), "kp20": ("PandaR85", {})}
GJ = ["finger_joint", "left_inner_finger_joint", "left_inner_knuckle_joint", "right_outer_knuckle_joint",
      "right_inner_finger_joint", "right_inner_knuckle_joint"]


def run(suite, tid, var):
    body, env = VAR[var]
    for k in ("R85_ARMATURE", "GRP_KP", "PF_RT_ARMATURE"):
        os.environ.pop(k, None)
    os.environ.update(env)
    task = suite.get_task(tid)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    np.random.seed(7)
    ref = ControlEnv(bddl_file_name=bddl, has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    ref.seed(0)
    be = BodyEnv(ref, bddl, body, 256, 10, log=lambda m: None, render=False)
    be.reset()
    be.set_init_state(suite.get_task_init_states(tid)[0])
    sim = be.env.sim
    rb = be.env.robots[0]
    arm_now = None
    if body == "PandaR85":
        arm_now = sim.model._model.dof_armature[[sim.model.jnt_dofadr[sim.model.joint_name2id("gripper0_" + j)] for j in GJ]].tolist()
        gi = [sim.model.get_joint_qpos_addr("gripper0_" + j) for j in GJ]
    rows = []
    for a in SEQ:
        be.step([0, 0, 0, 0, 0, 0, a])
        if body == "PandaR85":
            q6 = sim.data.qpos[gi].copy()
            rows.append({"fj": float(q6[0]), "q6": q6.round(5).tolist(),
                         "f_kp20": float(be._grp_proprio(q6, "kp20")[0] / be.PANDA_OPEN),
                         "f_kp100": float(be._grp_proprio(q6, "kp100")[0] / be.PANDA_OPEN)})
        else:
            p2 = sim.data.qpos[rb._ref_gripper_joint_pos_indexes].copy()
            rows.append({"f_panda": float(np.clip((p2[0] - p2[1]) / (2 * be.PANDA_OPEN), 0, 1)), "p2": p2.round(6).tolist()})
    be.close()
    return {"rows": rows, "armature": arm_now}


def steps_to(fr, lo, hi, closing):
    tr = fr[lo:hi]
    a, b = tr[0], tr[-1]
    out = {}
    for p in (0.1, 0.5, 0.9):
        target = a + p * (b - a)
        idx = np.nonzero((tr >= target) if b > a else (tr <= target))[0]
        out[str(p)] = int(idx[0]) if len(idx) else None
    return out


res = {"seq": "open15 close40 open40 close40", "out": {}}
for sname in args.suites.split(","):
    suite = OriginalObjectSuite() if sname == "libero_object" else OriginalSpatialSuite()
    for tid in [int(t) for t in args.tasks.split(",")]:
        R = {v: run(suite, tid, v) for v in VAR}
        fp = np.array([r["f_panda"] for r in R["panda"]["rows"]])
        key = f"{sname}:{tid}"
        res["out"][key] = {"panda_plateau": {"open": float(fp[54 - 40]), "closed": float(fp[54]), "reopen": float(fp[94])},
                           "panda_steps": {s: steps_to(fp, *SEG[s], s.startswith("close")) for s in SEG}}
        for v in ("a_kp100", "rt_kp100", "kp100", "kp20"):
            rows = R[v]["rows"]
            fj = np.array([r["fj"] for r in rows])
            ent = {"armature": R[v]["armature"], "fj_open": float(fj[14]), "fj_closed": float(fj[54]), "fj_reopen": float(fj[94]),
                   "fj_closed2": float(fj[134]), "fj_min": float(fj.min()), "fj_max": float(fj.max()),
                   "linkage_closed": rows[54]["q6"]}
            for mp in ("kp20", "kp100"):
                f = np.array([r["f_" + mp] for r in rows])
                ent[mp] = {"f_open": float(f[14]), "f_closed": float(f[54]), "f_reopen": float(f[94]),
                           "steps": {s: steps_to(f, *SEG[s], s.startswith("close")) for s in SEG},
                           "absdiff_vs_panda": {s: {"mean": float(np.abs(f[a:b] - fp[a:b]).mean()), "max": float(np.abs(f[a:b] - fp[a:b]).max())} for s, (a, b) in SEG.items()},
                           "monotone_close1": bool(np.all(np.diff(f[15:55]) <= 1e-9)), "monotone_open1": bool(np.all(np.diff(f[55:95]) >= -1e-9))}
            res["out"][key][v] = ent
        res["out"][key]["series"] = {v: [r.get("f_kp100", r.get("f_panda")) for r in R[v]["rows"]] for v in VAR}
        res["out"][key]["series_fj"] = {v: [r["fj"] for r in R[v]["rows"]] for v in ("a_kp100", "rt_kp100", "kp100", "kp20")}
        print(key, json.dumps({v: {"fj_closed": res["out"][key][v]["fj_closed"], "f100_closed": res["out"][key][v]["kp100"]["f_closed"],
                                   "f20_closed": res["out"][key][v]["kp20"]["f_closed"], "steps100": res["out"][key][v]["kp100"]["steps"]}
                               for v in ("a_kp100", "rt_kp100", "kp100", "kp20")}), "panda", res["out"][key]["panda_steps"], flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print("done")
