"""physfix regression (protocol_physfix.md 7.1 (ii)/(iii)), 2026-11-06. Physics-only (no GL), CPU is enough.
For each suite (spatial task 0, object task 0) and body (Panda, PandaR85) build the LIBERO env from embody_next twice,
R85_ARMATURE unset vs =1e-3, and compare
  (a) the robosuite task MJCF string (env.model.get_xml()): unified diff lines;
  (b) the compiled MjModel: every public ndarray field, list of fields that differ and, for dof_armature, which dofs.
Expected: Panda -> no difference at all; PandaR85 -> MJCF diff = exactly the 6 finger-joint lines gaining
armature="1e-3", compiled diff = dof_armature on those 6 dofs plus armature-derived constants only.
Usage: python xml_pf.py --code $PROJ_ROOT/embody_next --out FILE.json
"""
import argparse, difflib, json, os, sys
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--code", default=os.path.expanduser("$PROJ_ROOT/embody_next"))
ap.add_argument("--out", required=True)
ap.add_argument("--arm", default="1e-3")
args = ap.parse_args()
os.environ.pop("R85_ARMATURE", None)
sys.path.insert(0, args.code)
import embody  # noqa
assert os.path.abspath(embody.__file__).startswith(os.path.abspath(args.code)), embody.__file__
from embody import robots as RB  # noqa
from embody.original import OriginalSpatialSuite, OriginalObjectSuite  # noqa
from libero.libero import get_libero_path  # noqa
from libero.libero.envs.env_wrapper import ControlEnv  # noqa


def build(bddl, body, arm):
    if arm:
        os.environ["R85_ARMATURE"] = arm
    else:
        os.environ.pop("R85_ARMATURE", None)
    np.random.seed(7)
    e = ControlEnv(bddl_file_name=bddl, robots=[body], has_renderer=False, has_offscreen_renderer=False, use_camera_obs=False)
    e.seed(0)
    e.reset()
    xml = e.env.model.get_xml() if hasattr(e, "env") else e.model.get_xml()
    m = e.sim.model._model
    arrs = {}
    for n in dir(m):
        if n.startswith("_"):
            continue
        try:
            v = getattr(m, n)
        except Exception:
            continue
        if isinstance(v, np.ndarray):
            arrs[n] = v.copy()
    names = {"dof_joint": [e.sim.model.joint_id2name(int(j)) for j in m.dof_jntid]}
    gcls = type(e.robots[0].gripper).__name__
    e.close()
    os.environ.pop("R85_ARMATURE", None)
    return xml, arrs, names, gcls


res = {"code": args.code, "arm": args.arm, "robots_py_sha": __import__("hashlib").sha256(open(RB.__file__, "rb").read()).hexdigest()}
for sname, suite in (("libero_spatial", OriginalSpatialSuite()), ("libero_object", OriginalObjectSuite())):
    task = suite.get_task(0)
    bddl = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    for body in ("Panda", "PandaR85"):
        x0, a0, n0, g0 = build(bddl, body, None)
        x1, a1, n1, g1 = build(bddl, body, args.arm)
        dl = [l for l in difflib.unified_diff(x0.splitlines(), x1.splitlines(), lineterm="", n=0)
              if (l.startswith("+") or l.startswith("-")) and not l.startswith("+++") and not l.startswith("---")]
        fields = []
        for k in sorted(set(a0) | set(a1)):
            if k not in a0 or k not in a1 or a0[k].shape != a1[k].shape:
                fields.append({"field": k, "shape_changed": True})
                continue
            d = a0[k] != a1[k]
            if np.issubdtype(a0[k].dtype, np.floating):
                d = d & ~(np.isnan(a0[k]) & np.isnan(a1[k]))
            if d.any():
                ent = {"field": k, "n": int(d.sum()), "max_abs": float(np.abs(a1[k].astype(float) - a0[k].astype(float))[d].max())}
                if k == "dof_armature":
                    idx = np.nonzero(d)[0]
                    ent["dofs"] = [n0["dof_joint"][i] for i in idx]
                    ent["new_values"] = a1[k][idx].tolist()
                fields.append(ent)
        r = {"gripper_class_off": g0, "gripper_class_on": g1, "xml_diff_lines": dl, "n_xml_diff": len(dl),
             "model_fields_differ": fields}
        if body == "Panda":
            r["PASS"] = len(dl) == 0 and len(fields) == 0 and g0 == g1
        else:
            fj = {"gripper0_" + j for j in RB.R85_FINGER_JOINTS}
            minus = [l for l in dl if l.startswith("-")]
            plus = [l for l in dl if l.startswith("+")]
            ok_xml = (len(minus) == 6 and len(plus) == 6 and all(
                p.replace(f' armature="{args.arm}"', "")[1:] == m_[1:] for m_, p in zip(minus, plus))
                and all(any(f'name="{j}"' in p for p in plus) for j in fj))
            arm_f = [f for f in fields if f["field"] == "dof_armature"]
            ok_arm = len(arm_f) == 1 and sorted(arm_f[0]["dofs"]) == sorted(fj) and all(abs(v - float(args.arm)) < 1e-15 for v in arm_f[0]["new_values"])
            r["PASS"] = bool(ok_xml and ok_arm)
            r["xml_ok"], r["armature_ok"] = bool(ok_xml), bool(ok_arm)
        res[f"{sname}|{body}"] = r
        print(sname, body, "PASS" if r["PASS"] else "FAIL", "xml diff", len(dl), "fields", [f["field"] for f in fields], flush=True)
json.dump(res, open(args.out, "w"), indent=1)
print("XML_PF", "PASS" if all(v["PASS"] for k, v in res.items() if "|" in k) else "FAIL")
