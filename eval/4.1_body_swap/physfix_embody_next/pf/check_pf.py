"""physfix pre-collection checks (no outcome printed). Exit 0 only if every check passes.
  --cpu DIR      : xml_pf.json PASS; replay summary severe == 0 and qacc == 0 per cond (D_any only printed, 5.3)
  --regr DIR     : nx_regress outputs: <cell>_old (embody_toy) vs <cell>_new (embody_next off) qpos bitwise for every
                   key [(i)]; <cell>_new vs <cell>_newarm (R85_ARMATURE=1e-3) bitwise for Panda cells [(ii)]
  --smoke DIR... : launch_pf npz: pf_* fields present, pf_blind=1, pf_grip_contact rows == actions, armature fields
                   consistent with pf_cell_env; never reads `success`
  --logs FILE... : no outcome lines, no video lines
"""
import argparse, glob, json, os, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--cpu"); ap.add_argument("--regr"); ap.add_argument("--smoke", nargs="*", default=[]); ap.add_argument("--logs", nargs="*", default=[])
a = ap.parse_args()
ok = True
def rep(name, good, info=""):
    global ok
    ok &= bool(good)
    print(("PASS " if good else "FAIL ") + name, info, flush=True)
if a.cpu:
    x = json.load(open(os.path.join(a.cpu, "xml_pf.json")))
    for k, v in x.items():
        if "|" in k:
            rep(f"xml {k}", v["PASS"], f"xml_diff={v['n_xml_diff']} fields={[f['field'] for f in v['model_fields_differ']]}")
    s = json.load(open(os.path.join(a.cpu, "replay", "summary.json")))
    for c, v in s.items():
        print(f"INFO replay {c} (armature cells only, not a gate for S/T0/kp20)", f"n={v['n']} severe={v['severe']} qacc={v['qacc']} D_any={v['D_any']} keys={v['D_keys']} M_finger={v['M_finger']}", flush=True)
    g = json.load(open(os.path.join(a.cpu, "grip_sweep.json")))
    for k, v in g["out"].items():
        e = v["a_kp100"]
        rep(f"grip {k} armature applied", e["armature"] and all(abs(t - 1e-3) < 1e-12 for t in e["armature"]), f"fj_closed={e['fj_closed']:.4f} f100_closed={e['kp100']['f_closed']:.3f} f100_reopen={e['kp100']['f_reopen']:.3f}")
if a.regr:
    for d in sorted(glob.glob(os.path.join(a.regr, "*_new"))):
        base = d[:-4]
        for other, lab in ((base + "_old", "(i) next-off vs toy"), (base + "_newarm", "(ii) arm on vs off")):
            if not os.path.isdir(other):
                continue
            fs = sorted(glob.glob(os.path.join(d, "t*_e*.npz")))
            diffs = []
            for f in fs:
                o = os.path.join(other, os.path.basename(f))
                q1, q2 = np.load(f)["qpos"], np.load(o)["qpos"]
                diffs.append(float(np.abs(q1 - q2).max()) if q1.shape == q2.shape else float("inf"))
            rep(f"{lab} {os.path.basename(base)}", len(fs) > 0 and max(diffs) == 0.0, f"keys={len(fs)} maxdiff={max(diffs) if diffs else None}")
for d in a.smoke:
    fs = sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz")))
    bad = []
    for f in fs:
        z = np.load(f, allow_pickle=True)
        need = ("pf_qacc_count", "pf_qacc_first_step", "pf_r85_armature", "pf_grip_armature", "pf_blind", "pf_cell_env", "pf_grip_contact")
        miss = [k for k in need if k not in z.files]
        if miss:
            bad.append((os.path.basename(f), "missing", miss)); continue
        env = json.loads(str(z["pf_cell_env"]))
        arm = float(env.get("R85_ARMATURE") or 0)
        ga = np.asarray(z["pf_grip_armature"])
        good = int(z["pf_blind"]) == 1 and abs(float(z["pf_r85_armature"]) - arm) < 1e-15 and \
            (len(ga) == 0 if env.get("BODY") == "Panda" else (len(ga) == 6 and np.allclose(ga, arm, atol=0, rtol=0))) and \
            len(z["pf_grip_contact"]) == len(z["action"])
        if not good:
            bad.append((os.path.basename(f), env.get("BODY"), arm, ga.tolist(), len(z["pf_grip_contact"]), len(z["action"])))
    rep(f"smoke npz {d}", len(fs) > 0 and not bad, f"n={len(fs)} bad={bad[:3]}")
for f in a.logs:
    t = open(f, errors="replace").read()
    nbad = sum(t.count(b) for b in ("Success:", "# successes", "success rate", "Total successes", "Overall success rate", "Saved rollout MP4", "success="))
    ntb = t.count("Traceback")
    need_marker = os.path.basename(f).startswith("launch_")  # 2026-11-07: OFT EVAL-*.txt never carries PF_LAUNCH
    rep(f"blind log {os.path.basename(f)}", nbad == 0 and ("PF_LAUNCH" in t or not need_marker) and "Traceback" not in t, f"outcome_or_video_lines={nbad} tracebacks={ntb}")
print("CHECK_PF", "PASS" if ok else "FAIL")
sys.exit(0 if ok else 1)
