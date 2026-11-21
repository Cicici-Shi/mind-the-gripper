"""R7 (protocol_block3.md 3.1): per-task contact-episode fraction c_t from the closed-loop replay of the 194
successful SRC episodes (embody.replay with online contacts, out/replay_b3), and the frozen predictions.
Rule: c_t >= 0.5 -> predict policy SR <= 50% on that task; c_t == 0 -> predict SR > 25%, which is only
scored if S (sham_cg) reaches >= 75% on that task (the S condition is checked at evaluation time, not here,
so no policy result is read). Calibration (must hold or no predictions are issued): UR5e task 3 c_t close
to 1 (>= 0.8), Panda c_t = 0 on every task.
Writes out/r7_predictions.tsv. Usage: python tools/r7_predict.py
"""
import glob, json, os, hashlib
import numpy as np
O = os.path.expanduser("$PROJ_ROOT/embody_v3/out/replay_b3")
bodies = ["Panda", "IIWA", "UR5e", "UR5e_nc", "PandaL3", "Kinova3", "PandaRT2"]
C = {}
for b in bodies:
    rows = [json.loads(l) for f in glob.glob(f"{O}/{b}_t[0-9].jsonl") for l in open(f)]
    for t in range(10):
        r = [x for x in rows if x["task"] == t]
        C[(b, t)] = (sum(x["contacts"]["arm_env_contact_steps"] > 0 for x in r), len(r),
                     sorted({p.split("|")[1] for x in r for p in x["contacts"]["pairs"]}))
calib_ur5e = C[("UR5e", 3)][0] / max(C[("UR5e", 3)][1], 1)
calib_panda = all(C[("Panda", t)][0] == 0 for t in range(10))
print(f"calibration: UR5e task 3 c_t = {calib_ur5e:.2f} (need >= 0.8); Panda all zero: {calib_panda}")
ok = calib_ur5e >= 0.8 and calib_panda
lines = ["body\ttask\tn_contact_eps\tn_eps\tc_t\tprediction\tcontact_bodies"]
for b in bodies:
    for t in range(10):
        k, n, who = C[(b, t)]
        c = k / n if n else float("nan")
        if b in ("Panda", "UR5e", "UR5e_nc", "IIWA"):
            pred = "calibration only"
        elif b == "PandaRT2":
            pred = "info only (gripper body, not in R7)"
        elif not ok:
            pred = "none (calibration failed)"
        elif c >= 0.5:
            pred = "SR <= 50%"
        elif c == 0:
            pred = "SR > 25% (scored only if S >= 75% on this task)"
        else:
            pred = "none"
        lines.append(f"{b}\t{t}\t{k}\t{n}\t{c:.3f}\t{pred}\t{','.join(who)}")
path = os.path.expanduser("$PROJ_ROOT/embody_v3/out/r7_predictions.tsv")
open(path, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
print("sha256", hashlib.sha256(open(path, "rb").read()).hexdigest(), path)
