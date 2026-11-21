"""vla_zoo interface gate (2026-11-03), after embody_v3/tools/gate_s0.py (pi0.5 branch).

Per model and cell prints ONLY interface quantities, never a success rate for body cells:
  records / error records, model/proto/replan, host+GPU, scene identity vs the OFT src dump of the same key
  (pi0.5-style 'fix' rounded to 1e-5 m -> tolerance 0.01 mm), BodyEnv first-frame alignment (pos_mm / rot_deg,
  gate 0.01 mm / 0.05 deg), copy_grip_cmd (must be 1), fixture shift, missing objects, online contact log present.
--show-success only for native SRC cells (baseline alignment with the official number).
Usage: python zoo_gate.py --model NAME --cell CELL [--show-success]
"""
import argparse
import glob
import json
import os
import re

import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--cell", required=True)
ap.add_argument("--show-success", action="store_true")
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
args = ap.parse_args()
Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
ALIGN_RE = re.compile(r"BODY_ALIGN: body=(\S+) pos_mm=([\d.]+) rot_deg=([\d.]+) quat_sign=\S+ fixture_mm=([\d.]+) "
                      r"missing=(\[.*?\])(?:.*?copy_grip_cmd=(\d))?")

recs, errs = {}, []
for f in glob.glob(os.path.join(Z, "runs", args.model, args.cell, "t*_e*_i*.npz")):  # OFT-style dumps (run_oft_zoo)
    z = np.load(f, allow_pickle=True)
    m = re.search(r"t(\d+)_e(\d+)_i(\d+)", os.path.basename(f))
    k = (int(m.group(1)), int(m.group(2)))
    a = json.loads(str(z["align_json"])) if "align_json" in z.files else None
    recs[k] = {"task": k[0], "episode": k[1], "init": int(m.group(3)), "success": bool(z["success"]),
               "host": str(z["host"]) if "host" in z.files else None, "gpu": str(z["gpu"]) if "gpu" in z.files else None,
               "model": args.model, "proto": "oft-inproc", "replan": 8, "align": a,
               "fix": {str(n): list(z["ref_body_pos"][i]) for i, n in enumerate(z["ref_body_names"])},
               "contacts": {} if "contact_arm_env_contact_steps" in z.files else ({"error": 1} if "contact_error" in z.files else None),
               "steps": len(z["action"]) if "action" in z.files else 0, "queries": 0, "wall_s": 0.0}
for f in glob.glob(os.path.join(Z, "runs", args.model, f"{args.cell}_e*.jsonl")):
    for line in open(f):
        if line.strip():
            r = json.loads(line)
            if r.get("error") is not None:
                errs.append(((r["task"], r["episode"]), r["error"][:120]))
                continue
            recs.setdefault((r["task"], r["episode"]), r)
log_align = {}
for f in glob.glob(os.path.join(Z, "logs", f"c_{args.model}_{args.cell}_*_e*.log")):
    pend = None
    for line in open(f, errors="replace"):
        m = ALIGN_RE.search(line)
        if m:
            pend = dict(pos_mm=float(m.group(2)), rot_deg=float(m.group(3)), fixture_mm=float(m.group(4)),
                        missing=m.group(5), copy_grip_cmd=int(m.group(6)) if m.group(6) else -1)
        if line.startswith("EP {") and pend is not None:
            r = json.loads(line[3:line.rindex("}") + 1])
            log_align[(r["task"], r["episode"])] = pend
            pend = None


def src_fix(k):
    srcdir = "obj_src" if args.cell.startswith("obj_") else "src"   # published OFT native dumps of that suite
    fs = glob.glob(os.path.join(args.runs, srcdir, f"t{k[0]}_e{k[1]}_i*.npz"))
    if not fs:
        return None
    z = np.load(fs[0], allow_pickle=True)
    return {str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])}, int(re.search(r"_i(\d+)", fs[0]).group(1))


fixmax, nfix, init_mismatch, cont_ok, cont_err, hosts, metas = 0.0, 0, 0, 0, 0, set(), set()
for k, r in sorted(recs.items()):
    hosts.add((r.get("host"), r.get("gpu")))
    metas.add((r.get("model"), r.get("proto"), r.get("replan")))
    s = src_fix(k)
    if s is not None and r.get("fix"):
        ref, init = s
        nfix += 1
        init_mismatch += int(init != r["init"])
        for n, p in r["fix"].items():
            if not n or n == "world" or n.startswith(("robot0", "gripper0", "mount0")) or n not in ref:
                continue
            fixmax = max(fixmax, float(np.abs(np.asarray(p, float) - ref[n]).max()) * 1000)
    c = r.get("contacts")
    if c is not None and "error" not in c:
        cont_ok += 1
    elif c:
        cont_err += 1
print(f"== {args.model} / {args.cell}: records {len(recs)} (keys {sorted(recs)[:3]}...), error records {len(errs)} {errs[:3]}")
print(f"model/proto/replan {sorted(metas, key=str)}; hosts/gpus {sorted(hosts, key=str)}")
print(f"scene vs OFT src same key: {nfix} keys compared, init-id mismatches {init_mismatch}, max fixture diff {fixmax:.4f} mm (GATE <= 0.01)")
print(f"online contact log present {cont_ok}/{len(recs)}; contact errors {cont_err}")
al = [r.get("align") or log_align.get(k) for k, r in recs.items()]
al = [a for a in al if a]
if al:
    cg = [int(a.get("copy_grip_cmd", -1)) for a in al]
    print(f"BodyEnv align {len(al)} eps: copy_grip_cmd=1 {cg.count(1)}/{len(cg)} (GATE all) | max pos_mm "
          f"{max(float(a['pos_mm']) for a in al):.4f} max rot_deg {max(float(a['rot_deg']) for a in al):.4f} (GATE 0.01 / 0.05) | "
          f"max fixture shift {max(float(a['fixture_mm']) for a in al):.3f} mm | eps with missing objects "
          f"{sum(1 for a in al if str(a.get('missing')) not in ('[]', 'None'))}")
else:
    print("BodyEnv align: none (native SRC cell)" if args.cell.startswith("src") else "BodyEnv align: NONE FOUND")
ms = [r["steps"] for r in recs.values()]
q = [r["queries"] for r in recs.values()]
w = [r["wall_s"] for r in recs.values()]
rtt = [r["rtt_ms_mean"] for r in recs.values() if r.get("rtt_ms_mean")]
print(f"[note: OFT npz has no wall/query fields] " if any(r.get("proto") == "oft-inproc" for r in recs.values()) else "", end="")
print(f"wall_s/episode median {np.median(w):.1f}; queries/episode median {np.median(q):.0f}; rtt ms median {np.median(rtt):.0f}" if recs else "")
if args.show_success:
    assert args.cell.split("obj_")[-1].startswith("src"), "success only printed for native SRC baseline cells"
    n = len(recs); s = sum(r["success"] for r in recs.values())
    per = {}
    for (t, e), r in recs.items():
        per.setdefault(t, []).append(int(r["success"]))
    print(f"SRC success {s}/{n} = {100 * s / max(n, 1):.1f}% ; per task " + " ".join(f"{t}:{sum(v)}/{len(v)}" for t, v in sorted(per.items())))
