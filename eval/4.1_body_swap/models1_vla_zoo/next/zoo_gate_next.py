"""next (2026-11-06): BLIND interface gate for the next model batch (pi0-FAST, SmolVLA). Derived from the frozen
tools/zoo_gate_m1.py (sha256 38973373...) jsonl branch, plus tools/PATCH_NEXT_zoo_client_rot.md and detector D.
NEVER reads `success`, `steps`, `queries` or `wall_s` (asserts the records are --blind ones), and prints nothing that
reveals episode length: no frame / paint totals, no per-episode wall time. Throughput is per control step.
Per cell: records, error records, missing keys, init id and fixtures vs the published OFT native dump of the same key,
copy_grip_cmd, first-frame alignment, contact log presence, host/GPU and server composition, replan / chunk_len,
RTT, twin camera error at query frames (gate for TWIN_CAM=phys cells), IK residual fractions (pos > 1 mm gate < 1%;
pos > 2 mm or rot > 1 deg printed), stale frames, physics breakdown detector D and MuJoCo BADQACC, and per-step
loop / sim time ratio twin cell / its native cell. Agreement: only the first executed action block (a0) bitwise,
which is an input-level quantity (no outcome).
Usage: python zoo_gate_next.py --model pi0fast --root runs/next_smoke [--suite libero_spatial] [--keys t:e,...] [--json OUT]
"""
import argparse, glob, json, os, re
from collections import Counter
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--root", default="runs/next_smoke")
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--keys", default="", help="t:e,... ; default = every key found")
ap.add_argument("--cells", default="sham_cg,sham_twin,grpkp100,g_tfc,g_tw_fc,g_ta_fc,g_t")
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--json", default="")
args = ap.parse_args()
Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
PRE = {"libero_object": "obj_", "libero_goal": "goal_", "libero_10": "l10_"}.get(args.suite, "")
SRCDIR = {"libero_spatial": "src", "libero_object": "obj_src"}.get(args.suite)
KEYS = None if not args.keys else {tuple(int(x) for x in k.split(":")) for k in args.keys.split(",")}
TWIN = {"sham_twin", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"}
PANDA = {"sham_cg", "sham_twin"}
SYNC_CAM = {"g_tfc", "g_tw_fc"}
NATIVE = {"sham_twin": "sham_cg", "g_tfc": "grpkp100", "g_tw_fc": "grpkp100", "g_ta_fc": "grpkp100", "g_t": "grpkp100"}
FORBIDDEN = ("success", "steps", "queries", "wall_s")


def load(cell):
    out, errs = {}, []
    for f in sorted(glob.glob(os.path.join(Z, args.root, args.model, f"{PRE}{cell}_e*.jsonl"))):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            assert r.get("blind") is True and not any(k in r for k in FORBIDDEN), f"non-blind record in {f}; refusing"
            k = (r["task"], r["episode"])
            if KEYS is not None and k not in KEYS:
                continue
            if r.get("error") is not None:
                errs.append((k, r["error"][:100]))
                continue
            out.setdefault(k, r)
    return out, errs


_src = {}


def src_ref(k):
    if SRCDIR is None:
        return None
    if k not in _src:
        fs = glob.glob(os.path.join(args.runs, SRCDIR, f"t{k[0]}_e{k[1]}_i*.npz"))
        if not fs:
            _src[k] = None
        else:
            z = np.load(fs[0], allow_pickle=True)   # lazy: only ref_body_names / ref_body_pos are touched
            _src[k] = ({str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])},
                       int(re.search(r"_i(\d+)", fs[0]).group(1)))
    return _src[k]


def med(x):
    x = [v for v in x if v is not None]
    return round(float(np.median(x)), 2) if x else None


cells = [c for c in args.cells.split(",") if c]
data, report = {}, {"model": args.model, "suite": args.suite, "root": args.root, "cells": {}}
for c in cells:
    recs, errs = load(c)
    data[c] = recs
    R = {"records": len(recs), "error_records": len(errs), "errors": errs[:3], "keys": sorted(recs)}
    fixmax, init_mis, ncmp = 0.0, 0, 0
    for k, r in recs.items():
        s = src_ref(k)
        if s is None or not r.get("fix"):
            continue
        ref, init = s
        ncmp += 1
        init_mis += int(init != r["init"])
        for n, p in r["fix"].items():
            if not n or n == "world" or n.startswith(("robot0", "gripper0", "mount0")) or n not in ref:
                continue
            fixmax = max(fixmax, float(np.abs(np.asarray(p, float) - ref[n]).max()) * 1000)
    R.update(scene_keys=ncmp, init_mismatch=init_mis, fixture_max_mm=round(fixmax, 5),
             scene_gate=bool(ncmp == len(recs) and init_mis == 0 and fixmax <= 0.01) if SRCDIR else "no published SRC dump")
    al = [r["align"] for r in recs.values() if r.get("align")]
    cg = [int(a.get("copy_grip_cmd", -1)) for a in al]
    R.update(align_n=len(al), copy_grip_cmd_1=cg.count(1), copy_gate=bool(len(al) == len(recs) and cg.count(1) == len(cg)))
    if al:
        pm, rd = max(float(a["pos_mm"]) for a in al), max(float(a["rot_deg"]) for a in al)
        R.update(align_pos_mm_max=round(pm, 4), align_rot_deg_max=round(rd, 4),
                 align_gate=(bool(pm <= 0.01 and rd <= 0.05) if c in PANDA else "print-only (PandaR85, appendix 2 ruling 1)"))
    ct = [r.get("contacts") for r in recs.values()]
    R["contacts_present"] = sum(1 for x in ct if isinstance(x, dict) and "error" not in x)
    R["contacts_gate"] = R["contacts_present"] == len(recs)
    R["client_host_gpu"] = dict(Counter(f"{r.get('host')}|{r.get('gpu')}" for r in recs.values()))
    R["server"] = sorted({f"{r.get('server_host')}|{r.get('server_gpu')}" for r in recs.values()})
    R["embody_dir"] = sorted({str(r.get("embody_dir")) for r in recs.values()})
    R["cell_env"] = sorted({json.dumps(r.get("cell_env"), sort_keys=True) for r in recs.values()})
    R["replan"] = sorted({r.get("replan") for r in recs.values()}, key=str)
    R["chunk_len"] = sorted({r.get("chunk_len") for r in recs.values()}, key=str)
    R["fastsync"] = sorted({r.get("fastsync") for r in recs.values()}, key=str)
    tm = [r.get("timing") or {} for r in recs.values()]
    R["rtt_ms_med"] = med([r.get("rtt_ms_med") for r in recs.values()])
    R["server_infer_ms_med"] = med([t.get("server_infer_ms_med") for t in tm])
    R["sim_ms_step_med"] = med([t.get("sim_ms_step_med") for t in tm])
    R["loop_ms_step_med"] = med([t.get("loop_ms_step_mean") for t in tm])
    R["error_rate_gate"] = len(errs) <= 0.01 * max(len(recs), 1)
    D = [r.get("breakdown_D") or {} for r in recs.values()]
    R["D_breakdown_eps"] = sum(1 for d in D if d.get("breakdown"))
    R["D_arm_eps"] = sum(1 for d in D if d.get("A"))
    R["D_finger_eps"] = sum(1 for d in D if d.get("B"))
    R["D_finger_loose_eps"] = sum(1 for d in D if d.get("B_loose"))
    R["D_qacc_eps"] = sum(1 for d in D if (d.get("qacc_warn") or 0) > 0)
    R["D_qacc_unreadable"] = sum(1 for d in D if d.get("qacc_warn") is None)
    R["D_finger_joint_present"] = sorted({d.get("finger_joint") for d in D}, key=str)
    R["D_max_jump_deg"] = max([d.get("max_jump_deg", 0) for d in D] or [0])
    if c in TWIN:
        tws = [r["twin"] for r in recs.values() if r.get("twin")]
        R["twin_records"] = len(tws)
        pq = [x for t in tws for x in t.get("pos_err_mm_q", [])]
        rq = [x for t in tws for x in t.get("rot_err_deg_q", [])]
        pr = [(a, b) for t in tws for a, b in zip(t.get("pos_err_mm_q", []), t.get("rot_err_deg_q", []))]
        R["ik_over_1mm_frac"] = round(sum(1 for x in pq if x > 1) / max(len(pq), 1), 5)
        R["ik_gate"] = bool(pq) and R["ik_over_1mm_frac"] < 0.01
        R["ik_miss_2mm_or_1deg_frac"] = round(sum(1 for a, b in pr if a > 2 or b > 1) / max(len(pr), 1), 5)
        if pq:
            R["ik_pos_mm_q_max"] = round(max(pq), 4)
            R["ik_pos_mm_q_p99"] = round(float(np.percentile(pq, 99)), 4)
        if rq:
            R["ik_rot_deg_q_max"] = round(max(rq), 4)
        R["eps_with_ik_over_1mm"] = sum(1 for t in tws if any(x > 1 for x in t.get("pos_err_mm_q", [])))
        R["stale_total"] = sum(t.get("stale", 0) for t in tws)
        R["stale_gate"] = R["stale_total"] == 0
        cm = [x for t in tws for x in (t.get("cam_err_mm_q") or [])]
        cdg = [x for t in tws for x in (t.get("cam_err_deg_q") or [])]
        R["cam_err_mm_q_max"] = round(max(cm), 5) if cm else None
        R["cam_err_deg_q_max"] = round(max(cdg), 5) if cdg else None
        cmm = [t["cam_err_mm_mean"] for t in tws if t.get("cam_err_mm_mean") is not None]
        R["cam_err_mm_mean"] = round(float(np.mean(cmm)), 4) if cmm else None
        if c in SYNC_CAM:
            R["cam_gate"] = bool(cm and max(cm) <= 0.01 and max(cdg) <= 0.01)
        t0s = [t.get("t0") for t in tws if t.get("t0")]
        if t0s:
            R["twin_t0_agent_mean_max"] = round(max(x["agentview_image"][1] for x in t0s), 4)
            R["twin_t0_wrist_mean_max"] = round(max(x["robot0_eye_in_hand_image"][1] for x in t0s), 4)
    report["cells"][c] = R


def ratio(a, b, f):
    x = report["cells"].get(a, {}).get(f); y = report["cells"].get(b, {}).get(f)
    return round(x / y, 2) if x and y else None


report["per_step_ratio_twin_over_native"] = {f"{a}/{b}": {"loop": ratio(a, b, "loop_ms_step_med"), "sim": ratio(a, b, "sim_ms_step_med")}
                                             for a, b in NATIVE.items() if a in cells and b in cells}


def a0_same(a, b):
    if a not in data or b not in data:
        return None
    ks = sorted(set(data[a]) & set(data[b]))
    return {"keys": len(ks), "a0_bitwise_identical": sum(1 for k in ks if data[a][k].get("a0") is not None and data[a][k]["a0"] == data[b][k]["a0"])}


report["a0"] = {"T0_vs_S": a0_same("sham_twin", "sham_cg"), "GTw_vs_G": a0_same("g_tw_fc", "grpkp100"),
                "GTa_vs_GTfc": a0_same("g_ta_fc", "g_tfc")}
for c, R in report["cells"].items():
    R2 = {k: v for k, v in R.items() if k != "keys"}
    print(f"== {args.model} {PRE}{c}: " + json.dumps(R2, default=str))
print("per_step_ratio " + json.dumps(report["per_step_ratio_twin_over_native"]))
print("a0 " + json.dumps(report["a0"]))
if args.json:
    json.dump(report, open(args.json, "w"), indent=1, default=str)
