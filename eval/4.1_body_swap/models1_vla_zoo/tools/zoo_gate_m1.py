"""models1 interface gates (protocol_models1.md 7.1, 7.2, 7.4, 7.5), 2026-11-04. Extends common/zoo_gate.py.
Prints ONLY interface quantities and key-wise agreement rates; never a success rate of any non-src cell.
Per cell: records (only complete keys of the requested shards), error records, missing keys per shard, init-id and
fixture identity vs the published OFT native dump of the same key, copy_grip_cmd, first-frame alignment, contact log,
host/GPU composition (client and, for server models, server), replan / chunk_len / RTT (server models), twin camera
error at query frames, IK residual at query frames, stale frames, per-episode wall time ratio twin/native.
Agreement (7.2 / 5.4): T0 vs S same-key outcome agreement (gate >= 90%); OVLA E1 agree(GTw, G), E2 agree(GTa, GTfc)
outcome agreement and number of keys whose first executed block is bitwise identical.
Usage: python zoo_gate_m1.py --model pi0|openvla|oft_sp|oft_obj --suite libero_spatial|libero_object --offsets 0 [--json OUT]
"""
import argparse, glob, json, os, re
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--suite", default="libero_spatial")
ap.add_argument("--offsets", default="0")
ap.add_argument("--trials", type=int, default=5)
ap.add_argument("--cells", default="sham_cg,sham_twin,grpkp100,g_tfc,g_tw_fc,g_ta_fc,g_t")
ap.add_argument("--runs", default=os.path.expanduser("$PROJ_ROOT/runs"))
ap.add_argument("--json", default="")
args = ap.parse_args()
Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
OBJ = args.suite == "libero_object"
PRE = "obj_" if OBJ else ""
OFFS = [int(x) for x in args.offsets.split(",")]
KEYS = {(t, e) for t in range(10) for o in OFFS for e in range(o, o + args.trials)}
TWIN = {"sham_twin", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"}
PANDA = {"sham_cg", "sham_twin"}
SYNC_CAM = {"g_tfc", "g_tw_fc"}   # TWIN_CAM=phys cells: twin wrist camera on the physical one (<= 0.01 mm / 0.01 deg); T0 uses TWIN_CAM=twin (protocol table, = published sham_twin), its camera error is the IK residual, printed only
INPROC = args.model.startswith("oft")


def load_npz(cell):
    out = {}
    for f in glob.glob(os.path.join(Z, "runs", args.model, PRE + cell, "t*_e*_i*.npz")):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        if k not in KEYS:
            continue
        z = np.load(f, allow_pickle=True)
        tw = None
        if "twin_pos_err_mm" in z.files:
            pe = np.asarray(z["twin_pos_err_mm"]); re_ = np.asarray(z["twin_rot_err_deg"])
            ce = np.asarray(z["twin_cam_err_mm"]) if "twin_cam_err_mm" in z.files else None
            cd = np.asarray(z["twin_cam_err_deg"]) if "twin_cam_err_deg" in z.files else None
            tw = {"pos_q": pe[::8].tolist(), "rot_q": re_[::8].tolist(), "stale": int(z["twin_stale"]),
                  "cam_mm_q": None if ce is None else ce[::8].tolist(), "cam_deg_q": None if cd is None else cd[::8].tolist(),
                  "cam_mm_mean": None if ce is None else float(ce.mean())}
        out[k] = {"init": int(m.group(3)), "success": bool(z["success"]), "host": str(z["host"]), "gpu": str(z["gpu"]),
                  "align": json.loads(str(z["align_json"])) if "align_json" in z.files else None,
                  "fix": {str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])},
                  "contacts": "contact_arm_env_contact_steps" in z.files, "twin": tw, "mtime": os.path.getmtime(f),
                  "shard": max(o for o in OFFS if o <= k[1]), "a0": np.asarray(z["action"][:8]).tolist() if "action" in z.files else None}
    return out, []


def load_jsonl(cell):
    out, errs = {}, []
    for f in glob.glob(os.path.join(Z, "runs", args.model, f"{PRE}{cell}_e*.jsonl")):
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            k = (r["task"], r["episode"])
            if k not in KEYS:
                continue
            if r.get("error") is not None:
                errs.append((k, r["error"][:100]))
                continue
            tw = r.get("twin")
            if tw is not None:
                tw = {"pos_q": tw.get("pos_err_mm_q", []), "rot_q": [], "stale": tw.get("stale", 0),
                      "cam_mm_q": tw.get("cam_err_mm_q"), "cam_deg_q": tw.get("cam_err_deg_q"), "cam_mm_mean": tw.get("cam_err_mm_mean"),
                      "miss": tw.get("miss", 0), "n_paint": tw.get("n_paint", 0)}
            out.setdefault(k, {"init": r["init"], "success": bool(r["success"]), "host": r.get("host"), "gpu": r.get("gpu"),
                               "align": r.get("align"), "fix": {n: np.asarray(p) for n, p in (r.get("fix") or {}).items()},
                               "contacts": bool(r.get("contacts")) and "error" not in (r.get("contacts") or {}),
                               "twin": tw, "wall": r.get("wall_s"), "replan": r.get("replan"), "chunk_len": r.get("chunk_len"),
                               "rtt": r.get("rtt_ms_med"), "srv": (r.get("server_host"), r.get("server_gpu")), "a0": r.get("a0"),
                               "embody_dir": r.get("embody_dir"), "shard": max(o for o in OFFS if o <= k[1])})
    return out, errs


_src_cache = {}


def src_ref(k):
    if k not in _src_cache:
        fs = glob.glob(os.path.join(args.runs, "obj_src" if OBJ else "src", f"t{k[0]}_e{k[1]}_i*.npz"))
        if not fs:
            _src_cache[k] = None
        else:
            z = np.load(fs[0], allow_pickle=True)
            _src_cache[k] = ({str(n): z["ref_body_pos"][i] for i, n in enumerate(z["ref_body_names"])},
                             int(re.search(r"_i(\d+)", fs[0]).group(1)))
    return _src_cache[k]


cells = [c for c in args.cells.split(",") if c]
data, report = {}, {"model": args.model, "suite": args.suite, "offsets": OFFS, "cells": {}}
for c in cells:
    recs, errs = (load_npz if INPROC else load_jsonl)(c)
    data[c] = recs
    R = {"records": len(recs), "error_records": len(errs), "errors": errs[:3],
         "missing_per_shard": {o: sorted(k for k in KEYS if k not in recs and o <= k[1] < o + args.trials) for o in OFFS}}
    R["missing_per_shard"] = {o: len(v) for o, v in R["missing_per_shard"].items()}
    fixmax, init_mis, ncmp = 0.0, 0, 0
    for k, r in recs.items():
        s = src_ref(k)
        if s is None or not r["fix"]:
            continue
        ref, init = s
        ncmp += 1
        init_mis += int(init != r["init"])
        for n, p in r["fix"].items():
            if not n or n == "world" or n.startswith(("robot0", "gripper0", "mount0")) or n not in ref:
                continue
            fixmax = max(fixmax, float(np.abs(np.asarray(p, float) - ref[n]).max()) * 1000)
    R.update(scene_keys=ncmp, init_mismatch=init_mis, fixture_max_mm=round(fixmax, 5),
             scene_gate=bool(init_mis == 0 and fixmax <= (0.0001 if INPROC else 0.01)))
    al = [r["align"] for r in recs.values() if r["align"]]
    cg = [int(a.get("copy_grip_cmd", -1)) for a in al]
    R.update(align_n=len(al), copy_grip_cmd_1=cg.count(1), copy_gate=bool(len(al) == len(recs) and cg.count(1) == len(cg)))
    if al:
        pm, rd = max(float(a["pos_mm"]) for a in al), max(float(a["rot_deg"]) for a in al)
        R.update(align_pos_mm_max=round(pm, 4), align_rot_deg_max=round(rd, 4),
                 align_gate=(bool(pm <= 0.01 and rd <= 0.05) if c in PANDA else "print-only (PandaR85, appendix 2 ruling 1)"))
    R["contacts_present"] = sum(r["contacts"] for r in recs.values())
    R["contacts_gate"] = R["contacts_present"] == len(recs)
    from collections import Counter
    R["host_gpu"] = dict(Counter(f"{r['host']}|{r['gpu']}" for r in recs.values()))
    R["gpu_models"] = sorted({r["gpu"] for r in recs.values()})
    if not INPROC:
        R["server"] = sorted({f"{h}|{g}" for (h, g) in (r["srv"] for r in recs.values())})
        R["replan"] = sorted({r["replan"] for r in recs.values()}, key=str)
        R["chunk_len"] = sorted({r["chunk_len"] for r in recs.values()}, key=str)
        rt = [r["rtt"] for r in recs.values() if r["rtt"]]
        R["rtt_ms_median"] = round(float(np.median(rt)), 1) if rt else None
        R["eps_rtt_over_500"] = sum(1 for x in rt if x > 500)
        w = [r["wall"] for r in recs.values() if r["wall"]]
        R["wall_s_median"] = round(float(np.median(w)), 1) if w else None
        R["embody_dir"] = sorted({str(r["embody_dir"]) for r in recs.values()})
        R["error_rate_gate"] = len(errs) <= 0.01 * max(len(recs), 1)
    else:
        per = {}
        for r in recs.values():
            per.setdefault(r["shard"], []).append(r["mtime"])
        d = [x for v in per.values() for x in np.diff(sorted(v))]
        R["wall_s_median_from_mtime"] = round(float(np.median(d)), 1) if d else None
    if c in TWIN:
        tws = [r["twin"] for r in recs.values() if r["twin"]]
        R["twin_records"] = len(tws)
        pq = [x for t in tws for x in t["pos_q"]]
        R["ik_query_frames"] = len(pq)
        R["ik_over_1mm_frac"] = round(sum(1 for x in pq if x > 1) / max(len(pq), 1), 5)
        R["ik_over_2mm"] = sum(1 for x in pq if x > 2)
        R["ik_gate"] = R["ik_over_1mm_frac"] < 0.01
        R["stale_total"] = sum(t["stale"] for t in tws)
        cm = [x for t in tws for x in (t["cam_mm_q"] or [])]
        cdg = [x for t in tws for x in (t["cam_deg_q"] or [])]
        R["cam_err_mm_q_max"] = round(max(cm), 5) if cm else None
        R["cam_err_deg_q_max"] = round(max(cdg), 5) if cdg else None
        R["cam_err_mm_mean"] = round(float(np.mean([t["cam_mm_mean"] for t in tws if t["cam_mm_mean"] is not None])), 4) if tws else None
        if c in SYNC_CAM:
            R["cam_gate"] = bool(cm and max(cm) <= 0.01 and max(cdg) <= 0.01)
    report["cells"][c] = R


def ratio(a, b):
    ka = report["cells"].get(a, {}); kb = report["cells"].get(b, {})
    x = ka.get("wall_s_median", ka.get("wall_s_median_from_mtime")); y = kb.get("wall_s_median", kb.get("wall_s_median_from_mtime"))
    return round(x / y, 2) if x and y else None


report["wall_ratio"] = {"sham_twin/sham_cg": ratio("sham_twin", "sham_cg"),
                        **{f"{t}/grpkp100": ratio(t, "grpkp100") for t in ("g_tfc", "g_tw_fc", "g_ta_fc", "g_t")}}


def agree(a, b):
    if a not in data or b not in data:
        return None
    ks = sorted(set(data[a]) & set(data[b]))
    same = sum(data[a][k]["success"] == data[b][k]["success"] for k in ks)
    a0 = sum(1 for k in ks if data[a][k].get("a0") is not None and data[a][k]["a0"] == data[b][k]["a0"])
    return {"keys": len(ks), "outcome_agree": round(same / max(len(ks), 1), 4), "a0_bitwise_identical": a0}


report["agree"] = {"T0_vs_S": agree("sham_twin", "sham_cg")}
if report["agree"]["T0_vs_S"]:
    report["agree"]["T0_vs_S"]["gate_ge_0.90"] = report["agree"]["T0_vs_S"]["outcome_agree"] >= 0.90
if args.model == "openvla":
    report["agree"]["E1_GTw_vs_G"] = agree("g_tw_fc", "grpkp100")
    report["agree"]["E2_GTa_vs_GTfc"] = agree("g_ta_fc", "g_tfc")
for c, R in report["cells"].items():
    print(f"== {args.model} {PRE}{c}: " + json.dumps(R, default=str))
print("wall_ratio " + json.dumps(report["wall_ratio"]))
print("agree " + json.dumps(report["agree"]))
if args.json:
    json.dump(report, open(args.json, "w"), indent=1, default=str)
