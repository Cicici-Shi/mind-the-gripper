"""Summarise the physfix replay re-check (phys_replay.py jsonl, never a dump's success). Same 'severe' rule as
diag_next/phys/summarize.py; detector D as in protocol 5.3 (add1 2.2). Compares with the runtime-patch arm1e-3 run
of WORKLOG_vlazoo 14 (diag_next/phys/out, out_spatial) key by key. Usage: python replay_summ_pf.py DIR"""
import glob, json, os, sys
import numpy as np
D = sys.argv[1]
rows = [json.loads(l) for f in glob.glob(os.path.join(D, "*.jsonl")) for l in open(f)]
old = {}
for f in glob.glob(os.path.expanduser("$PROJ_ROOT/diag_next/phys/out/*__arm1e-3.jsonl")) + glob.glob(os.path.expanduser("$PROJ_ROOT/diag_next/phys/out_spatial/*__arm1e-3.jsonl")):
    for l in open(f):
        r = json.loads(l); old[(r["cond"], r["task"], r["ep"], r["seed"])] = r
bd = lambda r: r["D_replay"]["first"] is not None or r["qacc_count"] > 0
sev = lambda r: bd(r) and (r["D_replay"]["A"] is not None or r["qacc_count"] > 0 or r["D_replay"]["finger_max"] > 2 or r["D_replay"]["finger_min"] < -2)
out = {}
for c in sorted({r["cond"] for r in rows}):
    R = [r for r in rows if r["cond"] == c]
    o = [old.get((r["cond"], r["task"], r["ep"], r["seed"])) for r in R]
    out[c] = {"n": len(R), "M_finger": sorted({round(r["M_finger"], 7) for r in R}), "D_any": sum(bd(r) for r in R),
              "severe": sum(sev(r) for r in R), "qacc": sum(r["qacc_count"] > 0 for r in R),
              "D_keys": sorted({f"{r['task']}:{r['ep']}:s{r['seed']}" for r in R if bd(r)}),
              "D_detail": [{"key": f"{r['task']}:{r['ep']}:s{r['seed']}", "A": r["D_replay"]["A"], "B": r["D_replay"]["B"],
                            "finger_min": round(r["D_replay"]["finger_min"], 3), "finger_max": round(r["D_replay"]["finger_max"], 3)} for r in R if bd(r)],
              "finger_min_p1": float(np.percentile([r["D_replay"]["finger_min"] for r in R], 1)),
              "runtime_patch_old_n": sum(x is not None for x in o),
              "runtime_patch_old_D": sum(bd(x) for x in o if x is not None),
              "same_D_flag_as_runtime_patch": sum((bd(r) == bd(x)) for r, x in zip(R, o) if x is not None)}
    print(c, json.dumps(out[c]))
json.dump(out, open(os.path.join(D, "summary.json"), "w"), indent=1)
