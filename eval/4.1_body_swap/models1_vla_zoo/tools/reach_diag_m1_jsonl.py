"""reach_diag_m1 companion: per-cell twin residual distribution from jsonl (pi0 / OVLA). Reads only task, episode,
error and twin.pos_err_mm_q; never the success field. Prints no per-key episode length."""
import glob, json, os, sys
import numpy as np
Z = os.path.expanduser("$PROJ_ROOT/vla_zoo")
out = {}
for model, cells in [("openvla", ["g_tfc", "g_tw_fc", "g_ta_fc"]), ("pi0", ["g_tfc", "g_tw_fc", "g_ta_fc"])]:
    for cell in cells:
        f = os.path.join(Z, "runs", model, f"{cell}_e0.jsonl")
        per = {}
        for line in open(f):
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("error") is not None or not (0 <= r["episode"] < 5):
                continue
            per[(r["task"], r["episode"])] = np.asarray((r.get("twin") or {}).get("pos_err_mm_q", []), float)
        allv = np.concatenate(list(per.values()))
        ov = allv[allv > 1]
        cnt = sorted([(int((v > 1).sum()), k) for k, v in per.items()], reverse=True)
        cum = np.cumsum([c for c, _ in cnt]); tot = cum[-1] if len(cum) else 0
        k80 = int(np.searchsorted(cum, 0.8 * tot) + 1) if tot else 0
        bytask = {}
        for c, k in cnt:
            bytask[k[0]] = bytask.get(k[0], 0) + c
        res = {"keys": len(per), "frames": int(len(allv)), "over1": int(len(ov)), "over1_frac": round(len(ov) / len(allv), 5),
               "over2": int((allv > 2).sum()), "over5": int((allv > 5).sum()), "over10": int((allv > 10).sum()),
               "over1_p50_p90_p99_max": [round(float(x), 2) for x in np.percentile(ov, [50, 90, 99, 100])] if len(ov) else None,
               "keys_with_over1": sum(1 for c, _ in cnt if c), "min_keys_for_80pct": k80,
               "over1_by_task": {int(t): c for t, c in sorted(bytask.items()) if c}}
        out[f"{model}/{cell}"] = res
        print(model, cell, json.dumps(res), flush=True)
json.dump(out, open(os.path.join(Z, "out", "reach_diag_m1", "jsonl_resid_s0.json"), "w"), indent=1)
