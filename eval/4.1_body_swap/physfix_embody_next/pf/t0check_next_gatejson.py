"""Build gate_physfix_t0.json (format: cells.<name>.status / items, as models1) from t0check_next.py jsonl.
Refuses (exit 2, no json) unless every record's gl_renderer contains H200. Lines (protocol_physfix 7.2): T0 vs S both
cameras < 0.11; GTfc-type wrist both-hidden < 1.0, agent excess (actual - M) <= 0.02, physical joints vs G 0; background diff (outside gripper+arm union)
printed as NA = descriptive (protocol_physfix rev3 7.2 / impl list item 13, 2026-11-08); GTloc': arm-region diff 0 and outside diff 0 (gap printed only). Camera error <= 0.01 mm for TWIN_CAM=phys cells.
Usage: python t0check_next_gatejson.py --spatial S.jsonl --object O.jsonl --regr-ok 0|1 --out gate_physfix_t0.json"""
import argparse, json, sys
import numpy as np
ap = argparse.ArgumentParser()
ap.add_argument("--spatial", required=True); ap.add_argument("--object", required=True)
ap.add_argument("--regr-ok", type=int, required=True); ap.add_argument("--out", required=True)
a = ap.parse_args()
A, W = "agentview_image", "robot0_eye_in_hand_image"
cells, rens = {}, set()
P = lambda ok: "PASS" if ok else "FAIL"
for pre, f in (("pf_", a.spatial), ("pf_obj_", a.object)):
    R = [json.loads(l) for l in open(f) if l.strip()]
    rens |= {r["gl_renderer"] for r in R}
    n = {"status": "NA", "value": len(R)}
    if R and "t0_vs_s" in R[0]:
        ma = float(np.mean([r["t0_vs_s"][A]["mean"] for r in R])); mw = float(np.mean([r["t0_vs_s"][W]["mean"] for r in R]))
        cells[pre + "sham_twin"] = {"items": {"t0_vs_s_agent": {"status": P(ma < 0.11), "value": round(ma, 4)},
                                              "t0_vs_s_wrist": {"status": P(mw < 0.11), "value": round(mw, 4)}, "keys": n}}
    for tn in ("g_tfc_a", "g_tfc_a_m20", "g_tfc_kp20", "g_tloc_a"):
        X = [r[tn] for r in R if tn in r]
        if not X:
            continue
        it = {"keys": {"status": "NA", "value": len(X)}}
        cam = max(x["cam_err_mm0"] or 0 for x in X)
        it["cam_err_mm0_max"] = {"status": P(cam <= 0.01), "value": cam}
        jd = max(x["phys_joint_diff_vs_G"] for x in X)
        it["twin_phys_joint_diff"] = {"status": P(jd == 0), "value": jd}
        if tn == "g_tloc_a":
            for c in ("agent", "robot"):
                mx = max(x[f"{c}_arm_region"]["max"] for x in X); mo = max(x[f"{c}_outside"]["max"] for x in X)
                it[f"{c}_arm_region_max"] = {"status": P(mx == 0), "value": mx}
                it[f"{c}_outside_max"] = {"status": P(mo == 0), "value": mo}
                gp = [x[f"{c}_gap_px"] for x in X if x[f"{c}_gap_px"] is not None]
                it[f"{c}_gap_px"] = {"status": "NA", "value": {"median": float(np.median(gp)) if gp else None, "max": float(max(gp)) if gp else None, "n": len(gp)}}
        else:
            w = float(np.mean([x["robot_both_hidden"]["mean"] for x in X]))
            ag = float(np.mean([x["agent_both_hidden"]["mean"] for x in X]))
            Ms = [x["M_search"]["M"] for x in X]
            ok_m = all(m is not None for m in Ms)
            M = float(np.mean(Ms)) if ok_m else None
            exc = [x["agent_excess"] for x in X]
            it["wrist_both_hidden_mean"] = {"status": P(w < 1.0), "value": round(w, 4)}
            it["agent_both_hidden_mean"] = {"status": "NA", "value": round(ag, 4)}
            it["agent_lb"] = {"status": "NA", "value": M, "note": "finite IK search minimum under the chosen geometric constraint, not a theoretical lower bound"}
            it["agent_excess"] = {"status": P(ok_m and ag - M <= 0.02), "value": None if not ok_m else round(ag - M, 4)}
            it["agent_max_key_excess"] = {"status": "NA", "value": None if not ok_m else round(max(exc), 4)}
            bg = max(x["agent_background"]["max"] for x in X)
            it["background_identical"] = {"status": "NA", "value": bg, "note": "descriptive (protocol 7.2 rev3, impl item 13); was a gate (==0) before 2026-11-08"}
        cells[pre + tn] = {"items": it}
for v in cells.values():
    v["status"] = "FAIL" if any(i["status"] == "FAIL" for i in v["items"].values()) else "PASS"
if not rens or not all("H200" in r for r in rens):
    print("refuse: GL_RENDERER not H200:", rens); sys.exit(2)
out = {"gl_renderer": sorted(rens)[0], "source": [a.spatial, a.object],
       "regression": {"status": P(a.regr_ok), "items": {}}, "cells": cells}
json.dump(out, open(a.out, "w"), indent=1)
for k, v in cells.items():
    print(k, v["status"], {i: x["value"] for i, x in v["items"].items()})
