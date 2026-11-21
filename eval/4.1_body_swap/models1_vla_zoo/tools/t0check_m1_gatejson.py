"""Build the --gate-json for prereg_models1.py (toy1 format: cells.<dir>.status / items) from t0check_m1.py output,
plus the embody_toy regression result. 2026-11-04. Model-independent (BodyEnv only), so the same entry serves all models.
Usage: python t0check_m1_gatejson.py --spatial t0_spatial.jsonl --object t0_object.jsonl --regr-ok 1 --out gate_m1.json
"""
import argparse, json
import numpy as np
A, W = "agentview_image", "robot0_eye_in_hand_image"
ap = argparse.ArgumentParser()
ap.add_argument("--spatial", required=True)
ap.add_argument("--object", required=True)
ap.add_argument("--regr-ok", type=int, required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
cells = {}
for pre, f in (("", a.spatial), ("obj_", a.object)):
    R = [json.loads(l) for l in open(f) if l.strip()]
    reg = {"status": "PASS" if a.regr_ok else "FAIL", "value": "embody_toy vs embody_v3 replay qpos diff 0 on (0,0),(2,0) S and G"}
    m_a = float(np.mean([r["t0_vs_s"][A]["mean"] for r in R])); m_w = float(np.mean([r["t0_vs_s"][W]["mean"] for r in R]))
    cells[pre + "sham_twin"] = {"status": "PASS" if m_a < 0.11 and m_w < 0.11 else "FAIL", "items": {
        "t0_vs_s_agent": {"status": "PASS" if m_a < 0.11 else "FAIL", "value": round(m_a, 4)},
        "t0_vs_s_wrist": {"status": "PASS" if m_w < 0.11 else "FAIL", "value": round(m_w, 4)}, "regression": reg,
        "keys": {"status": "NA", "value": len(R)}}}
    for n, same, changed in (("g_tfc", None, (A, W)), ("g_tw_fc", A, (W,)), ("g_ta_fc", W, (A,)), ("g_t", None, (A, W))):
        it = {"keys": {"status": "NA", "value": len(R)}, "regression": reg}
        if same:
            # identical up to the known 1-grey-level EGL render noise (toy1 5.7 precedent: mean < 0.11, max <= 1)
            mx = max(r[f"{n}_vs_g"][same]["max"] for r in R)
            mm = float(np.mean([r[f"{n}_vs_g"][same]["mean"] for r in R]))
            it[f"unreplaced_{same[:5]}"] = {"status": "PASS" if (mx <= 1 and mm < 0.11) else "FAIL", "value": f"max {mx} mean {mm:.2e}"}
        for c in changed:
            m = float(np.mean([r[f"{n}_vs_g_both_hidden"][c]["mean"] for r in R])); lim = 1.0 if c == A else 8.0
            it[f"replaced_{c[:5]}_both_hidden_mean"] = {"status": "PASS" if m < lim else "FAIL", "value": round(m, 4)}
        st = "FAIL" if any(v["status"] == "FAIL" for v in it.values()) else "PASS"
        cells[pre + n] = {"status": st, "items": it}
json.dump({"source": [a.spatial, a.object], "cells": cells}, open(a.out, "w"), indent=1)
for k, v in cells.items():
    print(k, v["status"], {i: x["value"] for i, x in v["items"].items() if i != "regression"})
