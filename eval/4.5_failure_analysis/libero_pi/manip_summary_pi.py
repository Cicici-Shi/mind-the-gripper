"""maskinfo-pi copy of $PROJ_ROOT/maskinfo/tools/manip_summary.py for tools/manip_maskinfo_pi.py output: cells C0..C8,
item 6 per preprocessing (reported only; protocol v2 fixes the width, no width rule), item 7 at lag 15 / 20 with the
decision rule (agentview IoU median > 0.7 in EITHER suite -> 20), item 9 three md5 + same-moment state, item 15
visibility checks (hard: (i) input == expected, (ii) no vanished protected outline, expected composite == obs) and
reported quantities. Original docstring: Summarize manip_maskinfo.py snapshot-mode jsonl into the protocol 2.4 items 1-10 (median / p90, per suite and view)
and the two pre-registered adjustment rules (lag 16 -> 24 if agentview IoU median > 0.7; width 2 -> 3 if the 224
edge of C4 / C5 is < 0.8 x C1). Usage: python manip_summary.py FILE.jsonl [...] [--json OUT]"""
import json, sys
from collections import Counter, defaultdict
import numpy as np
LAGS = (15, 20)

files = [a for a in sys.argv[1:] if not a.startswith("--")]
out_json = sys.argv[sys.argv.index("--json") + 1] if "--json" in sys.argv else None
if out_json in files:
    files.remove(out_json)
rows, keys, hdr = [], [], []
for f in files:
    for l in open(f):
        r = json.loads(l)
        if r.get("t0_mode"):
            continue
        (hdr if r.get("header") else keys if r.get("key_row") else rows).append(r)
suite_of = {}
for h, f in zip(hdr, files):
    pass
for f in files:
    s = None
    for l in open(f):
        r = json.loads(l)
        if r.get("header"):
            s = r["suite"]
        elif "task" in r:
            suite_of[(f, r["task"], r["ep"])] = s
R = defaultdict(list)
for f in files:
    s = None
    for l in open(f):
        r = json.loads(l)
        if r.get("header"):
            s = r["suite"]
        elif "view" in r and not r.get("t0_mode"):
            R[(s, r["view"])].append(r)


def q(a):
    a = [x for x in a if x is not None]
    return None if not a else [round(float(np.median(a)), 4), round(float(np.percentile(a, 90)), 4), len(a)]


S = {}
for (s, v), rr in sorted(R.items()):
    d = {}
    bad1 = Counter()
    for r in rr:
        for c, x in r["item1"].items():
            if isinstance(x, dict) and x.get("ok") is False:
                bad1[c] += 1
            if isinstance(x, dict) and x.get("drawn_all_black") is False:
                bad1["drawn_not_black_" + c] += 1
    d["item1_frames_violating"] = dict(bad1)
    d["item1_cat_C1_outside_G"] = sum(r["item1_categories"]["C1"]["n"] - r["item1_categories"]["C1"]["in_G"] for r in rr)
    d["item2_B_eq_M7"] = Counter(str(r["item2"].get("B_eq_M7_plus_trunc")) for r in rr)
    mb = [r["item2"].get("M7_over_B") for r in rr if r["item2"].get("M7_over_B") is not None]
    d["item2_M7_over_B_median_min"] = None if not mb else [round(float(np.median(mb)), 4), round(float(min(mb)), 4), len(mb)]
    d["item2_M7_ge_08B_false"] = sum(1 for r in rr if r["item2"].get("M7_ge_08B") is False)
    d["item3_border_bad"] = sum(1 for r in rr for c in ("C4_border", "C5_border") if r["item3"].get(c, {}).get("ok") is False)
    d["item4_d_occP_C7"] = q([r.get("item4_d_occP_C7") for r in rr])
    d["item2_ratio_C5_C4"] = q([r["item2"].get("ratio_C5_C4") for r in rr])
    d["item3_outside_dilate8_frames"] = {c: sum(1 for r in rr if (r["item3"].get(c, {}).get("outside_dilate8_px") or 0) > 0)
                                         for c in ("C4", "C5")}
    CC = ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8")
    d["item4_occ"] = {c: q([r["item4_occ"].get(c) for r in rr]) for c in CC}
    d["item4_occP"] = {c: q([r.get("item4_occP", {}).get(c) for r in rr]) for c in CC}
    d["item4_occP_C6_eq_C1_frames"] = sum(1 for r in rr if r.get("item4_occP", {}).get("C6") == r.get("item4_occP", {}).get("C1"))
    d["item4_occP_C345_nonzero"] = sum(1 for r in rr for c in ("C3", "C4", "C5") if r.get("item4_occP", {}).get(c) not in (None, 0.0))
    d["item5_trunc"] = {c: q([r["item5"].get(c, {}).get("frac") for r in rr]) for c in ("C4", "C5")}
    d["item5_dG_adj_Pvis"] = q([r["item5"].get("dG_adj_frac") for r in rr])
    d["item6_edge224"], d["item6_ratio_to_C1"] = {}, {}
    for prep in ("openpi", "oft"):
        e = {c: [r["item6_edge224"][prep].get(c) for r in rr] for c in ("C1", "C4", "C5")}
        d["item6_edge224"][prep] = {c: q(x) for c, x in e.items()}
        m1 = d["item6_edge224"][prep]["C1"][0] if d["item6_edge224"][prep]["C1"] else None
        d["item6_ratio_to_C1"][prep] = {c: (None if not m1 or not d["item6_edge224"][prep][c] else
                                            round(d["item6_edge224"][prep][c][0] / m1, 3)) for c in ("C4", "C5")}
    i15 = [r["item15"] for r in rr]
    d["item15_i_false"] = sum(1 for x in i15 for ok in x["i_input_eq_expected"].values() if not ok)
    for c in ("C4", "C5"):
        xs = [x[c] for x in i15 if c in x]
        d[f"item15_{c}"] = {"frames": len(xs), "errors": sum(1 for x in xs if "error" in x),
                            "expected256_eq_obs_false": sum(1 for x in xs if x.get("expected256_eq_obs") is False),
                            "expected224_eq_input_false": sum(1 for x in xs if x.get("expected224_eq_input") is False),
                            "lag_not_t_minus_lag": sum(1 for x in xs if x.get("lag_ok") is False),
                            "own_Ibg_ne_C3": sum(1 for x in xs if x.get("own_Ibg_eq_C3") is False),
                            "trunc_protected": q([x.get("trunc_protected") for x in xs])}
        for prep in ("openpi", "oft"):
            d[f"item15_{c}"][prep] = {"vanished": sum(1 for x in xs if x.get(prep, {}).get("vanished")),
                                      "proj_empty_at_threshold": sum(1 for x in xs if x.get(prep, {}).get("proj_empty")),
                                      "visible_px": q([x.get(prep, {}).get("visible_px") for x in xs]),
                                      "contrast": q([x.get(prep, {}).get("contrast") for x in xs]),
                                      "in_input_frac": q([x.get(prep, {}).get("in_input_frac") for x in xs])}
    d["item15_effE"] = {c: {p: q([x["effE"][c][p] for x in i15 if c in x["effE"]]) for p in ("openpi", "oft")}
                        for c in CC if c != "C0"}
    d["item15_Pvis_retained"] = {p: q([x["Pvis_retained"][p] for x in i15]) for p in ("openpi", "oft")}
    wb = [x["wrist_bottom_G"] for x in i15 if x.get("wrist_bottom_G")]
    d["item15_wrist_bottom_G"] = None if not wb else {k: q([w[k] for w in wb]) for k in wb[0]}
    d["item1_C8_mask_ne_C1"] = sum(1 for r in rr if r["item1"].get("C8_mask_eq_C1", {}).get("ok") is False)
    d["item1_C8_mask_checked"] = sum(1 for r in rr if "C8_mask_eq_C1" in r["item1"])
    d["item9_S_false"] = sum(1 for r in rr for ok in r.get("item9_S", {}).values() if not ok)
    for L in LAGS:
        d[f"item7_iou_{L}"] = q([r["item7"].get(f"iou_{L}") for r in rr])
        o = [r["item7"][f"open_differs_{L}"] for r in rr if f"open_differs_{L}" in r["item7"]]
        d[f"item7_open_changed_frac_{L}"] = None if not o else [round(float(np.mean(o)), 4), len(o)]
        vv = [r["item7"][f"vis_changed_{L}"] for r in rr if f"vis_changed_{L}" in r["item7"]]
        d[f"item7_vis_changed_frac_{L}"] = None if not vv else [round(float(np.mean(vv)), 4), len(vv)]
    d["item8_overlap"] = q([r.get("item8", {}).get("overlap") for r in rr])
    d["item8_cand"] = dict(Counter(r.get("item8", {}).get("cand") for r in rr))
    d["item8_trunc_px"] = q([r.get("item8", {}).get("trunc_px") for r in rr])
    d["item9_false"] = sum(1 for r in rr for c, ok in r["item9"].items() if not ok)
    d["self_bad"] = sum(1 for r in rr for c, x in r["self"].items()
                        if x.get("out_E_diff_px") or x.get("same_substep") is False or x.get("final_eq_edited") is False)
    det = [x for r in rr for x in (r.get("det") or {}).values() if x]
    d["det_frames"] = len(det)
    d["det_seg_bad"] = sum(1 for x in det if x["seg_twice_px"] or x["bgseg_twice_px"])
    d["det_rgb_nonzero"] = sum(1 for x in det if x["rgb_twice"][0] or x["bg_twice"][0])
    d["det_rgb_bad"] = sum(1 for x in det if not ((x["rgb_twice"][0] == 0 or (x["rgb_twice"][0] <= 10 and x["rgb_twice"][1] <= 1))
                                                  and (x["bg_twice"][0] == 0 or (x["bg_twice"][0] <= 10 and x["bg_twice"][1] <= 1))))
    d["n_frames"] = len(rr)
    S[f"{s}/{v}"] = d
PPX = {}
for k in keys:
    for v, dd in k.get("P_px_by_name", {}).items():
        for n, x in dd.items():
            PPX.setdefault(v, {}).setdefault(n, 0)
            PPX[v][n] += x
K = {"P_px_by_name_all_keys": PPX, "n_keys": len(keys), "gpus": sorted({k["gpu"] for k in keys}),
     "physics_max": max(max((x or 0) for x in k["physics_qpos_max_vs_C0"].values()) for k in keys) if keys else None,
     "scene_qpos0_max": max((k["scene_qpos0_maxdiff"] or 0) for k in keys) if keys else None,
     "frame0_restored_vs_real_nonzero": sum(1 for k in keys for v in k["frame0_restored_vs_real"].values() if v[0]),
     "replay_vs_dump_img": Counter((x[1][0] == 0 and x[2][0] == 0) for k in keys for x in k.get("replay_vs_dump_img", [])),
     "sheets": [k["sheet"][0] for k in keys if "sheet" in k]}
bys = {s: S[f"{s}/agent"]["item7_iou_15"] for s in ("spatial", "object") if f"{s}/agent" in S}
K["rule_lag"] = {"by_suite_agent_iou15_median_p90_n": bys,
                 "decision": None if not bys or any(v is None for v in bys.values()) else
                 (20 if any(v[0] > 0.7 for v in bys.values()) else 15),
                 "rule": "protocol v2 2.3: any suite agentview median > 0.7 -> all four blocks 20"}
K["visibility_hard_ok"] = all(v["item15_i_false"] == 0 and all(
    v[f"item15_{c}"]["errors"] == 0 and v[f"item15_{c}"]["expected256_eq_obs_false"] == 0
    and v[f"item15_{c}"]["expected224_eq_input_false"] == 0 and v[f"item15_{c}"]["openpi"]["vanished"] == 0
    and v[f"item15_{c}"]["lag_not_t_minus_lag"] == 0
    for c in ("C4", "C5")) and v["item1_C8_mask_ne_C1"] == 0 and v["item1_C8_mask_checked"] > 0 for v in S.values())
K["edge_ratio_reported_only"] = {k: v["item6_ratio_to_C1"] for k, v in S.items()}
print(json.dumps(K, indent=1, default=str))
for k, v in S.items():
    print("==", k)
    print(json.dumps(v, default=str))
if out_json:
    json.dump({"keys": K, "per_suite_view": S}, open(out_json, "w"), indent=1, default=str)
