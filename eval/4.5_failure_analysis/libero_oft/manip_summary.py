"""Summarize manip_maskinfo.py snapshot-mode jsonl into the protocol 2.4 items 1-10 (median / p90, per suite and view)
and the two pre-registered adjustment rules (lag 16 -> 24 if agentview IoU median > 0.7; width 2 -> 3 if the 224
edge of C4 / C5 is < 0.8 x C1). Usage: python manip_summary.py FILE.jsonl [...] [--json OUT]"""
import json, sys
from collections import Counter, defaultdict
import numpy as np

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
    d["item4_occ"] = {c: q([r["item4_occ"].get(c) for r in rr]) for c in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    d["item4_occP"] = {c: q([r.get("item4_occP", {}).get(c) for r in rr]) for c in ("C0", "C1", "C2", "C3", "C4", "C5", "C6", "C7")}
    d["item4_occP_C6_eq_C1_frames"] = sum(1 for r in rr if r.get("item4_occP", {}).get("C6") == r.get("item4_occP", {}).get("C1"))
    d["item4_occP_C345_nonzero"] = sum(1 for r in rr for c in ("C3", "C4", "C5") if r.get("item4_occP", {}).get(c) not in (None, 0.0))
    d["item5_trunc"] = {c: q([r["item5"].get(c, {}).get("frac") for r in rr]) for c in ("C4", "C5")}
    d["item5_dG_adj_Pvis"] = q([r["item5"].get("dG_adj_frac") for r in rr])
    e = {c: [r["item6_edge224"].get(c) for r in rr] for c in ("C1", "C4", "C5")}
    d["item6_edge224"] = {c: q(x) for c, x in e.items()}
    m1 = d["item6_edge224"]["C1"][0] if d["item6_edge224"]["C1"] else None
    d["item6_ratio_to_C1"] = {c: (None if not m1 or not d["item6_edge224"][c] else round(d["item6_edge224"][c][0] / m1, 3))
                              for c in ("C4", "C5")}
    for L in (16, 24):
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
agent_iou = [r["item7"].get("iou_16") for (s, v), rr in R.items() if v == "agent" for r in rr]
K["rule_lag"] = {"agent_iou16_median_all": q(agent_iou),
                 "by_suite": {s: S[f"{s}/agent"]["item7_iou_16"] for s in ("spatial", "object") if f"{s}/agent" in S},
                 "decision": None if not q(agent_iou) else (24 if q(agent_iou)[0] > 0.7 else 16)}
K["rule_width"] = {k: v["item6_ratio_to_C1"] for k, v in S.items()}
K["rule_width"]["decision"] = 3 if any((x or 1) < 0.8 for v in S.values() for x in v["item6_ratio_to_C1"].values()) else 2
print(json.dumps(K, indent=1, default=str))
for k, v in S.items():
    print("==", k)
    print(json.dumps(v, default=str))
if out_json:
    json.dump({"keys": K, "per_suite_view": S}, open(out_json, "w"), indent=1, default=str)
