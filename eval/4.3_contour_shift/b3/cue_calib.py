"""b3 校准汇总（草案 protocol_cue_draft_v0 §2.2 到 §2.5、§7；专家意见 astra_cue_design_raw 问题 1、3）。
  python cue_calib.py params --root out_replay --blocks B2:8 F1:4,8,12 --out cue_params.json --table cue_disp.json
  python cue_calib.py report --root out_replay --blocks B2 F1 --params cue_params.json --out cue_report.json
params：由第 1 遍回放算 q_open（参考大开度）与位移分布、a、b。
  帧范围两种并列：
    shown_prefix（主口径，专家问题 3）：全部查询帧，滞后源 = max(t-L, 0)（策略实际看到的那一帧），
      只取 t <= 首次成功的步（未成功回合取全程），两帧掩膜都非空；
    legacy_ge_L（草案 v0 原口径）：t >= L 的全部查询帧，两帧都非空。
  a = round(配准位移 d 的中位数)，b = round(第 90 百分位)，下限 1；同时给质心位移同分位数、残差 r、IoU。
report：第 2 遍回放的模板、偏移、重排统计与状态核对。
"""
import argparse
import glob
import json
import math
import os

import numpy as np

QS = (5, 25, 50, 75, 90, 95)


def qt(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {f"p{q}": round(float(np.percentile(v, q)), 4) for q in QS} | {"n": int(len(v)), "mean": round(float(v.mean()), 4)}


def load(root, b, pas):
    return [json.load(open(p)) for p in sorted(glob.glob(os.path.join(root, b, f"pass{pas}", "*.json")))]


def close_idx(ep, body):
    g = np.asarray(ep["grip_actions"])
    hit = np.nonzero(g < 0)[0] if body == "widowx" else np.nonzero(g > 0)[0]
    return int(hit[0]) if len(hit) else None


def body_of(b):
    return "widowx" if b.startswith("B") else "google"


def disp_rows(eps, L, scope):
    rows = []
    for ep in eps:
        fs = ep["first_success_idx"]
        sx, sy = 224.0 / ep["W"], 224.0 / ep["H"]
        for f in ep["frames"]:
            t = f["step"]
            if scope == "legacy_ge_L" and t < L:
                continue
            if scope == "shown_prefix" and fs is not None and t > fs:
                continue
            d = f["lag"][str(L)]
            g0, g1 = f["geo"], d["geo_src"]
            if g0 is None or g1 is None:
                continue
            r = {"task": ep["task"], "key": ep["key"], "t": t, "prog": t / ep["max_steps"]}
            r["d"] = math.hypot(d["reg_dx"], d["reg_dy"])
            r["reg_dx"], r["reg_dy"] = d["reg_dx"], d["reg_dy"]
            r["d224_h"] = abs(d["reg_dx"]) * sx
            r["d224"] = math.hypot(d["reg_dx"] * sx, d["reg_dy"] * sy)
            r["resid"] = 1 - d["reg_iou"]
            r["cdx"], r["cdy"] = g1["cx"] - g0["cx"], g1["cy"] - g0["cy"]
            r["cd"] = math.hypot(r["cdx"], r["cdy"])
            r["iou_outline"], r["iou_mask"] = d["iou_outline"], d["iou_mask"]
            r["d_closed"] = ep["frames"][d["src"] // ep["qe"]]["kin"]["closedness"] - f["kin"]["closedness"]
            r["px_ratio"] = g1["n"] / g0["n"]
            r["gw"] = g0["x1"] - g0["x0"]
            r["is_src0"] = t < L
            r["cd224"] = math.hypot(r["cdx"] * sx, r["cdy"] * sy)
            rows.append(r)
    return rows


def disp_summary(rows):
    s = {k: qt([r[k] for r in rows]) for k in ("d", "reg_dx", "reg_dy", "d224", "d224_h", "resid", "cd", "cdx", "cdy",
                                                 "iou_outline", "iou_mask", "d_closed", "px_ratio", "gw")}
    s["abs_d_closed"] = qt([abs(r["d_closed"]) for r in rows])
    s["n_frames"] = len(rows)
    s["n_frames_src0"] = sum(r["is_src0"] for r in rows)
    s["frac_open_change_ge_0.1"] = (sum(abs(r["d_closed"]) >= 0.1 for r in rows) / len(rows)) if rows else None
    # 低 IoU 不等于大平移：IoU 低（轮廓 IoU < 0.3）的帧里配准位移的分布
    low = [r for r in rows if r["iou_outline"] is not None and r["iou_outline"] < 0.3]
    s["low_iou_frames"] = {"n": len(low), "d": qt([r["d"] for r in low]), "resid": qt([r["resid"] for r in low])}
    # 配准位移很小（<= 4 像素）但形状变了（残差 >= 0.3 或 |开合变化| >= 0.1）
    sm = [r for r in rows if r["d"] <= 4]
    s["small_d_shape_changed"] = {"n_small_d": len(sm), "n_changed": sum(1 for r in sm if r["resid"] >= 0.3 or abs(r["d_closed"]) >= 0.1),
                                  "frac_of_all": (sum(1 for r in sm if r["resid"] >= 0.3 or abs(r["d_closed"]) >= 0.1) / len(rows)) if rows else None}
    # 质心几乎不动（224 上 <= 1 像素）但开合或配准残差变化大
    cs = [r for r in rows if r["cd224"] <= 1.0]
    s["centroid_still_shape_changed"] = {"n_still": len(cs), "n_changed": sum(1 for r in cs if r["resid"] >= 0.3 or abs(r["d_closed"]) >= 0.1)}
    if rows:
        a = max(1, round(float(np.median([r["d"] for r in rows]))))
        b = max(1, round(float(np.percentile([r["d"] for r in rows], 90))))
        gw = float(np.median([r["gw"] for r in rows]))
        s["a"], s["b"] = a, b
        s["a_b_input_px_h"] = [round(a * 224 / 640, 2), round(b * 224 / 640, 2)]
        s["gripper_width_median_px"] = gw
        s["a_b_over_gripper_width"] = [round(a / gw, 3), round(b / gw, 3)]
        s["a_eq_b"] = a == b
        s["cd_p50_p90"] = [round(float(np.median([r["cd"] for r in rows])), 2), round(float(np.percentile([r["cd"] for r in rows], 90)), 2)]
        tr = []
        for i in range(5):
            bb = [r for r in rows if i / 5 <= r["prog"] < (i + 1) / 5]
            tr.append({"bin": f"{i/5:.1f}-{(i+1)/5:.1f}", "n": len(bb), "med_d": float(np.median([r["d"] for r in bb])) if bb else None,
                       "med_iou_outline": float(np.median([r["iou_outline"] for r in bb])) if bb else None,
                       "med_resid": float(np.median([r["resid"] for r in bb])) if bb else None})
        s["trend"] = tr
    return s


def q_open(eps):
    """参考大开度（草案 §2.2，专家问题 1 的叫法）：全部控制步的指尖间距第 99 百分位那一步的手指关节向量；
    参考最小开度（反方必须改 2 的闭合模板）：指尖间距最小的那一步。"""
    st = [(s["tip_dist"], s["finger_q"], ep["key"], i) for ep in eps for i, s in enumerate(ep["steps"])]
    st.sort(key=lambda x: x[0])
    j = max(0, int(math.ceil(0.99 * len(st))) - 1)
    d, q, k, i = st[j]
    ep0 = eps[0]
    return {"q_open": q, "tip_dist": d, "from_key": k, "from_step": i, "n_steps_pool": len(st),
            "q_closed": st[0][1], "q_closed_tip_dist": st[0][0], "q_closed_from": [st[0][2], st[0][3]],
            "tip_dist_max": st[-1][0], "tip_dist_min": st[0][0],
            "mechanical_open_limit": ep0["finger_open_limit"], "mechanical_closed_limit": ep0["finger_closed_limit"],
            "q_open_minus_limit": (np.asarray(q) - np.asarray(ep0["finger_open_limit"])).tolist(),
            "q0_reset": ep0["finger_q0"]}


def cmd_params(a):
    params, table = {}, {}
    for spec in a.blocks:
        b, lags = spec.split(":")
        eps = load(a.root, b, 1)
        t = {"n_keys": len(eps), "keys": [e["key"] for e in eps], "replay_gpx_equal": sum(e["replay_gpx_equal"] for e in eps),
             "first_success_idx": [e["first_success_idx"] for e in eps], "q_open": q_open(eps), "lags": {}}
        shifts = {}
        for L in (int(x) for x in lags.split(",")):
            t["lags"][str(L)] = {}
            for scope in ("shown_prefix", "legacy_ge_L"):
                rows = disp_rows(eps, L, scope)
                s = disp_summary(rows)
                tasks = sorted({r["task"] for r in rows})
                if len(tasks) > 1:
                    s["by_task"] = {tk: {k: v for k, v in disp_summary([r for r in rows if r["task"] == tk]).items()
                                         if k in ("n_frames", "d", "iou_outline", "resid", "cd", "a", "b", "abs_d_closed")} for tk in tasks}
                t["lags"][str(L)][scope] = s
                if scope == "shown_prefix" and rows:
                    for nm, m in (("a", s["a"]), ("b", s["b"])):
                        shifts[f"S_L{L}_{nm}+"] = [m, 0]
                        shifts[f"S_L{L}_{nm}-"] = [-m, 0]
                    shifts[f"Sv_L{L}_b+"] = [0, s["b"]]
                    shifts[f"Sv_L{L}_b-"] = [0, -s["b"]]
        # 夹爪可见宽度比例定幅度（反方必须改 4、协调方第 3 条）：在策略输入尺度上定义，再换算回环境像素
        nzf = [f for ep in eps for f in ep["frames"] if f["geo"] is not None]
        W, H = eps[0]["W"], eps[0]["H"]
        w_in = float(np.median([(f["geo"]["x1"] - f["geo"]["x0"]) * 224.0 / W for f in nzf]))
        h_in = float(np.median([(f["geo"]["y1"] - f["geo"]["y0"]) * 224.0 / H for f in nzf]))
        wf = {"gripper_visible_width_input_px_median": round(w_in, 3), "gripper_visible_height_input_px_median": round(h_in, 3),
              "width_input_px": qt([(f["geo"]["x1"] - f["geo"]["x0"]) * 224.0 / W for f in nzf]), "fractions": {}}
        for frac, tag in ((0.25, "w14"), (0.5, "w12"), (1.0, "w1")):
            m_in = frac * w_in
            m_env = int(round(m_in * W / 224.0))
            e = {"input_px": round(m_in, 3), "env_px_horizontal": m_env, "env_px_if_vertical_axis": int(round(m_in * H / 224.0)),
                 "percentile_in_lag_dist": {}}
            for L, dd in t["lags"].items():
                for scope in ("shown_prefix", "legacy_ge_L"):
                    rows = disp_rows(eps, int(L), scope)
                    if rows:
                        d224 = np.array([r["d224"] for r in rows]); dh = np.array([r["d224_h"] for r in rows])
                        e["percentile_in_lag_dist"][f"L{L}_{scope}"] = {
                            "pct_rank_vs_d224": round(100.0 * float((d224 < m_in).mean() + 0.5 * (d224 == m_in).mean()), 1),
                            "pct_rank_vs_horizontal_d224": round(100.0 * float((dh < m_in).mean() + 0.5 * (dh == m_in).mean()), 1)}
            wf["fractions"][tag] = e
            shifts[f"W_{tag}+"] = [m_env, 0]
            shifts[f"W_{tag}-"] = [-m_env, 0]
        t["width_fraction_amplitudes"] = wf
        L0 = lags.split(",")[0]
        bb = shifts[f"S_L{L0}_b+"][0]
        copies = {f"SC_L{L0}_b+": [bb, 0], f"SC_L{L0}_b-": [-bb, 0],
                  "SC_w12+": [wf["fractions"]["w12"]["env_px_horizontal"], 0], "SC_w12-": [-wf["fractions"]["w12"]["env_px_horizontal"], 0]}
        table[b] = t
        params[b] = {"q_open": t["q_open"]["q_open"], "q_closed": t["q_open"]["q_closed"], "shifts": shifts, "copies": copies}
    json.dump(params, open(a.out, "w"), indent=1)
    json.dump(table, open(a.table, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(params, indent=1))


def tpl_dose(eps, nm, body):
    """模板剂量（反方必须改 2、协调方第 4 条）：指尖间距之差（模板减当前，米）与掩膜 IoU，
    按「首次闭合指令之前/之后」「首次成功之前（含）/之后」分段。线 IoU 只并列。"""
    seg = {}
    for ep in eps:
        c, fs = close_idx(ep, body), ep["first_success_idx"]
        for f in ep["frames"]:
            if f["G_px"] == 0:
                continue
            d = f[nm]
            k1 = "before_close" if (c is None or f["step"] <= c) else "after_close"
            k2 = "before_success" if (fs is None or f["step"] <= fs) else "after_success"
            for k in (k1, k2, f"{k1}&{k2}", "all"):
                seg.setdefault(k, []).append(d)
    out = {}
    for k, ds in seg.items():
        tip = [d["tpl_tip_dist"] - d["cur_tip_dist"] for d in ds]
        out[k] = {"n": len(ds), "tip_dist_diff_m": qt(tip), "abs_tip_dist_diff_m": qt([abs(x) for x in tip]),
                  "iou_tpl_mask": qt([d["iou_tpl_mask"] for d in ds]), "iou_tpl_drawn_line": qt([d["iou_tpl_drawn"] for d in ds]),
                  "n_mask_iou_le_0.7": sum(1 for d in ds if (d["iou_tpl_mask"] or 0) <= 0.7),
                  "n_abs_tip_diff_ge_1cm": sum(1 for x in tip if abs(x) >= 0.01)}
    return out


def cmd_report(a):
    out = {}
    prm = json.load(open(a.params))
    for b in a.blocks:
        eps = load(a.root, b, 2)
        body = body_of(b)
        r = {"n_keys": len(eps), "state_sha_equal_pass1": sum(e["state_sha_equal_pass1"] for e in eps),
             "steps_compared": sum(len(e["state_sha"]) for e in eps), "tpl_renders": sum(e["tpl_renders"] for e in eps),
             "tpl_restore_all_equal": sum(e["tpl_restore_all_equal"] for e in eps),
             "tpl_rerender_all_equal": sum(e["tpl_rerender_all_equal"] for e in eps),
             "replay_gpx_equal": sum(e["replay_gpx_equal"] for e in eps)}
        F = [(ep, f) for ep in eps for f in ep["frames"]]
        nz = [(ep, f) for ep, f in F if f["G_px"] > 0]
        r["n_frames"], r["n_frames_G_nonempty"] = len(F), len(nz)
        r["edge_touch_frac"] = {e: sum(f["edge"][e] for _, f in nz) / len(nz) for e in ("left", "right", "top", "bottom")}
        r["C4"] = {"O_px": qt([f["C4"]["O_px"] for _, f in nz]), "frac_protected": qt([f["C4"]["frac_protected"] for _, f in nz]),
                   "line_px_224": qt([f["C4"]["line_px_224"] for _, f in nz]),
                   "line_gone_frac": sum(f["C4"]["line_px_224"] == 0 for _, f in nz) / len(nz),
                   "iou_prev": qt([f["C4"]["iou_prev"] for _, f in nz])}
        # 模板：闭合指令前后、首次成功前后
        T = {"q_open": prm[b]["q_open"]}
        segs = {"before_close": [], "after_close": []}
        per_ep_diff = []
        diff_before_success = diff_total = 0
        for ep in eps:
            c = close_idx(ep, body)
            fs = ep["first_success_idx"]
            nd = 0
            for f in ep["frames"]:
                if f["G_px"] == 0:
                    continue
                tt = f["T"]
                seg = "before_close" if (c is None or f["step"] <= c) else "after_close"
                segs[seg].append(tt)
                if (tt["iou_tpl_drawn"] or 0) <= 0.7:
                    nd += 1
                    diff_total += 1
                    if fs is None or f["step"] <= fs:
                        diff_before_success += 1
            per_ep_diff.append(nd)
        for sname, ds in segs.items():
            T[sname] = {"n": len(ds), "iou_tpl_drawn": qt([d["iou_tpl_drawn"] for d in ds]),
                        "iou_tpl_outline": qt([d["iou_tpl_outline"] for d in ds]), "iou_tpl_mask": qt([d["iou_tpl_mask"] for d in ds]),
                        "n_diff_iou_le_0.7": sum(1 for d in ds if (d["iou_tpl_drawn"] or 0) <= 0.7)}
        T["per_episode_diff_frames"] = per_ep_diff
        T["per_episode_diff_median"] = float(np.median(per_ep_diff)) if per_ep_diff else None
        T["diff_frames_total"], T["diff_frames_at_or_before_first_success"] = diff_total, diff_before_success
        T["close_idx"] = [close_idx(ep, body) for ep in eps]
        T["first_success_idx"] = [ep["first_success_idx"] for ep in eps]
        T["tpl_q_applied_minus_q_open_maxabs"] = float(max(np.max(np.abs(np.subtract(f["T"]["tpl_q_applied"], prm[b]["q_open"])))
                                                         for _, f in nz))
        all_T = [f["T"] for _, f in nz]
        T["stroke_ratio_vs_C4"] = qt([d["O_drawn_px"] / f["C4"]["D_px"] for (_, f), d in zip(nz, all_T) if f["C4"]["D_px"]])
        T["frac_protected"] = qt([d["O_protected_px"] / d["O_px"] for d in all_T if d["O_px"]])
        T["line_gone_frac"] = sum(d["line_px_224"] == 0 for d in all_T) / len(all_T)
        r["T"] = T
        r["T_closed"] = tpl_dose(eps, "T_closed", body)
        r["T_dose"] = tpl_dose(eps, "T", body)
        # 偏移
        S = {}
        for nm in list(prm[b]["shifts"]) + list(prm[b].get("copies", {})):
            ds = [(f, f[nm]) for _, f in nz]
            v = [d["frac_inview"] for _, d in ds]
            S[nm] = {"dxdy": prm[b]["shifts"].get(nm, prm[b].get("copies", {}).get(nm)), "frac_inview": qt(v),
                     "frac_frames_inview_lt_0.8": sum(1 for x in v if x is not None and x < 0.8) / len(v),
                     "frac_protected": qt([d["frac_protected"] for _, d in ds]), "frac_on_arm": qt([d["frac_on_arm"] for _, d in ds]),
                     "iou224_shiftQ_vs_Q": qt([d.get("iou224_shiftQ_vs_Q") for _, d in ds]),
                     "copy_overlap_with_cur": qt([d.get("copy_overlap_with_cur") for _, d in ds]),
                     "stroke_ratio_vs_C4": qt([d["O_drawn_px"] / f["C4"]["D_px"] for f, d in ds if f["C4"]["D_px"]]),
                     "line_px_224": qt([d["line_px_224"] for _, d in ds]),
                     "line_gone_frac": sum(d["line_px_224"] == 0 for _, d in ds) / len(ds)}
        r["S"] = S
        X = {}
        for nm in ("X_query", "X_episode"):
            ds = [(f, f[nm]) for _, f in nz]
            inf = [d["scr_info"] for _, d in ds]
            X[nm] = {"s_used": {str(s): sum(1 for i in inf if i.get("s") == s) for s in sorted({i.get("s") for i in inf})},
                     "n_nonempty_tiles": qt([i.get("n_nonempty") for i in inf]), "n_tiles": qt([i.get("n_tiles") for i in inf]),
                     "degenerate_frac": sum(1 for i in inf if i.get("degenerate")) / len(inf),
                     "stroke_ratio_vs_C4": qt([d["scr_px_ratio"] for _, d in ds]),
                     "frac_stroke_ratio_in_0.85_1.15": sum(1 for _, d in ds if d["scr_px_ratio"] is not None and 0.85 <= d["scr_px_ratio"] <= 1.15) / len(ds),
                     "bbox_area_ratio": qt([d["scr_bbox_area_ratio"] for _, d in ds]),
                     "iou_vs_C4": qt([d["iou_scr_vs_cur"] for _, d in ds]),
                     "outside_grid_frac": qt([d["scr_outside_grid_frac"] for _, d in ds]),
                     "frac_protected": qt([d["O_protected_px"] / d["O_px"] for _, d in ds if d["O_px"]]),
                     "iou_prev": qt([d["iou_prev"] for _, d in ds]),
                     "line_gone_frac": sum(d["line_px_224"] == 0 for _, d in ds) / len(ds)}
        r["X"] = X
        out[b] = r
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print("written", a.out)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["params", "report"])
    ap.add_argument("--root", required=True)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--table", default="cue_disp.json")
    ap.add_argument("--params", default="")
    a = ap.parse_args()
    cmd_params(a) if a.cmd == "params" else cmd_report(a)
