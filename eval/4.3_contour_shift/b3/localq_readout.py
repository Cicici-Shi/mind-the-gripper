"""局部响应检验的读数（协议 protocol_cue_local.md v0 §2、§3、§4；未冻结，参数都可改）。
  python localq_readout.py --root out_localq --block B1 --keys keysets/formal_B1.json [--src ../b2/out_formal] --out readout_B1.json
输入：localq.py 的 <root>/<块>/<键>.npz（核对标志、簇）与 <键>.json（逐查询、逐画面的 r_in 等）。
定义（输入像素，水平轴 = 输入图 x，竖直轴 = 输入图 y）：
  r(画面) = 末端参考点加上该画面指令平移和之后的投影，减去末端参考点的投影；R(画面) = r(画面) - r(零平移)。
  D_a = [R_h(h+a) - R_h(h-a)] / 2a，D_b 同理；β = 四档水平平移上 R_h 对 s 过原点最小二乘斜率（四档都有效的状态才算）。
  竖直档：D_vb_v = [R_v(v+b) - R_v(v-b)] / 2b（同轴），D_vb_h（交叉）；水平 b 档的交叉分量 D_b_v。
  副本：D_copy_b = [R_h(copy+b) - R_h(copy-b)] / 2b。剂量：M = mean|R_h(±b)| - mean|R_h(±a)|（同状态配对）。
  有效：画面的视野内比例 >= --inview-min（默认 0.8）；零平移恒有效（G 为空的状态整体剔除）。
  主分析只用「第一次闭合指令之前（含当步）」的查询（--queries preclose）。
  汇总：先在键内对状态取平均，再按 episode（cfg_id）聚类 bootstrap，统计量为重采样到的全部键的平均（按键加权）。
  某档有效键数 < --min-keys（默认 60）则该档不判，涉及它的估计量不判。
  记号：为负 = 上界 < 0；为正 = 下界 > 0；近零 = 区间落在 ±near（默认 0.05）之内。
"""
import argparse
import glob
import json
import os

import numpy as np


def load(root, block, keys):
    recs, missing, failed = [], [], {"check1": [], "check2": [], "error_only": []}
    for k in keys:
        p = os.path.join(root, block, k["key"] + ".npz")
        if not os.path.exists(p):
            if os.path.exists(os.path.join(root, block, "_errors", k["key"] + ".txt")):
                failed["error_only"].append(k["key"])
            else:
                missing.append(k["key"])
            continue
        z = np.load(p)
        if not (bool(z["check1_available"]) and bool(z["check1_success_steps_equal"])):
            failed["check1"].append(k["key"])
            continue
        if not bool(z["check2_zero_bit_equal_all"]):
            failed["check2"].append(k["key"])
            continue
        meta = json.load(open(os.path.join(root, block, k["key"] + ".json")))
        recs.append({"key": k["key"], "cluster": str(z["cfg_id"]), "meta": meta})
    return recs, missing, failed


def state_values(m, a, b, inmin):
    V = m["views"]
    z = V["zero"]
    if m["G_px"] == 0:
        return None

    def ok(n):
        if n not in V:
            return False
        f = V[n]["frac_inview"]
        return f is not None and f >= inmin

    def R(n, ax):
        return V[n]["r_in"][ax] - z["r_in"][ax]

    out = {"valid": {n: ok(n) for n in V if n != "zero"}}
    if ok("h+a") and ok("h-a"):
        out["D_a"] = (R("h+a", 0) - R("h-a", 0)) / (2 * a)
    if ok("h+b") and ok("h-b"):
        out["D_b"] = (R("h+b", 0) - R("h-b", 0)) / (2 * b)
        out["D_b_v"] = (R("h+b", 1) - R("h-b", 1)) / (2 * b)
    if all(ok(n) for n in ("h-b", "h-a", "h+a", "h+b")):
        ss = np.array([-b, -a, a, b], float)
        rr = np.array([R("h-b", 0), R("h-a", 0), R("h+a", 0), R("h+b", 0)])
        out["beta"] = float((ss * rr).sum() / (ss * ss).sum())
        out["M"] = (abs(R("h+b", 0)) + abs(R("h-b", 0))) / 2 - (abs(R("h+a", 0)) + abs(R("h-a", 0))) / 2
    if ok("v+b") and ok("v-b"):
        out["D_vb_v"] = (R("v+b", 1) - R("v-b", 1)) / (2 * b)
        out["D_vb_h"] = (R("v+b", 0) - R("v-b", 0)) / (2 * b)
    if ok("copy+b") and ok("copy-b"):
        out["D_copy_b"] = (R("copy+b", 0) - R("copy-b", 0)) / (2 * b)
    out["abs_r0_h"] = abs(z["r_in"][0])
    out["abs_r0"] = float(np.hypot(*z["r_in"]))
    # 夹爪那一维（后处理后，WidowX 为 ±1）的平均，随平移的变化：+b 减 -b
    if ok("h+b") and ok("h-b"):
        out["grip_b_diff"] = float(np.mean(V["h+b"]["gripper"]) - np.mean(V["h-b"]["gripper"]))
    return out


def boot(keyvals, clusters, B, seed):
    """keyvals：每键一个数；clusters：同长的簇名。返回 (点估计, 下界, 上界, 键数, 簇数)。"""
    v = np.asarray(keyvals, float)
    cl = np.asarray(clusters)
    if len(v) == 0:
        return None
    names = sorted(set(cl.tolist()))
    idx = {c: np.nonzero(cl == c)[0] for c in names}
    rng = np.random.default_rng(seed)
    stats = np.empty(B)
    for i in range(B):
        pick = rng.integers(0, len(names), len(names))
        ii = np.concatenate([idx[names[j]] for j in pick])
        stats[i] = v[ii].mean()
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return {"est": float(v.mean()), "lo": float(lo), "hi": float(hi), "n_keys": int(len(v)), "n_clusters": len(names)}


def sign_of(ci, near):
    if ci is None:
        return "不判"
    if ci["hi"] < 0:
        return "负"
    if ci["lo"] > 0:
        return "正"
    if ci["lo"] >= -near and ci["hi"] <= near:
        return "近零"
    return "不定"


def run(a):
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    recs, missing, failed = load(a.root, a.block, keys)
    if missing:
        raise SystemExit(f"拒读：{len(missing)} 个键既无记录也无报错文件（采集未完成）：{missing[:5]}")
    names = ["D_a", "D_b", "beta", "M", "D_b_v", "D_vb_v", "D_vb_h", "D_copy_b", "abs_r0_h", "abs_r0", "grip_b_diff"]
    per_key, n_states, n_excl = {n: [] for n in names}, 0, {}
    level_keys = {}
    strat = {}
    for r in recs:
        qs = [m for m in r["meta"] if (m["pre_close"] or a.queries == "all") and len(m["views"]) > 1]
        vals = [state_values(m, a.a, a.b, a.inview_min) for m in qs]
        vals = [v for v in vals if v is not None]
        n_states += len(vals)
        for v in vals:
            for n, okv in v["valid"].items():
                if not okv:
                    n_excl[n] = n_excl.get(n, 0) + 1
        for n in names:
            xs = [v[n] for v in vals if n in v]
            if xs:
                per_key[n].append((r["key"], r["cluster"], float(np.mean(xs))))
        for lv in ("h-b", "h-a", "h+a", "h+b", "v-b", "v+b", "copy-b", "copy+b"):
            if any(v["valid"].get(lv) for v in vals):
                level_keys[lv] = level_keys.get(lv, 0) + 1
        # 按离第一次闭合还有几次查询分层（描述）
        for m, v in zip([m for m in qs if m["G_px"] > 0], vals):
            if m["first_close_step"] is not None and "D_b" in v:
                d = (m["first_close_step"] - m["step"]) // max(1, int(a.act_steps))
                strat.setdefault(str(min(d, 5)), []).append(v["D_b"])
    level_ok = {lv: level_keys.get(lv, 0) >= a.min_keys for lv in ("h-b", "h-a", "h+a", "h+b", "v-b", "v+b", "copy-b", "copy+b")}
    need = {"D_a": ["h-a", "h+a"], "D_b": ["h-b", "h+b"], "beta": ["h-b", "h-a", "h+a", "h+b"], "M": ["h-b", "h-a", "h+a", "h+b"],
            "D_b_v": ["h-b", "h+b"], "D_vb_v": ["v-b", "v+b"], "D_vb_h": ["v-b", "v+b"], "D_copy_b": ["copy-b", "copy+b"]}
    est = {}
    for n in names:
        if n in need and not all(level_ok[l] for l in need[n]):
            est[n] = None
            continue
        kv = per_key[n]
        est[n] = boot([x[2] for x in kv], [x[1] for x in kv], a.B, a.seed) if kv else None
    sg = {n: sign_of(est.get(n), a.near) for n in ("D_a", "D_b", "beta", "D_vb_v", "D_vb_h", "D_b_v", "D_copy_b")}
    m_pos = est.get("M") is not None and est["M"]["lo"] > 0
    if any(est.get(n) is None for n in ("D_a", "D_b", "beta")):
        outcome = "判不出（档不足）"
    elif sg["D_a"] == "负" and sg["D_b"] == "负" and sg["beta"] == "负" and m_pos:
        outcome = "反向且有剂量"
    elif sg["D_a"] == "负" and sg["D_b"] == "负":
        outcome = "反向但无剂量"
    elif all(sg[n] == "近零" for n in ("D_a", "D_b", "beta")):
        outcome = "近零"
    elif any(sg[n] in ("负", "正") for n in ("D_a", "D_b", "beta")):
        outcome = "同向或非单调"
    else:
        outcome = "判不出"
    # 方向特异性：竖直平移的同轴分量显著且大于交叉分量，且水平 b 档同轴分量大于交叉分量
    dirspec = None
    if est.get("D_vb_v") and est.get("D_vb_h") and est.get("D_b") and est.get("D_b_v"):
        follow = (sg["D_vb_v"] in ("负", "正") and abs(est["D_vb_v"]["est"]) > abs(est["D_vb_h"]["est"])
                  and abs(est["D_b"]["est"]) > abs(est["D_b_v"]["est"]))
        dirspec = "响应跟着平移的方向走" if follow else "存在交叉分量，句子限定为「在水平方向上」"
    copy_read = None
    if est.get("D_copy_b") is not None and est.get("D_b") is not None:
        if sg["D_copy_b"] == "近零" and sg["D_b"] == "负":
            copy_read = "响应来自正确位置上的线被移走，而不是旁边多了一条线"
        elif sg["D_copy_b"] == "负":
            copy_read = "旁边多一条线就足以引起同方向的响应（主句改为「在偏移位置出现轮廓线」）"
        else:
            copy_read = "副本格两条读法都不成立，照报数字"
    # 按 C4 最终成败分层（描述；只在 --src 给出且记录带成败时）
    by_outcome = None
    if a.src:
        by_outcome = {}
        for kname, cl, val in per_key["D_b"]:
            p = os.path.join(a.src, a.block, "C4", kname + ".npz")
            if os.path.exists(p):
                z = np.load(p)
                if "success" in z.files:
                    by_outcome.setdefault("success" if bool(z["success"]) else "fail", []).append(val)
        by_outcome = {k: {"n_keys": len(v), "mean_D_b": float(np.mean(v))} for k, v in by_outcome.items()}
    res = {"block": a.block, "params": {k: getattr(a, k) for k in ("a", "b", "inview_min", "min_keys", "near", "B", "seed", "queries")},
           "n_keys_listed": len(keys), "n_keys_used": len(recs), "excluded": {k: len(v) for k, v in failed.items()},
           "excluded_keys": failed, "n_states": n_states, "n_state_level_excluded_inview": n_excl, "level_keys": level_keys,
           "level_judged": level_ok, "estimates": est, "signs": sg, "outcome": outcome, "direction_specificity": dirspec,
           "copy_reading": copy_read, "D_b_by_queries_to_close": {k: {"n": len(v), "mean": float(np.mean(v))} for k, v in sorted(strat.items())},
           "D_b_by_C4_outcome": by_outcome}
    return res


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--src", default="")
    ap.add_argument("--a", type=float, default=12.0)
    ap.add_argument("--b", type=float, default=24.0)
    ap.add_argument("--inview-min", type=float, default=0.8)
    ap.add_argument("--min-keys", type=int, default=60)
    ap.add_argument("--near", type=float, default=0.05)
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--act-steps", type=int, default=4)
    ap.add_argument("--queries", choices=["preclose", "all"], default="preclose")
    a = ap.parse_args(argv)
    if os.path.exists(a.out):
        raise SystemExit(f"拒读：{a.out} 已存在（一次性读数）")
    res = run(a)
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps({k: res[k] for k in ("block", "n_keys_used", "excluded", "outcome", "direction_specificity", "copy_reading")},
                     ensure_ascii=False))
    return res


if __name__ == "__main__":
    main()
