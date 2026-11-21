"""bench2 补充协议 v1.1 的一次性读数（protocol_bench2_ext_fill_lag.md §1、§3、§4）。只在协调方冻结后对真实数据运行；
执行方只在合成数据上跑。保护沿用第一阶段 b2/readout.py：
  * 键清单必须是冻结的那份（sha256 与键数都核），否则拒读，不写任何输出；
  * 块级量（FL 是否低反差、操纵强度、参考轨迹上的滞后几何量、各格滞后）只从冻结清单取，缺字段拒读；
  * 门槛现算（gate.run），不读外部 gate.json；某格某键既无记录也无报错文件 = 未采完，拒读；只有报错文件 = 剔除；
  * 输出固定在 <root>/READOUT_bench2_ext_<块>.txt / .json，任一已存在就拒读；两块合起来的句子用 --combine 另出
    <root>/READOUT_bench2_ext_combined.txt，同样一次性；
  * 根目录没有 SYNTHETIC_DATA 标记时，b2x/FROZEN_bench2_ext 必须存在；
  * §1：任何重跑的旧格（C0、C1、C3、C4、C5）与第一阶段正式记录有不逐位相同的键（或参照缺失），新格只报数、不判读法。
估计量（百分点；bootstrap 同第一阶段：B2 按 episode_id 聚 24 簇，F1 按键）：
  D(x) = SR(Fx) - SR(F0)，x = k（C1）、w（C1w，白）、L（C1L，浅色）；E = SR(FL) - SR(Fk)；
  勺子块的 D(L) 与 E、可乐罐块的 D(w) 只照报（v1.1 §3）；
  F1 另有 T = SR(C4) - SR(C5L)，U = SR(C5L) - SR(C3)，V = SR(C5) - SR(C5L)，并列 Δ2 = SR(C4) - SR(C5)。
判法（§1）：差值「提高 / 未提高 / 不确定」；相对原图「掉分 / 几乎不掉 / 中间」；「显著为负」= 区间上界 < 0。
读法（§3 两张表、两块合起来的两句、§4 五条）事先写死在下面，输出每块落在哪一行。
"""
import argparse
import collections
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate as G  # noqa: E402
import keys as K  # noqa: E402

# ---------------------------------------------------------------- 判定线（§1，沿用第一阶段，不调），全部在这里
B_BOOT = 10000
BOOT_SEED = 0
CI = (2.5, 97.5)
ADMIT_SR_C0 = 60.0          # 准入（第一阶段 §5）；补充协议没写，按「没写到的照第一阶段」
DROP_POINT_MAX = -15.0      # 「掉分」：点估计 <= -15 且上界 < 0
DROP_HI_MAX = 0.0
NEAR_LO_MIN = -10.0         # 「几乎不掉」：下界 >= -10；两者之间叫「中间」
P_POINT_MIN = 10.0          # 「提高」：点估计 >= 10 且下界 > 0
P_LO_MIN = 0.0
P_NOT_HI_MAX = 10.0         # 「未提高」：上界 < 10
SIG_NEG_HI_MAX = 0.0        # 「显著为负」：上界 < 0
EPS = 1e-9
FROZEN = G.FROZEN
CONTRASTS = {
    "B2": {"D(k)": ("C1", "C0"), "D(w)": ("C1w", "C0"), "D(L)": ("C1L", "C0"), "E": ("C1L", "C1")},
    "F1": {"D(k)": ("C1", "C0"), "D(w)": ("C1w", "C0"), "D(L)": ("C1L", "C0"), "E": ("C1L", "C1"),
           "T": ("C4", "C5L"), "U": ("C5L", "C3"), "V": ("C5", "C5L"), "Δ2": ("C4", "C5")},
}
REPORT_ONLY = {"B2": ("D(L)", "E"), "F1": ("D(w)", "Δ2")}  # 只印数字、不进读法（v1.1 §3）
TABLE_FILL = {"B2": "w", "F1": "L"}  # 各块读法表用的那种填充（低反差时该块的表不判）
LAG_CELLS = ("C3", "C4", "C5", "C5L")
F1_TABLE = {  # §3 抓可乐罐（浅色接近原貌，黑色远离原貌）
    "F1-1": ("D(L) 几乎不掉，E 提高", "用接近原貌的颜色平涂几乎不掉分；这个任务上涂黑的掉分主要与替换的颜色有关，不能读成盖住了需要的信息"),
    "F1-2": ("D(L) 中间，E 提高", "接近原貌的平涂掉得比涂黑少；并排报 D(L) 与 D(k)，不写「主要」「一部分」这类比例词"),
    "F1-3": ("D(L) 掉分，E 提高", "平涂本身就伤，涂黑更伤；并排报两个数"),
    "F1-4": ("D(L) 掉分，E 不确定或未提高", "换成接近原貌的颜色没有明显好转；只能写「平涂本身就伤」，不能写「与颜色无关」，因为平涂还抹掉了内部边界"),
    "F1-5": ("D(L) 任意，E 显著为负", "浅色比黑色更差，照写；这是草案没有预期的方向"),
    "F1-6": ("其余", "照报数字，写「所测填充下未能区分」"),
}
B2_TABLE = {  # §3 勺子放毛巾（黑色接近原貌，白色远离原貌且与背景反差高）
    "B2-1": ("D(w) 掉分", "远离原貌的平涂在这个任务上也伤；第一阶段「涂黑不掉」与黑色接近原貌有关"),
    "B2-2": ("D(w) 几乎不掉", "远离原貌的平涂在这个任务上也几乎不掉；并排报它的操纵强度"),
    "B2-2s": ("D(w) 几乎不掉，且白色强度低于可乐罐上黑色的强度", "所测强度下不掉"),
    "B2-3": ("D(w) 中间", "照报数字"),
}
COMBINED = {
    "合-1": ("可乐罐 D(L) 几乎不掉，且勺子 D(w) 掉分", "两个任务上，接近原貌的平涂都不掉、远离原貌的平涂都掉"),
    "合-2": ("可乐罐 D(L) 掉分，且勺子 D(w) 几乎不掉", "可乐罐任务上接近原貌的平涂也掉，勺子任务上远离原貌的平涂也不掉，差别跟着任务块走"),
}
COMBINED_SUFFIX = "两块用的远离原貌的颜色不同（可乐罐是黑、勺子是白）"
LOW_CONTRAST_TEXT = "平涂且与背景反差低时掉分"
FILL_NOTE = "不写「纹理不重要」「外观不重要」，也不写「涂黑不掉在 Google Robot 上复现」；结论的主语是任务块"
LAG_TEXT = {
    "L-1": ("T 提高，且 V 的区间下界 > 0", "滞后 {s4} 秒时与当前轮廓分不出，滞后更长（{sl} 秒）时当前轮廓明显更好"),
    "L-2": ("T 提高，但 V 的区间下界 <= 0", "当前轮廓好于更长滞后（{sl} 秒）的轮廓；更长滞后与 {s4} 秒滞后之间没有分出差别"),
    "L-3": ("T 未提高", "在更长的所测滞后（{sl} 秒）下仍有容忍度，不外推到任意滞后"),
    "L-4": ("T 不确定", "第一阶段那一句不改，写「所测条件下未确定时效性优势」"),
    "L-5": ("U 提高", "数秒前的轮廓仍比不画好（不许写成「有个图形就行」）"),
}
LAG_NOTE = "并排给出 §2 的几何量与「轮廓不动的那一段」的占比；T、U、V 都不单独归因于时间"


class Refuse(Exception):
    pass


def ge(x, t):
    return x >= t - EPS


def le(x, t):
    return x <= t + EPS


def gt(x, t):
    return x > t + EPS


def lt(x, t):
    return x < t - EPS


# ---------------------------------------------------------------- 载入
def cells_of(block):
    return tuple(K.EXT_CELLS[block])


def load_block(root, block, keys, gate, field):
    excl = set(gate.get("excluded_keys", []))
    kept = [k for k in keys if k["key"] not in excl]
    if not kept:
        raise Refuse(f"{block}：门槛后没有保留的键")
    succ = {}
    for c in cells_of(block):
        v = []
        for k in kept:
            p = os.path.join(root, block, c, k["key"] + ".npz")
            if not os.path.exists(p):
                raise Refuse(f"{block}：块不完整，缺 {c}/{k['key']}")
            with np.load(p, allow_pickle=False) as z:
                if field not in z.files or not bool(z["complete"]):
                    raise Refuse(f"{block}：{c}/{k['key']} 缺 {field} 或不完整")
                v.append(z[field].copy() if field == "success_steps" else bool(z[field]))
        succ[c] = v if field == "success_steps" else np.array(v, np.float64)
    return succ, [k["cluster"] for k in kept], kept


def check_collected(root, block, keys):
    for c in cells_of(block):
        for k in keys:
            if not os.path.exists(os.path.join(root, block, c, k["key"] + ".npz")) and \
                    not os.path.exists(os.path.join(root, block, c, "_errors", k["key"] + ".txt")):
                raise Refuse(f"{block}：块不完整，{c}/{k['key']} 既没有记录也没有报错文件（采集未完成）")


# ---------------------------------------------------------------- 估计（与第一阶段同一实现）
def estimate(succ, clusters, contrasts, B=B_BOOT, seed=BOOT_SEED):
    labs, inv = np.unique(np.asarray(clusters), return_inverse=True)
    nC = len(labs)
    cnt = np.bincount(inv, minlength=nC).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nC, size=(B, nC))
    w = np.zeros((B, nC))
    np.add.at(w, (np.repeat(np.arange(B), nC), idx.ravel()), 1.0)
    out = {"n_keys": int(len(inv)), "n_clusters": int(nC), "SR": {c: 100.0 * float(v.mean()) for c, v in succ.items()}}
    for name, (a, b) in contrasts.items():
        d = succ[a] - succ[b]
        sums = np.bincount(inv, weights=d, minlength=nC)
        boot = 100.0 * (w @ sums) / (w @ cnt)
        lo, hi = np.percentile(boot, CI)
        out[name] = {"point": 100.0 * float(d.mean()), "lo": float(lo), "hi": float(hi)}
    return out


# ---------------------------------------------------------------- 判定
def p_level(e):
    if ge(e["point"], P_POINT_MIN) and gt(e["lo"], P_LO_MIN):
        return "提高"
    if lt(e["hi"], P_NOT_HI_MAX):
        return "未提高"
    return "不确定"


def drop_level(e):
    if le(e["point"], DROP_POINT_MAX) and lt(e["hi"], DROP_HI_MAX):
        return "掉分"
    if ge(e["lo"], NEAR_LO_MIN):
        return "几乎不掉"
    return "中间"


def sig_neg(e):
    return lt(e["hi"], SIG_NEG_HI_MAX)


def fill_rows(block, est, info):
    """§3：返回（行号列表, 附注列表）。本块读法表用的填充（可乐罐 FL、勺子白）被标低反差时不进表、不判。"""
    x = TABLE_FILL[block]
    dl = drop_level(est[f"D({x})"])
    notes = []
    if info["low_contrast"][x]:
        notes.append(f"{'FL' if x == 'L' else '白色'} 在本块被标低反差，涉及它的读法不判；D({x}) {dl}"
                     + (f"：写「{LOW_CONTRAST_TEXT}」" if dl == "掉分" else ""))
        return [], notes
    if block == "F1":
        e = est["E"]
        el = p_level(e)
        rows = []
        if el == "提高":
            rows.append({"几乎不掉": "F1-1", "中间": "F1-2", "掉分": "F1-3"}[dl])
        elif dl == "掉分":
            rows.append("F1-4")
        if sig_neg(e):
            rows.append("F1-5")
        return rows or ["F1-6"], notes
    row = {"掉分": "B2-1", "几乎不掉": "B2-2", "中间": "B2-3"}[dl]
    if row == "B2-2":
        sw, sk = info["strength_w_here"], info["strength_k_F1"]
        if lt(sw, sk):  # v1.1：直接比大小
            row = "B2-2s"
        notes.append(f"并排的操纵强度：白色在勺子块 {sw:.1f}，可乐罐块上的黑色 {sk:.1f}")
    return [row], notes


def lag_rows(est, info):
    wk = info["outline_weak"]
    out = {}
    out["T"] = "不判（轮廓操纵不足）" if (wk.get("C4") or wk.get("C5L")) else p_level(est["T"])
    out["U"] = "不判（C5L 轮廓操纵不足）" if wk.get("C5L") else p_level(est["U"])
    out["V"] = "不判（轮廓操纵不足）" if (wk.get("C5") or wk.get("C5L")) else ("下界 > 0" if gt(est["V"]["lo"], 0.0) else "下界 <= 0")
    rows = []
    if out["T"] == "提高":
        rows.append("L-1" if out["V"] == "下界 > 0" else "L-2")
    elif out["T"] == "未提高":
        rows.append("L-3")
    elif out["T"] == "不确定":
        rows.append("L-4")
    if out["U"] == "提高":
        rows.append("L-5")
    g = info["geom_ref"]
    i4, il = g[str(info["lag"])]["all"]["iou_median"], g[str(info["lag_long"])]["all"]["iou_median"]
    out["geom_clause"] = (f"参考轨迹上更长滞后的 IoU 中位数 {il:.3f} 不低于 4 步的 {i4:.3f}：只写「更陈旧」，不写「更错位」"
                          if ge(il, i4) else f"参考轨迹上更长滞后的 IoU 中位数 {il:.3f} 低于 4 步的 {i4:.3f}")
    return rows, out


def decide(block, est, info):
    r = {"info": info, "levels": {}}
    for n in CONTRASTS[block]:
        if n.startswith("D("):
            r["levels"][n] = drop_level(est[n])
        elif n in ("E", "T", "U", "Δ2"):
            r["levels"][n] = p_level(est[n])
        if n not in ("T", "U", "V") and sig_neg(est[n]):
            r["levels"][n] += "（显著为负）"
    if not info["rerun_ok"]:
        r.update(admitted=None, numbers_only=True, reason="重跑的旧格与第一阶段记录不逐位相同（或参照缺失）：新格只报数、不判读法")
        return r
    if not ge(est["SR"]["C0"], ADMIT_SR_C0):
        r.update(admitted=False, numbers_only=True, reason="基线不足（SR(C0) < 60），只报数")
        return r
    r.update(admitted=True, numbers_only=False)
    rows, notes = fill_rows(block, est, info)
    r.update(fill_rows=rows, fill_notes=notes)
    if block == "F1":
        lr, lo = lag_rows(est, info)
        r.update(lag_rows=lr, lag=lo)
    return r


def table_of(block):
    return F1_TABLE if block == "F1" else B2_TABLE


def fmt(block, est, dec, tag, contrasts, extra=None):
    s = [f"[{block}] {tag}：键 {est['n_keys']}（{est['n_clusters']} 簇）；SR " +
         ", ".join(f"{c}={v:.1f}" for c, v in est["SR"].items())]
    for n in contrasts:
        e = est[n]
        ro = n in REPORT_ONLY[block]
        lab = "" if (dec is None or ro) else f"  {dec['levels'].get(n, '')}"
        s.append(f"  {n} = {e['point']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}]{lab}" + ("  （只照报）" if ro else ""))
    if dec is None:
        return "\n".join(s)
    if dec["numbers_only"]:
        s.append(f"  {dec['reason']}")
        return "\n".join(s)
    info = dec["info"]
    s.append(f"  操纵强度（参考轨迹）：{info['strength_here']}；环反差：{info['ring_here']}；低反差：{info['low_contrast']}")
    for row in dec["fill_rows"]:
        cond, text = table_of(block)[row]
        s.append(f"  填充表 {row}【{cond}】：{text}")
    for n in dec["fill_notes"]:
        s.append(f"  填充附注：{n}")
    s.append(f"  （{FILL_NOTE}）")
    if "lag" in dec:
        lo = dec["lag"]
        s.append(f"  滞后：T {lo['T']}；U {lo['U']}；V {lo['V']}")
        for row in dec["lag_rows"]:
            cond, text = LAG_TEXT[row]
            s.append(f"  滞后读法 {row}【{cond}】：" + text.format(s4=f"{info['lag_sec']:.2f}", sl=f"{info['lag_long_sec']:.2f}"))
        if not dec["lag_rows"]:
            s.append("  滞后读法：T 不判")
        s.append(f"  几何：{lo['geom_clause']}；参考轨迹几何量 {json.dumps(info['geom_ref'], ensure_ascii=False)}")
        s.append(f"  轮廓不动的那一段：{json.dumps(info['frozen_segment'], ensure_ascii=False)}")
        s.append(f"  （{LAG_NOTE}）")
    if extra:
        s.append(extra)
    return "\n".join(s)


def info_for(block, gate, man):
    mb = man.get("blocks", {}).get(block)
    need = ["fill_strength", "ring_contrast", "low_contrast"]
    if mb is None or any(f not in mb for f in need):
        raise Refuse(f"冻结清单 blocks.{block} 缺 {need}")
    if any(x not in mb["low_contrast"] for x in ("k", "w", "L")) or any(x not in mb["fill_strength"] for x in ("k", "w", "L")):
        raise Refuse(f"冻结清单 blocks.{block} 的 low_contrast / fill_strength 缺 k / w / L")
    other = "F1" if block == "B2" else "B2"
    info = {"low_contrast": {x: bool(v) for x, v in mb["low_contrast"].items()},
            "strength_here": mb["fill_strength"], "ring_here": mb["ring_contrast"],
            "strength_L_here": float(mb["fill_strength"]["L"]), "strength_w_here": float(mb["fill_strength"]["w"]),
            "strength_k_F1": float(man["blocks"].get("F1", {}).get("fill_strength", {}).get("k", float("nan")))
            if block == "B2" else float(mb["fill_strength"]["k"]),
            "other_block": other}
    if block == "F1":
        for f in ("geom_ref", "cell_lag", "control_freq", "frozen_segment"):
            if f not in mb:
                raise Refuse(f"冻结清单 blocks.F1 缺 {f}")
        lag, lag_long = int(mb["cell_lag"]["C5"]), int(mb["cell_lag"]["C5L"])
        for l in (lag, lag_long):
            if str(l) not in mb["geom_ref"] or "all" not in mb["geom_ref"][str(l)]:
                raise Refuse(f"冻结清单 blocks.F1.geom_ref 缺滞后 {l} 的 all 段")
        info.update(geom_ref={str(l): mb["geom_ref"][str(l)] for l in (lag, lag_long)}, lag=lag, lag_long=lag_long,
                    lag_sec=lag / float(mb["control_freq"]), lag_long_sec=lag_long / float(mb["control_freq"]),
                    frozen_segment=mb["frozen_segment"])
    cf = gate.get("cell_flags", {})
    info["outline_weak"] = {c: G.OUTLINE_WEAK in cf.get(c, []) for c in ("C4", "C5", "C5L")}
    r13 = gate.get("items", {}).get("13_rerun_identical_vs_ref")
    if isinstance(r13, dict):
        info["rerun"] = {c: v for c, v in r13.items() if c != "ref_root"}
        info["rerun_ok"] = all(not v["differ"] and v["ref_missing"] == 0 and v["identical"] == v["compared"]
                               for v in info["rerun"].values())
    else:
        info["rerun"], info["rerun_ok"] = r13, False
    return info


def lag_extras(root, block, kept):
    """§4 并列描述（不判）：任一步成功、成功回合的首次成功步数中位数、按罐子朝向分开的成功数。"""
    lines, out = [], {}
    if block != "F1":
        return "", out
    for c in LAG_CELLS:
        ever = fin = 0
        steps = []
        by_task = collections.Counter()
        n_task = collections.Counter()
        for k in kept:
            with np.load(os.path.join(root, block, c, k["key"] + ".npz"), allow_pickle=False) as z:
                s, e = bool(z["success"]), bool(z["success_ever"])
                ss = np.asarray(z["success_steps"], bool) if "success_steps" in z.files else None
            t = k["task"].replace("google_robot_pick_", "").replace("_coke_can", "")
            n_task[t] += 1
            fin += s
            ever += e
            by_task[t] += s
            if s and ss is not None and ss.any():
                steps.append(int(np.argmax(ss)) + 1)
        out[c] = {"success_final": fin, "success_ever": ever, "first_success_step_median": float(np.median(steps)) if steps else None,
                  "by_orientation": {t: f"{by_task[t]}/{n_task[t]}" for t in sorted(n_task)}}
        lines.append(f"  并列（只描述）{c}：最后一步成功 {fin}，任一步成功 {ever}，成功回合首次成功步数中位数 "
                     f"{out[c]['first_success_step_median']}，按朝向 {out[c]['by_orientation']}")
    return "\n".join(lines), out


def readout(root, block, keys, gate, man):
    info = info_for(block, gate, man)
    con = CONTRASTS[block]
    res = {"block": block, "excluded_keys": gate.get("excluded_keys", [])}
    succ, cl, kept = load_block(root, block, keys, gate, "success")
    est = estimate(succ, cl, con)
    dec = decide(block, est, info)
    extra, lag_desc = lag_extras(root, block, kept)
    res["final"] = {"estimate": est, "decision": dec}
    res["lag_descriptive"] = lag_desc
    succ2, cl2, _ = load_block(root, block, keys, gate, "success_ever")
    est2 = estimate(succ2, cl2, con)
    res["ever"] = {"estimate": est2, "note": "并列读数，不进判定"}
    res["text"] = fmt(block, est, dec, "主读数（最后一步）", con, extra) + "\n" + \
        fmt(block, est2, None, "并列：任一步成功过", con)
    return res


def out_paths(root, block):
    b = os.path.join(root, f"READOUT_bench2_ext_{block}")
    return b + ".txt", b + ".json"


def combine(root):
    """两块合起来只允许的两句（§3）。读两块各自的读数 json，不再读任何记录。"""
    out = os.path.join(root, "READOUT_bench2_ext_combined.txt")
    if os.path.exists(out):
        raise Refuse(f"{out} 已存在，拒绝第二次")
    decs = {}
    for b in ("B2", "F1"):
        js = out_paths(root, b)[1]
        if not os.path.exists(js):
            raise Refuse(f"缺 {js}：两块都读完再合")
        decs[b] = json.load(open(js))["final"]["decision"]
    lines = []
    if any(decs[b].get("numbers_only") for b in decs):
        lines.append("有块只报数（重跑不一致或基线不足），不写合并句")
    elif decs["F1"]["info"]["low_contrast"]["L"] or decs["B2"]["info"]["low_contrast"]["w"]:
        lines.append("可乐罐的 FL 或勺子的白色被标低反差，不写合并句")
    else:
        f1, b2 = decs["F1"]["levels"]["D(L)"].split("（")[0], decs["B2"]["levels"]["D(w)"].split("（")[0]
        row = "合-1" if (f1 == "几乎不掉" and b2 == "掉分") else "合-2" if (f1 == "掉分" and b2 == "几乎不掉") else None
        lines.append(f"可乐罐 D(L) {f1}；勺子 D(w) {b2}")
        lines.append(f"合并句 {row}【{COMBINED[row][0]}】：{COMBINED[row][1]}；{COMBINED_SUFFIX}" if row else "其余组合：不写合并句")
    fd = os.open(out, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    return lines


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", choices=sorted(CONTRASTS))
    ap.add_argument("--keys", help="冻结的正式键清单（keysets/formal_<块>.json），哈希须与清单一致")
    ap.add_argument("--combine", action="store_true", help="两块都读完后，出两块合起来的句子")
    a = ap.parse_args(argv)
    if a.combine:
        return combine(a.root)
    if not a.block or not a.keys:
        raise Refuse("需要 --block 与 --keys（或 --combine）")
    txt, js = out_paths(a.root, a.block)
    if os.path.exists(txt) or os.path.exists(js):
        raise Refuse(f"{txt} 已存在，拒绝第二次读数")
    synthetic = os.path.exists(os.path.join(a.root, "SYNTHETIC_DATA"))
    if synthetic:
        mp = os.path.join(a.root, "SYNTHETIC_MANIFEST.json")
        if not os.path.exists(mp):
            raise Refuse("合成数据缺 SYNTHETIC_MANIFEST.json")
        ref_root = os.path.join(a.root, "SYNTHETIC_REF")  # 合成的第一阶段参照
    else:
        if not os.path.exists(FROZEN):
            raise Refuse("真实数据只能在补充协议冻结（FROZEN_bench2_ext 存在）后读")
        mp = FROZEN
        ref_root = G.REF_ROOT
    with open(mp) as f:
        man = json.load(f)
    with open(a.keys, "rb") as f:
        raw = f.read()
    want = man.get("keylist_sha256", {}).get(a.block)
    got = hashlib.sha256(raw).hexdigest()
    if want is None or got != want:
        raise Refuse(f"--keys {a.keys} 的 sha256 {got} 与冻结清单 keylist_sha256.{a.block}={want} 不符")
    all_keys = json.loads(raw)
    keys = [k for k in all_keys if k["block"] == a.block]
    n_want = man.get("blocks", {}).get(a.block, {}).get("n_keys")
    if n_want is None or len(keys) != int(n_want) or len(all_keys) != len(keys):
        raise Refuse(f"--keys 键数 {len(keys)}（文件共 {len(all_keys)}）与冻结清单 n_keys={n_want} 不符")
    info_for(a.block, {}, man)  # 清单字段缺失时在动任何数据之前就拒读
    check_collected(a.root, a.block, keys)
    gate = G.run(a.root, a.block, keys, manifest=man, ref_root=ref_root)
    res = readout(a.root, a.block, keys, gate, man)
    res["gate_summary"] = G.summary(gate)
    res["manifest_path"] = mp
    fd = os.open(txt, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
    with os.fdopen(fd, "w") as fh:
        fh.write(res["text"] + "\n\n" + res["gate_summary"] + "\n")
    with open(js + ".tmp", "w") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=1)
    os.replace(js + ".tmp", js)
    print(res["text"])
    return res


if __name__ == "__main__":
    try:
        main()
    except Refuse as e:
        print("拒读：", e, file=sys.stderr)
        sys.exit(2)
