"""第 0 步行为描述：从 feats.json 出全部表（事后描述，不进任何判定）。

用法：python3 -I summarize.py feats.json > tables.txt
所有阈值都是看过数据分布后定的（事后），报告里照写。
"""
import json
import sys

import numpy as np

R = json.load(open(sys.argv[1]))
BLOCKS = ["B1", "B2", "F1", "F2"]
CELLS = ["C0", "C1", "C3", "C4", "C5"]
idx = {(r["block"], r["cell"], r["key"]): r for r in R}
keys = {b: sorted({r["key"] for r in R if r["block"] == b}) for b in BLOCKS}

# 事后阈值（按身体）
POS_THR = {"widowx": 0.05, "google": 0.10}   # 指令积分位置与同键 C0 的持续分开阈值（米）
GRIP_W_SEC = 2.0                              # 夹爪指令状态与 C0 持续不同多长才算分开（秒）
MIN_RUN = 3                                   # 闭合段至少几步才计一次闭合（去抖）
START_THR = {"widowx": 0.05, "google": 0.10}  # 「启动」：累计指令路径达到多少米


def runs(gs):
    """闭合段 (起点, 长度)。"""
    out, i, n = [], 0, len(gs)
    while i < n:
        if gs[i] == 1:
            j = i
            while j < n and gs[j] == 1:
                j += 1
            out.append((i, j - i))
            i = j
        else:
            i += 1
    return out


def ep_feats(r):
    T, hz = r["T"], r["hz"]
    s = np.array(r["step_mag"])
    P = np.array(r["P"])
    gs = np.array(r["gs"])
    rr = [x for x in runs(gs) if x[1] >= MIN_RUN]
    fc = rr[0][0] if rr else None
    third = T // 3
    iou = np.array([np.nan if v is None else v for v in r["iou_lag_mask"]], float)
    G = np.array(r["G_px"])
    A, Rp = np.array(r["arm_px"]), np.array(r["R_px"])
    nq = len(G)
    q3 = slice(2 * nq // 3, nq)
    cum = np.cumsum(s)
    st = np.where(cum >= START_THR[r["body"]])[0]
    return dict(
        path=s.sum(), path_early=s[:third].sum(), mag_late=np.median(s[T // 2:]) * 1000,
        net=np.linalg.norm(P[-1]), z_end=P[-1, 2], rot_path=float(np.sum(np.linalg.norm(np.array(r["ea"])[:, 3:6], axis=1))),
        any_close=fc is not None, fc_sec=None if fc is None else fc / hz, n_close=len(rr),
        frac_closed_last=gs[2 * third:].mean(),
        t_start=None if not len(st) else st[0] / hz,
        img_move_late=np.nanmedian(1 - iou[q3]) if np.isfinite(iou[q3]).any() else np.nan,
        img_move_early=np.nanmedian(1 - iou[: nq // 3]) if np.isfinite(iou[: nq // 3]).any() else np.nan,
        arm_share_late=np.median(A[q3] / np.maximum(Rp[q3], 1)),
        g_out=(G == 0).mean(),
        fc_step=fc, gs=gs, P=P,
    )


F = {k: ep_feats(r) for k, r in idx.items()}


def med(xs, fmt="%.2f"):
    xs = [x for x in xs if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not xs:
        return "无"
    q = np.percentile(xs, [25, 50, 75])
    return (fmt % q[1]) + " [" + (fmt % q[0]) + ", " + (fmt % q[2]) + "]"


def pct(xs):
    xs = list(xs)
    return "%.0f%%" % (100 * np.mean(xs)) if xs else "无"


def persist_onset(mask):
    """mask[t] = 是否「不同」。返回最早的 t，使得 t 之后全都不同；从不满足返回 None。"""
    m = np.asarray(mask, bool)
    if not m[-1]:
        return None
    good = np.where(~m)[0]
    return 0 if not len(good) else int(good[-1] + 1)


def grip_onset(ga, gb, W):
    d = ga != gb
    n = 0
    for t, x in enumerate(d):
        n = n + 1 if x else 0
        if n >= W:
            return t - W + 1
    return None


ROWS = [("成功率", lambda f, r: None, None)]


def table_block_cell():
    print("## 表 1  逐块逐格的指令与画面特征（每格 120 键；中位数 [四分位]）")
    hdr = ["块", "格", "成功", "路径m", "前1/3路径m", "后半步长mm", "净位移m", "z终点m", "旋转路径rad", "曾闭合",
           "首次闭合s", "闭合次数", "末1/3闭合占比", "启动s", "图像动early", "图像动late", "臂像素占比late", "夹爪出画占比"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for b in BLOCKS:
        for c in CELLS:
            ks = [(b, c, k) for k in keys[b]]
            f = [F[x] for x in ks]
            print("| " + " | ".join([b, c, pct(idx[x]["success"] for x in ks),
                                     med([x["path"] for x in f]), med([x["path_early"] for x in f]),
                                     med([x["mag_late"] for x in f], "%.1f"), med([x["net"] for x in f]),
                                     med([x["z_end"] for x in f]), med([x["rot_path"] for x in f], "%.1f"),
                                     pct(x["any_close"] for x in f), med([x["fc_sec"] for x in f], "%.1f"),
                                     med([x["n_close"] for x in f], "%.0f"), med([x["frac_closed_last"] for x in f]),
                                     med([x["t_start"] for x in f], "%.1f"),
                                     med([x["img_move_early"] for x in f]), med([x["img_move_late"] for x in f]),
                                     med([x["arm_share_late"] for x in f]), "%.3f" % np.mean([x["g_out"] for x in f])]) + " |")
    print()


def divergence():
    print("## 表 2  与同键 C0 的分开时间（秒；每格 120 键）")
    print("第一次不同：env_actions 第一次出现任一维差 > 1e-9 的步。位置持续分开：指令积分位置与 C0 的距离从该步起直到回合结束都 > 阈值"
          "（WidowX 5 cm，Google 10 cm）；没分开记为「不分开」。夹爪持续分开：指令夹爪状态与 C0 连续不同至少 2 秒的起点。")
    hdr = ["块", "格", "第一次不同s 中位", "位置持续分开 占比", "位置持续分开s", "夹爪分开 占比", "夹爪分开s",
           "首次闭合差s（格减C0，两者都闭合的键）", "n两者都闭合"]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "---|" * len(hdr))
    for b in BLOCKS:
        for c in CELLS[1:]:
            fd, po, go, dfc = [], [], [], []
            for k in keys[b]:
                r0, r1 = idx[(b, "C0", k)], idx[(b, c, k)]
                f0, f1 = F[(b, "C0", k)], F[(b, c, k)]
                hz, body = r0["hz"], r0["body"]
                e0, e1 = np.array(r0["ea"]), np.array(r1["ea"])
                dd = np.where(np.abs(e0 - e1).max(1) > 1e-9)[0]
                fd.append(dd[0] / hz if len(dd) else None)
                d = np.linalg.norm(f0["P"] - f1["P"], axis=1)
                o = persist_onset(d > POS_THR[body])
                po.append(None if o is None else o / hz)
                g = grip_onset(f0["gs"], f1["gs"], int(round(GRIP_W_SEC * hz)))
                go.append(None if g is None else g / hz)
                if f0["fc_sec"] is not None and f1["fc_sec"] is not None:
                    dfc.append(f1["fc_sec"] - f0["fc_sec"])
            print("| " + " | ".join([b, c, med(fd, "%.1f"), pct(x is not None for x in po), med(po, "%.1f"),
                                     pct(x is not None for x in go), med(go, "%.1f"), med(dfc, "%.1f"), str(len(dfc))]) + " |")
    print()


FEATS = [("路径m", "path", "%.2f"), ("前1/3路径m", "path_early", "%.2f"), ("后半步长mm", "mag_late", "%.1f"),
         ("净位移m", "net", "%.2f"), ("z终点m", "z_end", "%.2f"), ("曾闭合", "any_close", None),
         ("首次闭合s", "fc_sec", "%.1f"), ("闭合次数", "n_close", "%.0f"), ("末1/3闭合占比", "frac_closed_last", "%.2f"),
         ("图像动late", "img_move_late", "%.2f"), ("臂像素占比late", "arm_share_late", "%.2f")]


def fmt_feat(fs, key, fm):
    if fm is None:
        return pct(f[key] for f in fs)
    return med([f[key] for f in fs], fm)


def subset_table(title, b, ks, cells):
    print(f"### {title}（{b}，n = {len(ks)}）")
    if not ks:
        print("无键\n")
        return
    print("| 量 | " + " | ".join(cells) + " |")
    print("|---|" + "---|" * len(cells))
    for name, key, fm in FEATS:
        print("| " + name + " | " + " | ".join(fmt_feat([F[(b, c, k)] for k in ks], key, fm) for c in cells) + " |")
    print()


def mean_traj_dist(b, ca, cb, k):
    return float(np.mean(np.linalg.norm(F[(b, ca, k)]["P"] - F[(b, cb, k)]["P"], axis=1)))


def grip_agree(b, ca, cb, k):
    return float(np.mean(F[(b, ca, k)]["gs"] == F[(b, cb, k)]["gs"]))


def close_loc(b, c, k):
    """该格首次闭合时的指令积分位置，与同键 C0 首次闭合时位置的距离（米）。"""
    f, f0 = F[(b, c, k)], F[(b, "C0", k)]
    if f["fc_step"] is None or f0["fc_step"] is None:
        return None
    return float(np.linalg.norm(f["P"][f["fc_step"]] - f0["P"][f0["fc_step"]]))


def reach(b, c, k):
    """该格指令积分轨迹离同键 C0 首次闭合点最近的距离（米），只看 C0 成功的键。"""
    f, f0 = F[(b, c, k)], F[(b, "C0", k)]
    if f0["fc_step"] is None or not idx[(b, "C0", k)]["success"]:
        return None
    return float(np.min(np.linalg.norm(f["P"] - f0["P"][f0["fc_step"]], axis=1)))


def table_reach():
    print("## 表 3  接近与闭合位置的指令积分代理（米；只用 C0 成功的键；中位数 [四分位]）")
    print("最近距离：该格指令积分轨迹离同键 C0 首次闭合点最近多少。闭合位置差：该格首次闭合时的指令积分位置与 C0 首次闭合位置的距离。")
    print("| 块 | 量 | " + " | ".join(CELLS) + " |")
    print("|---|---|" + "---|" * len(CELLS))
    for b in BLOCKS:
        ks = [k for k in keys[b] if idx[(b, "C0", k)]["success"]]
        print(f"| {b} (n={len(ks)}) | 最近距离 | " + " | ".join(med([reach(b, c, k) for k in ks], "%.3f") for c in CELLS) + " |")
        print(f"| {b} | 闭合位置差 | " + " | ".join(med([close_loc(b, c, k) for k in ks], "%.3f") for c in CELLS) + " |")
        thr = 0.03 if b.startswith("B") else 0.05
        print(f"| {b} | 最近距离 < {int(thr*100)} cm | " + " | ".join(
            pct((reach(b, c, k) or 9) < thr for k in ks) for c in CELLS) + " |")
    print()


def q2():
    print("## 表 4  C3 失败的键按 C4 成败分开")
    for b in BLOCKS:
        a = [k for k in keys[b] if not idx[(b, "C3", k)]["success"] and idx[(b, "C4", k)]["success"]]
        z = [k for k in keys[b] if not idx[(b, "C3", k)]["success"] and not idx[(b, "C4", k)]["success"]]
        subset_table("C3 失败、C4 成功", b, a, ["C0", "C3", "C4"])
        subset_table("C3、C4 都失败", b, z, ["C0", "C3", "C4"])
        for name, ks in (("C3 失败、C4 成功", a), ("C3、C4 都失败", z)):
            if ks:
                print(f"{b} {name}：最近距离 C3 {med([reach(b, 'C3', k) for k in ks], '%.3f')}，"
                      f"C4 {med([reach(b, 'C4', k) for k in ks], '%.3f')}；"
                      f"闭合位置差 C3 {med([close_loc(b, 'C3', k) for k in ks], '%.3f')}，C4 {med([close_loc(b, 'C4', k) for k in ks], '%.3f')}；"
                      f"夹爪状态与 C0 一致的步占比 C3 {med([grip_agree(b, 'C3', 'C0', k) for k in ks])}，C4 {med([grip_agree(b, 'C4', 'C0', k) for k in ks])}")
        print()


def q2_c5():
    print("## 表 5  C5 更像 C3 还是 C4（每块 120 键）")
    print("轨迹距离：指令积分位置逐步距离的回合平均（米）。夹爪一致：指令夹爪状态逐步相同的占比。")
    print("| 块 | d(C5,C3) | d(C5,C4) | d(C5,C3)<d(C5,C4) 的键占比 | 夹爪一致 C5~C3 | 夹爪一致 C5~C4 | d(C4,C3) 参照 | d(C1,C0) 参照 |")
    print("|---|---|---|---|---|---|---|---|")
    for b in BLOCKS:
        ks = keys[b]
        a = [mean_traj_dist(b, "C5", "C3", k) for k in ks]
        c4 = [mean_traj_dist(b, "C5", "C4", k) for k in ks]
        print(f"| {b} | {med(a, '%.3f')} | {med(c4, '%.3f')} | {pct(x < y for x, y in zip(a, c4))} | "
              f"{med([grip_agree(b, 'C5', 'C3', k) for k in ks])} | {med([grip_agree(b, 'C5', 'C4', k) for k in ks])} | "
              f"{med([mean_traj_dist(b, 'C4', 'C3', k) for k in ks], '%.3f')} | {med([mean_traj_dist(b, 'C1', 'C0', k) for k in ks], '%.3f')} |")
    print()


def q3():
    print("## 表 6  Google Robot：C0 成功而 C1 失败的键，C1 更像 C0 还是 C3")
    for b in ["F1", "F2"]:
        a = [k for k in keys[b] if idx[(b, "C0", k)]["success"] and not idx[(b, "C1", k)]["success"]]
        ok = [k for k in keys[b] if idx[(b, "C0", k)]["success"] and idx[(b, "C1", k)]["success"]]
        subset_table("C0 成功、C1 失败", b, a, ["C0", "C1", "C3", "C4"])
        subset_table("参照：C0、C1 都成功", b, ok, ["C0", "C1", "C3", "C4"])
        for name, ks in (("C0 成功 C1 失败", a), ("C0、C1 都成功", ok)):
            d10 = [mean_traj_dist(b, "C1", "C0", k) for k in ks]
            d13 = [mean_traj_dist(b, "C1", "C3", k) for k in ks]
            print(f"{b} {name}（n={len(ks)}）：d(C1,C0) {med(d10, '%.3f')}，d(C1,C3) {med(d13, '%.3f')}，"
                  f"更近 C3 的键占比 {pct(x > y for x, y in zip(d10, d13))}；夹爪一致 C1~C0 {med([grip_agree(b, 'C1', 'C0', k) for k in ks])}，"
                  f"C1~C3 {med([grip_agree(b, 'C1', 'C3', k) for k in ks])}；最近距离 C1 {med([reach(b, 'C1', k) for k in ks], '%.3f')}，"
                  f"C3 {med([reach(b, 'C3', k) for k in ks], '%.3f')}")
        print()


def q5():
    print("## 表 7  「轮廓只是让策略肯动起来」的检查（指令层面）")
    print("前 1/3 路径比：该格前三分之一回合的指令路径 / 同键 C0 的；启动：累计指令路径到 5 cm（WidowX）或 10 cm（Google）的时间。"
          "低启动：前 1/3 路径比 < 0.5。")
    print("| 块 | 格 | 前1/3路径比 中位 [四分位] | 低启动占比 | 启动s | 低启动键里的成功率 |")
    print("|---|---|---|---|---|---|")
    for b in BLOCKS:
        for c in ["C1", "C3", "C4", "C5"]:
            ratio = [F[(b, c, k)]["path_early"] / max(F[(b, "C0", k)]["path_early"], 1e-9) for k in keys[b]]
            low = [k for k, x in zip(keys[b], ratio) if x < 0.5]
            print(f"| {b} | {c} | {med(ratio)} | {pct(x < 0.5 for x in ratio)} | {med([F[(b, c, k)]['t_start'] for k in keys[b]], '%.1f')} | "
                  f"{pct(idx[(b, c, k)]['success'] for k in low) if low else '无'} (n={len(low)}) |")
    print()
    print("C3 失败、C4 成功的键上，C4 减 C3 的前 1/3 路径（米）与首次闭合（秒）：")
    for b in BLOCKS:
        a = [k for k in keys[b] if not idx[(b, "C3", k)]["success"] and idx[(b, "C4", k)]["success"]]
        dp = [F[(b, "C4", k)]["path_early"] - F[(b, "C3", k)]["path_early"] for k in a]
        print(f"  {b}（n={len(a)}）：前1/3路径差 {med(dp, '%.3f')}，C4 比 C3 大的键 {pct(x > 0 for x in dp)}；"
              f"C3 曾闭合 {pct(F[(b, 'C3', k)]['any_close'] for k in a)}，C4 曾闭合 {pct(F[(b, 'C4', k)]['any_close'] for k in a)}")
    print()


def qstats_corr():
    print("## 表 8  指令运动与真实画面运动是否一致（后 1/3 回合；每格 120 键）")
    print("图像动late = 1 - 当前与 L 步前真实可见夹爪掩膜的 IoU 在后三分之一查询上的中位数；后半步长 = 指令平移步长中位数。"
          "列出两者的 Spearman 相关，以及「指令大而图像几乎不动」的键数（后半步长 > 本块 C0 的 75 分位且图像动late < 0.1）。")
    print("| 块 | 格 | Spearman | 指令大而图像不动 键数 |")
    print("|---|---|---|---|")
    for b in BLOCKS:
        q75 = np.percentile([F[(b, "C0", k)]["mag_late"] for k in keys[b]], 75)
        for c in CELLS:
            x = np.array([F[(b, c, k)]["mag_late"] for k in keys[b]])
            y = np.array([F[(b, c, k)]["img_move_late"] for k in keys[b]])
            ok = np.isfinite(y)
            rx = np.argsort(np.argsort(x[ok]))
            ry = np.argsort(np.argsort(y[ok]))
            rho = np.corrcoef(rx, ry)[0, 1]
            n = int(np.sum((x > q75) & (y < 0.1)))
            print(f"| {b} | {c} | {rho:.2f} | {n} |")
    print()


table_block_cell()
divergence()
table_reach()
q2()
q2_c5()
q3()
q5()
qstats_corr()
