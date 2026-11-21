"""回放（真实状态）的表。用法：python3 -I replay_summary.py replay_feats.json > replay_tables.txt"""
import json
import sys
from collections import Counter

import numpy as np

R = json.load(open(sys.argv[1]))
BLOCKS = ["B1", "B2", "F1", "F2"]
CELLS = ["C0", "C1", "C3", "C4", "C5"]
I = {(r["block"], r["cell"], r["key"]): r for r in R}
K = {b: sorted({r["key"] for r in R if r["block"] == b}) for b in BLOCKS}
SN = {0: "没接近", 1: "接近没碰", 2: "碰到或推动没夹住", 3: "夹住没搬动/抬起", 4: "搬动或抬起但最后没成", 5: "成功"}


def med(xs, f="%.2f"):
    xs = [x for x in xs if x is not None]
    if not xs:
        return "无"
    q = np.percentile(xs, [25, 50, 75])
    return f % q[1] + " [" + f % q[0] + ", " + f % q[2] + "]"


def pct(xs):
    xs = list(xs)
    return "%.0f%%" % (100 * np.mean(xs)) if xs else "无"


def stage_dist(rs):
    c = Counter(r["stage"] for r in rs)
    return " / ".join("%d" % c.get(s, 0) for s in range(6))


print("回放可信性：%d 条，success_steps 逐位一致 %d 条" % (len(R), sum(r["match"] for r in R)))
print()
print("## 表 R1  每格最远到过的阶段（键数，顺序：" + " / ".join(f"{k}{v}" for k, v in SN.items()) + "）")
print("| 块 | " + " | ".join(CELLS) + " |")
print("|---|" + "---|" * len(CELLS))
for b in BLOCKS:
    print(f"| {b} | " + " | ".join(stage_dist([I[(b, c, k)] for k in K[b]]) for c in CELLS) + " |")
print()

FEAT = [("末端离任务物最近 m", "dmin", "%.3f"), ("首次进入 8 cm s", "t_near", "%.1f"), ("手指碰到", "contact", None),
        ("两指都碰到", "both_contact", None), ("夹住过", "grasp", None), ("首次夹住 s", "t_grasp", "%.1f"),
        ("夹住步占比", "grasp_frac", "%.2f"), ("最后一步仍夹住", "grasp_end", None), ("任务物最高抬升 m", "lift_max", "%.3f"),
        ("任务物最大水平位移 m", "xy_max", "%.3f"), ("没夹住而被推动 >=3cm", "pushed_no_grasp", None),
        ("末端真实路径 m", "tcp_path", "%.2f"), ("前1/3 真实路径 m", "tcp_path_early", "%.2f"),
        ("后半 真实路径 m", "tcp_path_late", "%.2f"), ("末端最低点相对任务物初始高度 m", "tcp_z_min_rel_src0", "%.3f"),
        ("夹爪实际开度最小", "grip_width_min", "%.2f"), ("开度<0.5 的步占比", "grip_closed_frac", "%.2f")]


def ftab(title, b, ks, cells):
    print(f"### {title}（{b}，n = {len(ks)}）")
    if not ks:
        print("无键\n")
        return
    print("| 量 | " + " | ".join(cells) + " |")
    print("|---|" + "---|" * len(cells))
    print("| 阶段分布 0/1/2/3/4/5 | " + " | ".join(stage_dist([I[(b, c, k)] for k in ks]) for c in cells) + " |")
    for name, key, f in FEAT:
        vals = []
        for c in cells:
            xs = [I[(b, c, k)][key] for k in ks]
            vals.append(pct(xs) if f is None else med(xs, f))
        print(f"| {name} | " + " | ".join(vals) + " |")
    print()


print("## 表 R2  逐块逐格真实状态特征（每格 120 键）")
for b in BLOCKS:
    ftab("全部键", b, K[b], CELLS)

print("## 表 R3  C3 失败的键：C4 救回的与没救回的")
for b in BLOCKS:
    a = [k for k in K[b] if not I[(b, "C3", k)]["success"] and I[(b, "C4", k)]["success"]]
    z = [k for k in K[b] if not I[(b, "C3", k)]["success"] and not I[(b, "C4", k)]["success"]]
    ftab("C3 失败、C4 成功", b, a, ["C0", "C3", "C4"])
    ftab("C3、C4 都失败", b, z, ["C0", "C3", "C4"])
    ct = Counter((I[(b, "C3", k)]["stage"], I[(b, "C4", k)]["stage"]) for k in a + z)
    print(f"{b} 交叉表（C3 阶段 -> C4 阶段，C3 失败的 {len(a) + len(z)} 键）：" +
          "，".join(f"{x}->{y}: {n}" for (x, y), n in sorted(ct.items())))
    print()

print("## 表 R4  Google Robot：C0 成功而 C1 失败的键")
for b in ["F1", "F2"]:
    a = [k for k in K[b] if I[(b, "C0", k)]["success"] and not I[(b, "C1", k)]["success"]]
    ftab("C0 成功、C1 失败", b, a, ["C0", "C1", "C3", "C4"])

print("## 表 R5  指令积分与真实末端运动")
print("| 块 | 格 | 真实路径/指令路径 中位 | Spearman(指令路径, 真实路径) | 前1/3 真实/指令 中位 |")
print("|---|---|---|---|---|")
for b in BLOCKS:
    for c in CELLS:
        rs = [I[(b, c, k)] for k in K[b]]
        x = np.array([r["cmd_path"] for r in rs])
        y = np.array([r["tcp_path"] for r in rs])
        rho = np.corrcoef(np.argsort(np.argsort(x)), np.argsort(np.argsort(y)))[0, 1]
        print(f"| {b} | {c} | {med(list(y / x))} | {rho:.2f} | "
              f"{med([r['tcp_path_early'] / r['cmd_path_early'] for r in rs])} |")
print()

print("## 表 R6  前 1/3 回合真实末端路径：该格 / 同键 C0（检查「轮廓只是让它动起来」）")
print("| 块 | C1 | C3 | C4 | C5 | C3 失败 C4 成功键上 C4 减 C3（米） |")
print("|---|---|---|---|---|---|")
for b in BLOCKS:
    cols = [med([I[(b, c, k)]["tcp_path_early"] / I[(b, "C0", k)]["tcp_path_early"] for k in K[b]]) for c in ["C1", "C3", "C4", "C5"]]
    a = [k for k in K[b] if not I[(b, "C3", k)]["success"] and I[(b, "C4", k)]["success"]]
    d = [I[(b, "C4", k)]["tcp_path_early"] - I[(b, "C3", k)]["tcp_path_early"] for k in a]
    print(f"| {b} | " + " | ".join(cols) + f" | {med(d, '%.3f')} (n={len(a)}) |")
