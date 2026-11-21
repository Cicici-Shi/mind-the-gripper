"""Google Robot 夹爪：策略原始输出的闭合意图 与 粘滞闸门之后实际送进环境的指令。

用法：python3 -I grip_intent.py <out_formal 目录>
raw_actions[..., 6] 是策略输出（训练时 0 = 闭合，1 = 张开；适配器 simpler.py 第 222 到 252 行）。
意图闭合 := 原始输出 < 0.5（取反后相对指令 > 0，正为闭合）。
同时核对：粘滞计数按 postprocess 处理的全部 4 个动作计，只执行前 2 个，所以闸门开放的步固定。
"""
import glob
import os
import sys
from collections import Counter

import numpy as np

ROOT = sys.argv[1]


def gate_steps(n_queries, horizon=4, act=2, rep=15):
    """假设每个窗口起点都触发粘滞（输出 |相对指令| > 0.5），闸门在哪些执行步重新取值。"""
    out = []
    for c in range(0, n_queries * horizon, rep):
        q, pos = divmod(c, horizon)
        step = q * act + pos if pos < act else (q + 1) * act
        out.append(step)
    return out


print("理论闸门步（每窗口起点都触发粘滞时）:", gate_steps(40))
for b in ["F1", "F2", "B1", "B2"]:
    for c in ["C0", "C1", "C3", "C4", "C5"]:
        anyi, firsti, frac, gated, n = [], [], [], [], 0
        fc_steps = Counter()
        for f in sorted(glob.glob(os.path.join(ROOT, b, c, "*.npz"))):
            z = np.load(f)
            ra = z["raw_actions"][:, :, 6]
            qe = int(z["query_every"])
            ex = ra[:, :qe]  # 执行了的那几个
            intent = ex < 0.5
            idx = np.where(intent.reshape(-1))[0]
            anyi.append(len(idx) > 0)
            firsti.append(idx[0] / float(z["control_freq"]) if len(idx) else None)
            frac.append(intent.mean())
            ea = z["env_actions"][:, 6]
            closed_cmd = (ea > 0.5) if b.startswith("F") else (ea < 0)
            gated.append(len(idx) > 0 and not closed_cmd.any())
            n += 1
        fi = [x for x in firsti if x is not None]
        print(f"{b} {c}: 执行动作里出现闭合意图的回合 {100*np.mean(anyi):.0f}%，首次意图 s 中位 "
              f"{np.median(fi) if fi else float('nan'):.1f}，意图步占比均值 {np.mean(frac):.2f}，"
              f"有意图但环境指令从未闭合 {sum(gated)} / {n}")
