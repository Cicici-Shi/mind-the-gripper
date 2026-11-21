"""由 cue_params.json 生成冒烟的格清单与对照图面板。
主口径（草案 v0 加反方与协调方 11-13 的增补）：T 张开、T 闭合、X 瓦片重排（每次查询重排 / 每回合固定两种种子）、
配准幅度 a、b 的水平偏移 ±、偏移副本（b 与半个夹爪宽）±、按夹爪可见宽度比例的偏移（1/4、1/2、1 倍）。
  python cue_smoke_specs.py cue_params.json B2 8 cells   -> 每行 "标签|格|var JSON"（策略冒烟用，每种画法各一回合）
  python cue_smoke_specs.py cue_params.json F1 4 sheet   -> 对照图面板 JSON（F1 另加 12 步滞后的 b±）"""
import json
import sys

prm = json.load(open(sys.argv[1]))
b, L, what = sys.argv[2], int(sys.argv[3]), sys.argv[4]
p = prm[b]
sh, cp = p["shifts"], p["copies"]


def S(m, sg):
    return {"mode": "axis", "axis": [1, 0], "mag": abs(m), "sign": sg}


a_, b_ = sh[f"S_L{L}_a+"][0], sh[f"S_L{L}_b+"][0]
panel = [("T open", "T", {"tpl": p["q_open"]}), ("T closed", "T", {"tpl": p["q_closed"]}),
         ("X per-query", "X", {"seed_mode": "query"}), ("X per-episode", "X", {"seed_mode": "episode"}),
         (f"Sa+ {a_}px", "S", S(a_, 1)), (f"Sa- {a_}px", "S", S(a_, -1)), (f"Sb+ {b_}px", "S", S(b_, 1)), (f"Sb- {b_}px", "S", S(b_, -1)),
         (f"copy b+ {b_}px", "SC", S(b_, 1)), (f"copy w/2+ {cp['SC_w12+'][0]}px", "SC", S(cp["SC_w12+"][0], 1))]
for tag in ("w14", "w12", "w1"):
    m = sh[f"W_{tag}+"][0]
    panel.append((f"W{tag[1:]}+ {m}px", "S", S(m, 1)))
if b == "F1" and "S_L12_b+" in sh:
    m12 = sh["S_L12_b+"][0]
    panel += [(f"Sb12+ {m12}px", "S", S(m12, 1)), (f"Sb12- {m12}px", "S", S(m12, -1))]
if what == "cells":
    # 策略冒烟：每种画法各一回合
    run = [("T_open", panel[0]), ("T_closed", panel[1]), ("X_query", panel[2]), ("X_episode", panel[3]), ("Sb_p", panel[6]),
           ("SC_b_p", panel[8]), ("W12_p", [x for x in panel if x[0].startswith("W12+")][0])]
    for t, (_, c, v) in run:
        print(f"{t}|{c}|{json.dumps(v)}")
else:
    sp = [["C0 orig", "C0", {}], ["C3 erase", "C3", {}], ["C4 outline", "C4", {}]] + [list(x) for x in panel]
    json.dump(sp, sys.stdout, indent=1)
