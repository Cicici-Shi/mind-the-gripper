"""CogACT 回放特征的汇总表。用法：python3 -I cogact_summary.py cogact_replay_feats.json > cogact_replay_tables.txt
全部是事后描述。分位数用 numpy 默认的线性插值。"""
import json, sys
import numpy as np
R = json.load(open(sys.argv[1]))
B = ["B1", "B2", "F1", "F2"]; C = ["C0", "C1", "C3", "C4", "C5"]
NAME = {"B1": "WidowX 茄子", "B2": "WidowX 勺子", "F1": "Google 可乐罐", "F2": "Google move near"}
def sel(b, c): return [r for r in R if r["block"] == b and r["cell"] == c]
def med(x):
    x = [v for v in x if v is not None]
    return "无" if not x else f"{np.median(x):.1f} [{np.percentile(x,25):.1f}, {np.percentile(x,75):.1f}]"
def medcm(x):
    x = [v * 100 for v in x if v is not None]
    return "无" if not x else f"{np.median(x):.1f} [{np.percentile(x,25):.1f}, {np.percentile(x,75):.1f}]"

print("== T0 可信性：每块每格 条数 / success_steps 逐位相同 / 初始状态与设置全部相同")
for b in B:
    print(b, "；".join(f"{c} {len(sel(b,c))}/{sum(r['match'] for r in sel(b,c))}/{sum(r['init_ok'] for r in sel(b,c))}" for c in C))
print(f"合计 {len(R)}，逐位相同 {sum(r['match'] for r in R)}，初始相同 {sum(r['init_ok'] for r in R)}")

print("\n== T1 阶段分布（键数，0 没接近 / 1 接近没碰 / 2 碰到或推动没夹住 / 3 夹住没搬动 / 4 搬动或抬起过最后没成 / 5 成功）")
for b in B:
    for c in C:
        s = sel(b, c); k = [sum(r["stage"] == i for r in s) for i in range(6)]
        print(f"{b} {c}  " + "/".join(map(str, k)) + f"   成功率 {100*np.mean([r['success'] for r in s]):.1f}")

print("\n== T2 闭合：出现闭合指令的回合数；首次闭合指令步 中位 [四分位]；物理闭合的回合数；闭合前已碰到任务物的回合数")
for b in B:
    for c in C:
        s = sel(b, c); cl = [r for r in s if r["t_close"] is not None]
        print(f"{b} {c}  指令 {len(cl)}/{len(s)}；t_c {med([r['t_close'] for r in cl])}；物理 {sum(r['t_phys'] is not None for r in s)}/{len(s)}；"
              f"闭合指令段数中位 {np.median([r['n_close_onsets'] for r in s]):.0f}；闭合前已碰到 {sum(r.get('contact_before_close', False) for r in cl)}/{len(cl)}")

print("\n== T3 闭合处末端相对任务物（厘米）：水平距离 中位 [四分位]；三维距离；水平距离 <= 5 cm 的回合数；闭合前任务物已被推动的距离")
for b in B:
    for c in C:
        cl = [r for r in sel(b, c) if r["t_close"] is not None]
        if not cl: print(f"{b} {c}  无闭合"); continue
        print(f"{b} {c}  水平 {medcm([r['dxy_close'] for r in cl])}；三维 {medcm([r['d_close'] for r in cl])}；<=5cm {sum(r['dxy_close']<=0.05 for r in cl)}/{len(cl)}；"
              f"此前推动 {medcm([r['src_moved_before_close'] for r in cl])}")

print("\n== T4 闭合处位移向量的散布（厘米，世界系 x、y、z）：格内均值；格内标准差；绕格均值的水平均方根半径")
for b in B:
    for c in C:
        cl = [r for r in sel(b, c) if r["t_close"] is not None]
        if len(cl) < 3: print(f"{b} {c}  闭合不足 3 回合"); continue
        O = np.array([r["off_close"] for r in cl]) * 100
        m = O.mean(0); sd = O.std(0, ddof=1); rr = np.sqrt(((O[:, :2] - m[:2]) ** 2).sum(1).mean())
        print(f"{b} {c}  n={len(cl)}  均值 ({m[0]:+.1f}, {m[1]:+.1f}, {m[2]:+.1f})  标准差 ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f})  水平均方根半径 {rr:.1f}")

print("\n== T5 同键配对：闭合处位移向量 某格 减 C0（两格都闭合的键；厘米）：差的均值；差的逐键标准差；键数")
for b in B:
    K0 = {r["key"]: r for r in sel(b, "C0") if r["t_close"] is not None}
    for c in ["C1", "C3", "C4", "C5"]:
        P = [(np.array(r["off_close"]) - np.array(K0[r["key"]]["off_close"])) * 100 for r in sel(b, c) if r["t_close"] is not None and r["key"] in K0]
        if len(P) < 3: print(f"{b} {c}-C0  配对不足"); continue
        P = np.array(P); m = P.mean(0); sd = P.std(0, ddof=1)
        print(f"{b} {c}-C0  n={len(P)}  均值 ({m[0]:+.1f}, {m[1]:+.1f}, {m[2]:+.1f})  标准差 ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f})  均值的标准误 ({sd[0]/np.sqrt(len(P)):.2f}, {sd[1]/np.sqrt(len(P)):.2f}, {sd[2]/np.sqrt(len(P)):.2f})")
    K4 = {r["key"]: r for r in sel(b, "C4") if r["t_close"] is not None}
    P = [(np.array(r["off_close"]) - np.array(K4[r["key"]]["off_close"])) * 100 for r in sel(b, "C5") if r["t_close"] is not None and r["key"] in K4]
    if len(P) >= 3:
        P = np.array(P); m = P.mean(0); sd = P.std(0, ddof=1)
        print(f"{b} C5-C4  n={len(P)}  均值 ({m[0]:+.1f}, {m[1]:+.1f}, {m[2]:+.1f})  标准差 ({sd[0]:.1f}, {sd[1]:.1f}, {sd[2]:.1f})")

print("\n== T6 全回合：末端离任务物最近距离（厘米）；碰到过 / 两指都碰到过 / 夹住过 的回合数；任务物最大抬升与最大水平位移（厘米）")
for b in B:
    for c in C:
        s = sel(b, c)
        print(f"{b} {c}  最近 {medcm([r['dmin'] for r in s])}；碰到 {sum(r['contact'] for r in s)}，两指 {sum(r['both_contact'] for r in s)}，夹住 {sum(r['grasp'] for r in s)}（共 {len(s)}）；"
              f"抬升 {medcm([r['lift_max'] for r in s])}；水平位移 {medcm([r['xy_max'] for r in s])}")

print("\n== T7 按成败分开看 C4 的闭合处水平距离（厘米）")
for b in B:
    cl = [r for r in sel(b, "C4") if r["t_close"] is not None]
    print(f"{b} C4  成功键 n={sum(r['success'] for r in cl)} 水平 {medcm([r['dxy_close'] for r in cl if r['success']])}；失败键 n={sum(not r['success'] for r in cl)} 水平 {medcm([r['dxy_close'] for r in cl if not r['success']])}")
