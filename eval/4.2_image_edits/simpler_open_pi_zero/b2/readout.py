"""bench2 一次性读数（协议 protocol_bench2.md v1.1 §5）。只在协调方冻结协议后对真实数据运行；执行方只在合成数据上跑。

输入：--root 采集根目录（<root>/<块>/<格>/<键>.npz），--block，--keys 键清单，
      块级标记（blacken_weak 涂黑操纵不足、lag_weak 滞后操纵不足）不从命令行取，直接读冻结清单 blocks.<块>，缺字段就拒读。
键清单必须是冻结的那份：--keys 文件的 sha256 等于清单 keylist_sha256.<块>、该块键数等于 blocks.<块>.n_keys，否则拒读，
      不写任何输出（防止把一次性读数用在切片或改过的清单上）。合成数据同样核对（合成清单里写合成键清单的哈希）。
门槛不读外部文件：本脚本对同一个根目录、同一份键清单直接调 gate.run 现算（冻结清单 = FROZEN_bench2；
      合成数据用 <root>/SYNTHETIC_MANIFEST.json），防止采集中途生成的旧 gate.json 把后来补齐的键静默丢掉。
输出固定在 <root>/READOUT_bench2_<块>.txt（文字）与 .json（全部数字）；任一已存在就拒读。
拒读：块不完整（某格某键既没有记录也没有报错文件，或门槛保留的键缺成败字段）；输出已存在；
      根目录没有 SYNTHETIC_DATA 标记时，$PROJ_ROOT/bench2_simpler/b2/FROZEN_bench2 必须存在（防执行方误跑真实数据）。
主读数 = 最后一步 success；另出「任一步成功过」的并列读数（不进判定）。
F2 另出按摆放布局（episode_id // 6，10 簇）聚类的并列区间，敏感性参考，不进判定。逐块输出，不出计票总句。
"""
import argparse
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate as G  # noqa: E402

# ---------------------------------------------------------------- 判定线（v1 §5），全部在这里
CELLS = ("C0", "C1", "C3", "C4", "C5")
B_BOOT = 10000
BOOT_SEED = 0
CI = (2.5, 97.5)
ADMIT_SR_C0 = 60.0          # 准入：SR(C0) >= 60
S1_POINT_MAX = -15.0        # S1：D3 点估计 <= -15 且上界 < 0
S1_HI_MAX = 0.0
S2_LO_MIN = -10.0           # S2：D1 下界 >= -10
S3_POINT_MIN = 15.0         # S3：Δ7 点估计 >= 15 且下界 > 0
S3_LO_MIN = 0.0
B_D1_POINT_MAX = -15.0      # 标签 B：S1 且 D1 点估计 <= -15、上界 < 0
B_D1_HI_MAX = 0.0
C_D3_LO_MIN = -10.0         # 标签 C：D3 下界 >= -10
P_POINT_MIN = 10.0          # P1 / P2「提高」：点估计 >= 10 且下界 > 0
P_LO_MIN = 0.0
P_NOT_HI_MAX = 10.0         # 「未提高」：上界 < 10
EPS = 1e-9                  # 浮点比较容差（成功率以百分点计，来自整数计数）
FROZEN = G.FROZEN
# F2（move near v0）的摆放布局：episode_id // 6。环境代码 move_near_in_scene.py：triplet = ep // 12（543-545 行），
# 源 / 目标物体对 = ep % 6（546-547 行，6 个有序对，621-626 行），xy 布局 = (ep % 12) // 6（548-554 行，2 种，627-630 行）；
# 所以 (triplet, xy) = ep // 6，60 个 episode = 10 个布局 x 6 种指令。
F2_LAYOUT_DIV = 6
CONTRASTS = {"D1": ("C1", "C0"), "D3": ("C3", "C0"), "Δ7": ("C1", "C3"), "Δ1": ("C4", "C3"), "Δ2": ("C4", "C5")}
TEXT = {
    "A": "与 LIBERO 相同：机器人夹爪从画面里擦掉后掉分，只涂黑几乎不掉",
    "A′": "擦掉后掉分；涂黑在这个身体上近乎空操作，分不出来",
    "B": "擦掉和涂黑都掉分；此处遮住夹爪像素本身就伤（再报 Δ7）",
    "C": "在给了本体状态的前提下，擦掉夹爪也几乎不掉分",
    "D": "照报数字，不下结论",
    "基线不足": "基线不足，不判",
}


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
def load_block(root, block, keys, gate, field):
    """返回 (succ[cell] -> np.array over keys, clusters list, kept keys)。field = success 或 success_ever。"""
    excl = set(gate.get("excluded_keys", []))
    kept = [k for k in keys if k["key"] not in excl]
    if not kept:
        raise Refuse(f"{block}：门槛后没有保留的键")
    succ = {}
    for c in CELLS:
        v = []
        for k in kept:
            p = os.path.join(root, block, c, k["key"] + ".npz")
            if not os.path.exists(p):
                raise Refuse(f"{block}：块不完整，缺 {c}/{k['key']}")
            with np.load(p, allow_pickle=False) as z:
                if field not in z.files or not bool(z["complete"]):
                    raise Refuse(f"{block}：{c}/{k['key']} 缺 {field} 或不完整")
                v.append(bool(z[field]))
        succ[c] = np.array(v, np.float64)
    return succ, [k["cluster"] for k in kept], kept


def check_collected(root, block, keys):
    """采集是否完成：每格每键要么有记录，要么有报错文件（_errors/<键>.txt，重试后仍失败）。
    两者都没有 = 还没采完，拒读（不让门槛把缺的键当成剔除静默丢掉）。"""
    for c in CELLS:
        for k in keys:
            if not os.path.exists(os.path.join(root, block, c, k["key"] + ".npz")) and \
                    not os.path.exists(os.path.join(root, block, c, "_errors", k["key"] + ".txt")):
                raise Refuse(f"{block}：块不完整，{c}/{k['key']} 既没有记录也没有报错文件（采集未完成）")


# ---------------------------------------------------------------- 估计
def estimate(succ, clusters, B=B_BOOT, seed=BOOT_SEED):
    """成功率（百分点）、配对差点估计与聚类 bootstrap 百分位区间。clusters 每键一个标签（Fractal 每键自成一簇）。"""
    labs, inv = np.unique(np.asarray(clusters), return_inverse=True)
    nC = len(labs)
    cnt = np.bincount(inv, minlength=nC).astype(np.float64)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, nC, size=(B, nC))
    w = np.zeros((B, nC))
    np.add.at(w, (np.repeat(np.arange(B), nC), idx.ravel()), 1.0)  # 每次重采样里每簇被抽中的次数
    out = {"n_keys": int(len(inv)), "n_clusters": int(nC), "SR": {c: 100.0 * float(succ[c].mean()) for c in CELLS}}
    for name, (a, b) in CONTRASTS.items():
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


def decide(est, flags):
    """flags：blacken_weak（涂黑操纵不足）、lag_weak（滞后操纵不足）、outline_weak_C4、outline_weak_C5（门槛 6）。"""
    r = {"flags": {k: bool(v) for k, v in flags.items()}}
    if not ge(est["SR"]["C0"], ADMIT_SR_C0):
        r.update(admitted=False, label="基线不足", text=TEXT["基线不足"], S1=None, S2=None, S3=None, P1="不判", P2="不判")
        return r
    r["admitted"] = True
    D1, D3, D7, P1e, P2e = est["D1"], est["D3"], est["Δ7"], est["Δ1"], est["Δ2"]
    S1 = le(D3["point"], S1_POINT_MAX) and lt(D3["hi"], S1_HI_MAX)
    weak = bool(flags.get("blacken_weak"))
    S2 = None if weak else ge(D1["lo"], S2_LO_MIN)
    S3 = None if weak else (ge(D7["point"], S3_POINT_MIN) and gt(D7["lo"], S3_LO_MIN))
    Bc = None if weak else (S1 and le(D1["point"], B_D1_POINT_MAX) and lt(D1["hi"], B_D1_HI_MAX))
    Cc = ge(D3["lo"], C_D3_LO_MIN)
    if weak:
        label = "A′" if S1 else ("C" if Cc else "D")
    elif S1 and S2 and S3:
        label = "A"
    elif Bc:
        label = "B"
    elif Cc:
        label = "C"
    else:
        label = "D"
    if S1:
        # 协议 v1.1 §4 第 6 项：按估计量涉及的格。P1 = SR(C4) - SR(C3)，只看 C4；P2 = SR(C4) - SR(C5)，看 C4 与 C5
        p1 = "不判（C4 轮廓操纵不足）" if flags.get("outline_weak_C4") else p_level(P1e)
        if flags.get("outline_weak_C4") or flags.get("outline_weak_C5"):
            p2 = "不判（轮廓操纵不足）"
        elif flags.get("lag_weak"):
            p2 = "不判（滞后操纵不足）"
        else:
            p2 = p_level(P2e)
    else:
        p1 = p2 = "不判（S1 不成立）"
    r.update(S1=S1, S2=S2, S3=S3, B_cond=Bc, C_cond=Cc, label=label, text=TEXT[label], P1=p1, P2=p2)
    return r


def fmt(block, est, dec, tag):
    s = [f"[{block}] {tag}：键 {est['n_keys']}（{est['n_clusters']} 簇）；SR " +
         ", ".join(f"{c}={est['SR'][c]:.1f}" for c in CELLS)]
    for n in CONTRASTS:
        e = est[n]
        s.append(f"  {n} = {e['point']:+.1f} [{e['lo']:+.1f}, {e['hi']:+.1f}]")
    if dec is not None:
        s.append(f"  标签 {dec['label']}：{dec['text']}；S1={dec['S1']} S2={dec['S2']} S3={dec['S3']}；"
                 f"P1={dec['P1']}；P2={dec['P2']}；标记 {dec['flags']}")
    return "\n".join(s)


def flags_for(block, gate, man):
    """块级标记只从冻结清单取（blocks.<块>.blacken_weak / lag_weak），格标记来自现算的门槛。"""
    cf = man.get("blocks", {}).get(block)
    if cf is None or "blacken_weak" not in cf or "lag_weak" not in cf:
        raise Refuse(f"冻结清单 blocks.{block} 缺 blacken_weak / lag_weak")
    cell = gate.get("cell_flags", {})
    return {"blacken_weak": bool(cf["blacken_weak"]), "lag_weak": bool(cf["lag_weak"]),
            "outline_weak_C4": G.OUTLINE_WEAK in cell.get("C4", []),
            "outline_weak_C5": G.OUTLINE_WEAK in cell.get("C5", [])}


def layout_clusters(block, kept):
    """F2 的敏感性参考：按摆放布局聚类（episode_id // 6）。"""
    return [f"layout{int(k['reset']['episode_id']) // F2_LAYOUT_DIV}" for k in kept]


def readout(root, block, keys, gate, man):
    f = flags_for(block, gate, man)
    res = {"block": block, "excluded_keys": gate.get("excluded_keys", [])}
    succ, cl, kept = load_block(root, block, keys, gate, "success")
    est = estimate(succ, cl)
    res["final"] = {"estimate": est, "decision": decide(est, f)}
    succ2, cl2, _ = load_block(root, block, keys, gate, "success_ever")
    est2 = estimate(succ2, cl2)
    res["ever"] = {"estimate": est2, "note": "并列读数，不进判定"}
    res["text"] = fmt(block, est, res["final"]["decision"], "主读数（最后一步）") + "\n" + \
        fmt(block, est2, None, "并列：任一步成功过")
    if block == "F2":
        est3 = estimate(succ, layout_clusters(block, kept))
        res["layout_cluster_sensitivity"] = {"estimate": est3, "note": "按摆放布局（episode_id // 6）聚类，敏感性参考，不进判定"}
        res["text"] += "\n" + fmt(block, est3, None, "并列：按摆放布局聚类（敏感性参考，不进判定）")
    return res


def out_paths(root, block):
    b = os.path.join(root, f"READOUT_bench2_{block}")
    return b + ".txt", b + ".json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True, help="冻结的正式键清单（keysets/formal_<块>.json），哈希须与清单一致")
    a = ap.parse_args(argv)
    txt, js = out_paths(a.root, a.block)
    if os.path.exists(txt) or os.path.exists(js):
        raise Refuse(f"{txt} 已存在，拒绝第二次读数")
    synthetic = os.path.exists(os.path.join(a.root, "SYNTHETIC_DATA"))
    if synthetic:
        mp = os.path.join(a.root, "SYNTHETIC_MANIFEST.json")
        if not os.path.exists(mp):
            raise Refuse("合成数据缺 SYNTHETIC_MANIFEST.json")
    else:
        if not os.path.exists(FROZEN):
            raise Refuse("真实数据只能在协议冻结（FROZEN_bench2 存在）后读")
        mp = FROZEN
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
    flags_for(a.block, {}, man)  # 标记字段缺失时在动任何数据之前就拒读
    check_collected(a.root, a.block, keys)
    gate = G.run(a.root, a.block, keys, manifest=man)  # 现算，不读外部 gate.json
    res = readout(a.root, a.block, keys, gate, man)
    res["gate_summary"] = G.summary(gate)
    res["manifest_path"] = mp
    fd = os.open(txt, os.O_WRONLY | os.O_CREAT | os.O_EXCL)  # 并发下也只允许写一次
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
