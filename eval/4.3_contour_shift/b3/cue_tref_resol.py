"""v3.1 冻结前的零卡时数（protocol_cue_widowx_v3_1_final.md §4 参考时刻、§14 待填项、§17 第 8、10 条）。登录节点跑。
  python cue_tref_resol.py tref --keyset B1=keysets/cue_B1.json --keyset B2=keysets/cue_B2.json \
         --c0-root $PROJ_ROOT/bench2_simpler/b2/out_formal --phase1-keyset-dir $PROJ_ROOT/bench2_simpler/b2/keysets --out tref_cue_widowx_v31.json
     t_ref(键) = 第一阶段同一块同一 episode 五个试次（正式键，试次 0..4）原图格首次闭合步的中位数；写到主键集的每个键上。
     五个试次里缺记录或缺闭合步就报错。只读 env_actions。
  python cue_tref_resol.py resol --sr-root $PROJ_ROOT/bench2_simpler/b2/out_formal --phase1-keyset-dir ... --out cue_resol_v31.json
     §14 成功率配对差的预计区间半宽（伪对，零卡时；只读第一阶段两块 C3、C4 正式记录的最后一步成败，那批已读过数）。
     每个 episode 取 t = 0..3 作伪键（96 个），试次号对 5 取模：
       D_SR、D_F：伪当前轮廓 = C4 试次 t；两个伪错位格 = 试次 t+1、t+2；零效应情形取 C4 成败，见底情形取 C3 成败；
                  逐伪键 = 两个伪错位格平均 - 伪当前轮廓。
       D_Fb、D_Fa：两个伪副本 = 试次 t+1、t+2，两个伪平移 = 试次 t+3、t+4；无差别情形四格都取 C4；一边见底情形伪平移取 C3；
                  逐伪键 = 伪副本平均 - 伪平移平均。
     按 episode 聚类 bootstrap（B = 10000，seed 0，95% 百分位），单位百分点。另附 v1 时的位置分辨力（k 的半宽）。
  python cue_tref_resol.py localq0 --root out_localq --out cue_localq0.json
     §5.6 / §17 第 10 条：localq 已有输出里只取 q = 0 的状态，按 localq_readout.py 的 D_a、D_b 同一口径
     （R = r(画面) - r(零)，水平分量，[R(+s) - R(-s)] / 2s，视野内 >= 0.8；键内平均，按 cfg_id 聚类 bootstrap 均值）。
     只读 localq 的输出，不跑任何查询。
"""
import argparse
import glob
import hashlib
import json
import os
import sys

import numpy as np

B = os.path.expanduser("$PROJ_ROOT/bench2_simpler")
TARGET = {"B1": "eggplant", "B2": "bridge_spoon_generated_modified"}


def first_close(ea):
    g = ea[:, -1]
    return int(np.nonzero(g < 0)[0][0]) if (g < 0).any() else None


def tref(a):
    ks = dict(x.split("=", 1) for x in a.keyset)
    assert set(ks) == {"B1", "B2"}, "--keyset 必须给 B1、B2 两块"
    out = {}
    for b in ("B1", "B2"):
        p1 = json.load(open(os.path.join(a.phase1_keyset_dir, f"formal_{b}.json")))
        by_ep = {}
        for k in p1:
            p = os.path.join(a.c0_root, b, "C0", k["key"] + ".npz")
            assert os.path.exists(p), f"缺第一阶段原图格记录 {p}"
            with np.load(p) as z:
                fc = first_close(z["env_actions"])
            assert fc is not None, f"{k['key']} 原图格没有闭合步"
            by_ep.setdefault(k["cfg_id"], []).append(fc)
        assert all(len(v) == 5 for v in by_ep.values()) and len(by_ep) == 24, "每个 episode 须恰有 5 个试次"
        med = {ep: int(np.median(v)) for ep, v in by_ep.items()}
        assert all(float(np.median(v)) == int(np.median(v)) for v in by_ep.values())  # 5 个整数的中位数是整数
        out[b] = {k["key"]: med[k["cfg_id"]] for k in json.load(open(ks[b]))}
        print(b, "keys", len(out[b]), "episode 取值范围", min(med.values()), max(med.values()),
              "极差中位数", float(np.median([max(v) - min(v) for v in by_ep.values()])))
    s = json.dumps(out, sort_keys=True, indent=1)
    open(a.out, "w").write(s)
    print("sha256", hashlib.sha256(s.encode()).hexdigest())


def cboot_half(vals, clusters, Bn=10000, seed=0):
    v = np.asarray(vals, float)
    cl = np.asarray(clusters)
    names = sorted(set(cl.tolist()))
    idx = [np.nonzero(cl == c)[0] for c in names]
    rng = np.random.default_rng(seed)
    st = np.empty(Bn)
    for i in range(Bn):
        pick = rng.integers(0, len(names), len(names))
        st[i] = v[np.concatenate([idx[j] for j in pick])].mean()
    lo, hi = np.percentile(st, [2.5, 97.5])
    return {"est": round(float(v.mean()), 3), "lo": round(float(lo), 3), "hi": round(float(hi), 3), "half": round(float(hi - lo) / 2, 3),
            "n": int(len(v))}


def resol(a):
    res = {}
    for b in ("B1", "B2"):
        sr = {}
        for c in ("C3", "C4"):
            for p in glob.glob(os.path.join(a.sr_root, b, c, "*.npz")):
                with np.load(p) as z:
                    sr[(c, str(z["cfg_id"]), int(z["trial"]))] = 100.0 * float(bool(z["success"]))
        eps = sorted({e for (_, e, _) in sr})
        rows = {"D_SR_null": [], "D_SR_floor": [], "D_Fx_null": [], "D_Fx_floor": []}
        cl = []
        for e in eps:
            for t in range(4):
                g = lambda c, tt: sr[(c, e, tt % 5)]
                cur = g("C4", t)
                rows["D_SR_null"].append((g("C4", t + 1) + g("C4", t + 2)) / 2 - cur)
                rows["D_SR_floor"].append((g("C3", t + 1) + g("C3", t + 2)) / 2 - cur)
                cp = (g("C4", t + 1) + g("C4", t + 2)) / 2
                rows["D_Fx_null"].append(cp - (g("C4", t + 3) + g("C4", t + 4)) / 2)
                rows["D_Fx_floor"].append(cp - (g("C3", t + 3) + g("C3", t + 4)) / 2)
                cl.append(e)
        res[b] = {k: cboot_half(v, cl) for k, v in rows.items()}
        res[b]["note"] = "D_SR 与 D_F 同一伪对算法（零效应 / 见底）；D_Fb 与 D_Fa 同一伪对算法（无差别 / 一边见底）"
        res[b]["reach_not_10pt_drop"] = res[b]["D_SR_null"]["half"] < 10
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res, indent=1, ensure_ascii=False))


def localq0(a):
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import localq_readout as LR
    res = {}
    for b in ("B1", "B2"):
        keys = json.load(open(os.path.join(a.phase1_keyset_dir, f"formal_{b}.json")))
        recs, missing, failed = LR.load(a.root, b, keys)
        assert not missing, missing[:3]
        per = {"D_a": [], "D_b": []}
        for r in recs:
            m0 = [m for m in r["meta"] if m["q"] == 0]
            assert len(m0) == 1
            v = LR.state_values(m0[0], 12, 24, 0.8)
            if v is None:
                continue
            for n in per:
                if n in v:
                    per[n].append((r["cluster"], v[n]))
        res[b] = {n: LR.boot([x[1] for x in kv], [x[0] for x in kv], 10000, 0) for n, kv in per.items()}
        res[b]["excluded"] = {k: len(v) for k, v in failed.items()}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["tref", "resol", "localq0"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--keyset", action="append", default=[])
    ap.add_argument("--c0-root")
    ap.add_argument("--sr-root")
    ap.add_argument("--root")
    ap.add_argument("--phase1-keyset-dir", required=True)
    a = ap.parse_args()
    if a.mode == "tref":
        assert a.c0_root and a.keyset, "tref 需要 --c0-root 与 --keyset（无默认值）"
        tref(a)
    elif a.mode == "resol":
        assert a.sr_root, "resol 需要 --sr-root"
        resol(a)
    else:
        assert a.root, "localq0 需要 --root"
        localq0(a)
