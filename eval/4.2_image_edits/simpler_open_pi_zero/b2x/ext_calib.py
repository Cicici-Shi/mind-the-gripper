"""补充协议 v1 的校准量（protocol_bench2_ext_fill_lag.md §2、§3、§4、§7），只读校准键的原图格（C0）记录。

输入：--root out_calib_ext（b2x 用 --color-stats 重跑的校准键 C0；F1 另带 --geom-lags），
      --ref-root ../b2/out_calib（第一阶段校准记录，只读；用于逐位比对，以及 F1 原图格的逐步 success 求首次成功步数，
      校准键的 C0 成败按第一阶段协议可以看）。新记录本身盲成败，这里不读它们的任何成败字段。
输出 --out calib_ext.json：
  * FL：Google Robot（F1）校准键原图格全部查询帧 G 像素的逐通道中位数，再换成离它最近（RGB 欧氏距离）的一个
    真实出现过的 G 像素颜色；并列给出逐通道直方图、各通道的峰数（多峰核对）、G 像素落在 FL 附近的比例；
  * 操纵强度：对每个被盖像素取三通道 |原值 - 填充值| 的平均，再在 G 内按像素平均；三种填充（黑、白、FL；v1.1）x 两块；F1 另按 urdf 版本；
  * 环反差：G 外 6 像素一圈的像素对填充色的同一量；低于同块黑色的一半标「低反差」；
  * 滞后几何量（F1，候选 4、8、12 步）：按策略实际看到的轮廓（含未满滞后的帧），分 pre / full / all 三段的
    IoU、质心位移（224 像素）、可见像素比的中位数；按规则取更长滞后：候选（8、12）中 all 段 IoU 中位数 <= 0.08
    的最小值，都不满足取 12 并注明；
  * 开头轮廓不动的那一段：l + 查询间隔 步（t <= l 的各次查询都显示第 0 步的轮廓），占原图格首次成功之前步数的比例；
  * 重跑核对：raw_actions、env_actions 与第一阶段校准记录逐位相同的键数；第 0 帧与第一阶段存的 png 逐字节相同的键数。
"""
import argparse
import collections
import glob
import hashlib
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys as K  # noqa: E402

FILL_WHITE = [255, 255, 255]  # 补充协议 v1.1：Fw 白
LAG_CANDIDATES = (8, 12)
LAG_IOU_MAX = 0.08
LAG_FALLBACK = 12
READ_FIELDS = ("key", "cell", "qstats", "raw_actions", "env_actions", "frame0_sha256", "color_stats", "G_colors",
               "ring_colors", "init_urdf_version", "query_every", "task")


def load(p):
    with np.load(p, allow_pickle=False) as z:
        return {f: z[f] for f in READ_FIELDS if f in z.files}


def merge(arrs):
    d = collections.Counter()
    for a in arrs:
        for r, g, b, n in a.tolist():
            d[(r, g, b)] += n
    if not d:
        return np.zeros((0, 4), np.int64)
    return np.array([[*k, v] for k, v in d.items()], np.int64)


def channel_median(cc):
    out = []
    for ch in range(3):
        h = np.bincount(cc[:, ch], weights=cc[:, 3], minlength=256)
        n = h.sum()
        c = np.cumsum(h)
        lo = int(np.searchsorted(c, (n - 1) // 2 + 1))
        hi = int(np.searchsorted(c, n // 2 + 1))
        out.append((lo + hi) / 2.0)
    return out


def channel_hist(cc):
    return [np.bincount(cc[:, ch], weights=cc[:, 3], minlength=256).astype(int).tolist() for ch in range(3)]


def n_peaks(h, frac=0.05, smooth=5):
    """平滑后的直方图里高于最高峰 frac 的局部极大个数（粗略的多峰核对）。"""
    h = np.convolve(np.asarray(h, float), np.ones(smooth) / smooth, mode="same")
    m = h.max()
    return int(sum(1 for i in range(1, 255) if h[i] >= frac * m and h[i] > h[i - 1] and h[i] >= h[i + 1]))


def nearest_real(cc, target):
    """离 target 最近（RGB 欧氏距离）的真实颜色；距离相同取像素数多的，再按 RGB 字典序。"""
    d = ((cc[:, :3] - np.asarray(target, float)) ** 2).sum(1)
    order = np.lexsort((cc[:, 2], cc[:, 1], cc[:, 0], -cc[:, 3], d))
    return [int(x) for x in cc[order[0], :3]], float(np.sqrt(d[order[0]]))


def weighted_absdiff(cc, rgb):
    """mean over 像素 of mean_ch |px - rgb|（操纵强度 / 环反差）。"""
    if len(cc) == 0:
        return None
    per = np.abs(cc[:, :3] - np.asarray(rgb)).mean(1)
    return float((per * cc[:, 3]).sum() / cc[:, 3].sum())


def first_success_step(succ_steps):
    s = np.asarray(succ_steps, bool)
    return int(np.argmax(s)) + 1 if s.any() else None  # 第几步（从 1 数）首次成功


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--ref-root", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    recs = {}
    for b in ("B2", "F1"):
        allowed = {k["key"] for k in K.make_keys(b, "calib")}
        recs[b] = {}
        for p in sorted(glob.glob(os.path.join(a.root, b, "*", "*.npz"))):
            k = os.path.basename(p)[:-4]
            if k not in allowed or os.path.basename(os.path.dirname(p)) != "C0":
                sys.exit(f"拒绝运行：{p} 不是校准键的 C0 记录")
            r = load(p)
            assert bool(r["color_stats"]), p
            recs[b][k] = r
    res = {"blocks": {}}
    # ---- FL：取自 Google Robot（F1）
    g_f1 = merge(r["G_colors"] for r in recs["F1"].values())
    med = channel_median(g_f1)
    fl, dist = nearest_real(g_f1, med)
    hist = channel_hist(g_f1)
    near = g_f1[np.sqrt(((g_f1[:, :3] - np.array(fl)) ** 2).sum(1)) <= 20]
    res["FL"] = {"rgb": fl, "channel_median": med, "nearest_distance": dist, "n_G_pixels": int(g_f1[:, 3].sum()),
                 "n_distinct_colors": int(len(g_f1)), "channel_hist": hist,
                 "channel_peaks": [n_peaks(h) for h in hist],
                 "mean_rgb": [float((g_f1[:, ch] * g_f1[:, 3]).sum() / g_f1[:, 3].sum()) for ch in range(3)],
                 "frac_G_within_20_of_FL": float(near[:, 3].sum() / g_f1[:, 3].sum())}
    fills = {"k": [0, 0, 0], "w": FILL_WHITE, "L": fl}
    res["fills"] = fills
    for b, rs in recs.items():
        G_all = merge(r["G_colors"] for r in rs.values())
        R_all = merge(r["ring_colors"] for r in rs.values())
        d = {"n_keys": len(rs), "n_G_pixels": int(G_all[:, 3].sum()), "n_ring_pixels": int(R_all[:, 3].sum()),
             "G_mean_rgb": [float((G_all[:, ch] * G_all[:, 3]).sum() / G_all[:, 3].sum()) for ch in range(3)],
             "fill_strength": {x: weighted_absdiff(G_all, c) for x, c in fills.items()},
             "ring_contrast": {x: weighted_absdiff(R_all, c) for x, c in fills.items()}}
        d["low_contrast"] = {x: bool(d["ring_contrast"][x] < 0.5 * d["ring_contrast"]["k"]) for x in fills}
        by_urdf = collections.defaultdict(list)
        for r in rs.values():
            by_urdf[str(r["init_urdf_version"])].append(r)
        if len(by_urdf) > 1:
            d["fill_strength_by_urdf"] = {}
            for u, lst in sorted(by_urdf.items()):
                Gu = merge(r["G_colors"] for r in lst)
                d["fill_strength_by_urdf"][u or "(默认)"] = {"n_keys": len(lst), **{x: weighted_absdiff(Gu, c) for x, c in fills.items()}}
        # ---- 滞后几何量（只有带 geom 的块）
        geo = collections.defaultdict(lambda: collections.defaultdict(lambda: collections.defaultdict(list)))
        for r in rs.values():
            for q in json.loads(str(r["qstats"])):
                for l, g in q.get("geom", {}).items():
                    for ph in (g["phase"], "all"):
                        for f in ("iou", "centroid_shift_224", "visible_ratio_224"):
                            if g.get(f) is not None:
                                geo[l][ph][f].append(g[f])
        if geo:
            d["geometry"] = {l: {ph: {**{f + "_median": float(np.median(v)) for f, v in fs.items()},
                                      "frames": len(fs["iou"])} for ph, fs in phs.items()}
                             for l, phs in sorted(geo.items(), key=lambda t: int(t[0]))}
            ok = [l for l in LAG_CANDIDATES if str(l) in d["geometry"]
                  and d["geometry"][str(l)]["all"]["iou_median"] <= LAG_IOU_MAX]
            d["lag_long_rule"] = {"candidates": list(LAG_CANDIDATES), "iou_max": LAG_IOU_MAX,
                                  "all_iou_median": {str(l): d["geometry"][str(l)]["all"]["iou_median"]
                                                     for l in LAG_CANDIDATES if str(l) in d["geometry"]},
                                  "chosen": min(ok) if ok else LAG_FALLBACK,
                                  "note": "满足规则" if ok else "两个候选都不满足，取 12 步：剂量仍高于 WidowX"}
            # 开头轮廓不动的那一段 / 首次成功步数（第一阶段校准记录的逐步 success，校准键 C0 可看）
            qe = int(next(iter(rs.values()))["query_every"])
            fs_steps = []
            for k in rs:
                rp = os.path.join(a.ref_root, b, "C0", k + ".npz")
                if os.path.exists(rp):
                    with np.load(rp, allow_pickle=False) as z:
                        if "success_steps" in z.files:
                            f = first_success_step(z["success_steps"])
                            if f is not None:
                                fs_steps.append(f)
            med_fs = float(np.median(fs_steps)) if fs_steps else None
            d["frozen_segment"] = {str(l): {"frozen_steps": l + qe,
                                            "ratio_to_median_first_success": None if not med_fs else (l + qe) / med_fs,
                                            "per_key_ratio_median": None if not fs_steps else float(np.median([(l + qe) / s for s in fs_steps]))}
                                   for l in sorted(int(x) for x in d["geometry"])}
            d["first_success_steps"] = {"n_success_keys": len(fs_steps), "median": med_fs, "values": sorted(fs_steps)}
        # ---- 重跑核对
        same_act = same_f0 = n_ref = 0
        diff = []
        import imageio.v2 as iio
        for k, r in rs.items():
            rp = os.path.join(a.ref_root, b, "C0", k + ".npz")
            png = os.path.join(a.ref_root, b, "C0", "frames", k + "_q000_full.png")
            if not os.path.exists(rp):
                continue
            n_ref += 1
            with np.load(rp, allow_pickle=False) as z:
                ok = all(z[f].shape == r[f].shape and np.array_equal(z[f], r[f]) for f in ("raw_actions", "env_actions"))
            same_act += ok
            f0 = None
            if os.path.exists(png):
                f0 = hashlib.sha256(np.ascontiguousarray(iio.imread(png)).tobytes()).hexdigest() == str(r["frame0_sha256"])
                same_f0 += bool(f0)
            if not ok or f0 is False:
                diff.append(k)
        d["rerun_vs_phase1_calib"] = {"n_ref": n_ref, "actions_bit_equal": same_act, "frame0_equal": same_f0, "differ": diff}
        res["blocks"][b] = d
    with open(a.out + ".tmp", "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    os.replace(a.out + ".tmp", a.out)
    short = {k: v for k, v in res.items() if k != "FL"}
    print(json.dumps({"FL": {k: v for k, v in res["FL"].items() if k != "channel_hist"}, **short}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
