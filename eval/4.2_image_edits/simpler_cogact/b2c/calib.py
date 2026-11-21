"""bench2 第二阶段校准汇总（只读校准键；成败只读 C0，第一阶段协议 v1.3 §3）。由第一阶段 b2/calib.py 改写。

  python calib.py --root out_calib --aa-root out_calib_aa --blocks B1 B2 F1 F2 --out calib.json --flags calib_flags.json
输出：
  * 每块 C0 校准键成功数（最后一步 / 任一步）；
  * A/A（--aa-root 里同键 C0 第二遍）：env_actions、raw_actions（集成后）、pred_chunks（每次查询的 16 x 7 扩散输出）
    是否逐位相同、最大差、成败翻转数（补充协议 §3 第 5 条：动作逐位相同是冻结前置条件）；
  * 滞后：L 用第一阶段的步数（补充协议 §3 第 3 条），按第一阶段规则核查：当前与 L 步前轮廓线 IoU 中位数，
    > 0.7 看 2L 的中位数，仍 > 0.7 标「滞后操纵不足」。CogACT 每步查询，所以 IoU 在每个控制步上算；
  * 涂黑操纵强度：C0 回合 G 掩膜内像素三通道平均的均值（按像素加权），< 30 标「涂黑操纵不足」；
  * 线在 CogACT 224 输入里的名义有效宽度（线宽 x 224 / 图宽、x 224 / 图高）；每回合秒数。
第一阶段的本体状态探针对 CogACT 不适用（补充协议 §2），删去。
只许读校准键：根目录（含 A/A 根目录）里出现任何非校准键就拒绝运行，不读任何成败。
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import keys as K  # noqa: E402

IOU_MAX = 0.7
BLACKEN_MIN = 30.0


def check_calib_only(roots, blocks):
    for b in blocks:
        allowed = {k["key"] for k in K.make_keys(b, "calib")}
        formal = {k["key"] for k in K.make_keys(b, "formal")}
        assert not (allowed & formal), b
        for root in roots:
            for p in glob.glob(os.path.join(root, b, "*", "*.npz")):
                k = os.path.basename(p)[:-4]
                if k not in allowed:
                    sys.exit(f"拒绝运行：{p} 不是校准键（calib.py 只读校准键的 C0 成败）")


def recs(root, block, cell):
    out = {}
    for p in sorted(glob.glob(os.path.join(root, block, cell, "*.npz"))):
        z = np.load(p, allow_pickle=False)
        out[str(z["key"])] = z
    return out


def same(a, b):
    return a.shape == b.shape and a.dtype == b.dtype and bool(np.array_equal(a, b))


def maxdiff(a, b):
    if a.shape != b.shape:
        return float("inf")
    return float(np.abs(a.astype(np.float64) - b.astype(np.float64)).max()) if a.size else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--aa-root", default=None)
    ap.add_argument("--blocks", nargs="+", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--flags", required=True)
    a = ap.parse_args()
    check_calib_only([a.root] + ([a.aa_root] if a.aa_root else []), a.blocks)
    res, flags = {}, {}
    for b in a.blocks:
        r0 = recs(a.root, b, "C0")
        if not r0:
            continue
        d = {"n_keys": len(r0)}
        d["success_final"] = int(sum(bool(z["success"]) for z in r0.values()))
        d["success_ever"] = int(sum(bool(z["success_ever"]) for z in r0.values()))
        z0 = next(iter(r0.values()))
        L, w = int(z0["lag_steps"]), int(z0["line_width"])
        H, W = (int(x) for x in z0["img_hw"])
        d.update(L=L, query_every=int(z0["query_every"]), control_freq=float(z0["control_freq"]),
                 sim_freq=float(z0["sim_freq"]), max_episode_steps=int(z0["max_episode_steps"]), line_width=w,
                 img_hw=[H, W], line_eff_width_224_horiz=w * 224 / W, line_eff_width_224_vert=w * 224 / H,
                 g_links=json.loads(str(z0["g_links"])), robot_uid=str(z0["robot_uid"]),
                 policy_cfg=json.loads(str(z0["policy_cfg"])),
                 sec_per_episode_median=float(np.median([float(z["sec"]) for z in r0.values()])),
                 sec_per_episode_mean=float(np.mean([float(z["sec"]) for z in r0.values()])),
                 steps_per_episode=sorted({int(z["n_queries"]) for z in r0.values()}))
        ious, ious2, gsum, gpx, protect = [], [], 0.0, 0, set()
        for z in r0.values():
            protect |= set(json.loads(str(z["p_names"])))
            for q in json.loads(str(z["qstats"])):
                if q.get("iou_lag_outline") is not None:
                    ious.append(q["iou_lag_outline"])
                if q.get("iou_lag2_outline") is not None:
                    ious2.append(q["iou_lag2_outline"])
                gsum += q["G_rgb_sum"]
                gpx += q["G_px"]
        m1 = float(np.median(ious)) if ious else None
        m2 = float(np.median(ious2)) if ious2 else None
        if m1 is not None and m1 > IOU_MAX:
            L_final = 2 * L
            lag_weak = m2 is None or m2 > IOU_MAX
        else:
            L_final, lag_weak = L, False
        blacken = gsum / gpx if gpx else None
        d.update(iou_lag_outline_median=m1, iou_lag2_outline_median=m2, iou_frames=len(ious), L_final=L_final,
                 lag_weak=lag_weak, blacken_strength=blacken, blacken_weak=bool(blacken is not None and blacken < BLACKEN_MIN),
                 protect_names_seen=sorted(protect))
        if a.aa_root:
            r1 = recs(a.aa_root, b, "C0")
            common = sorted(set(r0) & set(r1))
            if common:
                eq = {f: int(sum(same(r0[k][f], r1[k][f]) for k in common))
                      for f in ("env_actions", "raw_actions", "pred_chunks", "seeds", "frame0_sha256")}
                mx = {f: max(maxdiff(r0[k][f], r1[k][f]) for k in common)
                      for f in ("env_actions", "raw_actions", "pred_chunks")}
                flips = [k for k in common if bool(r0[k]["success"]) != bool(r1[k]["success"])]
                d["aa"] = {"n": len(common), "bit_equal": eq, "max_abs_diff": mx,
                           "all_actions_bit_equal": all(eq[f] == len(common) for f in ("env_actions", "raw_actions", "pred_chunks")),
                           "success_flips": len(flips), "flipped_keys": flips,
                           "init_state_equal": int(sum(np.array_equal(r0[k]["init_actor_poses"], r1[k]["init_actor_poses"])
                                                       and np.array_equal(r0[k]["init_robot_qpos"], r1[k]["init_robot_qpos"])
                                                       for k in common)),
                           "hosts": sorted({str(r0[k]["host"]) + ">" + str(r1[k]["host"]) for k in common})}
        res[b] = d
        flags[b] = {"blacken_weak": d["blacken_weak"], "lag_weak": lag_weak, "L": L_final}
    for p, o in ((a.out, res), (a.flags, flags)):
        with open(p, "w") as f:
            json.dump(o, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
