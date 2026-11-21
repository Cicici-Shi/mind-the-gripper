"""bench2 校准汇总（只读校准键；成败只读 C0，协议 v1 §3）。

  python calib.py --root out_calib --aa-root out_calib_aa2 --blocks B1 B2 F1 F2 --out calib.json --flags flags.json
输出：
  * 每块 C0 校准键成功数（最后一步 / 任一步）；
  * A/A（--aa-root 里同键 C0 第二遍）：动作逐位相同的键数、原始动作最大差、成败翻转数；
  * 滞后：当前与 L 步前轮廓线 IoU 中位数（v1 判据），> 0.7 看 2L 的中位数，仍 > 0.7 标「滞后操纵不足」；
  * 涂黑操纵强度：C0 回合 G 掩膜内像素三通道平均的均值（涂黑后为 0，所以就是差），< 30 标「涂黑操纵不足」；
  * 本体状态探针汇总；线在 224 输入里的横向、纵向有效宽度（线宽 x 224 / 图宽、x 224 / 图高）。
flags.json 给 readout.py 用：{块: {blacken_weak, lag_weak, L}}。
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


# 协议 v1.1 改键之前 F2 的校准键（旧的种子键，seed 900000..900039；作业 <job> 的 out_calib/F2 用的就是它们）。
# 与 v1.1 的正式键（F2__epXX_uY__t0 / t1）名字不相交，只用来认这批已有的校准记录。
LEGACY_CALIB = {"F2": {f"F2__move_near_v0_s{K.FRACTAL_CALIB_SEED0 + i}__t0" for i in range(40)}}


def check_calib_only(roots, blocks):
    """S4：根目录（含 A/A 第二遍）里只许出现校准键；出现任何别的键（可能是正式键）就拒绝运行，不读任何成败。"""
    for b in blocks:
        allowed = {k["key"] for k in K.make_keys(b, "calib")} | LEGACY_CALIB.get(b, set())
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
                 max_episode_steps=int(z0["max_episode_steps"]), line_width=w, img_hw=[H, W],
                 line_eff_width_224_horiz=w * 224 / W, line_eff_width_224_vert=w * 224 / H,
                 g_links=json.loads(str(z0["g_links"])), robot_uid=str(z0["robot_uid"]),
                 sec_per_episode_median=float(np.median([float(z["sec"]) for z in r0.values()])))
        ious, ious2, gsum, gpx, protect = [], [], 0.0, 0, set()
        probe = {"same": [], "gripper_flip": [], "pos_shift": []}
        for z in r0.values():
            protect |= set(json.loads(str(z["p_names"])))
            for q in json.loads(str(z["qstats"])):
                if q.get("iou_lag_outline") is not None:
                    ious.append(q["iou_lag_outline"])
                if q.get("iou_lag2_outline") is not None:
                    ious2.append(q["iou_lag2_outline"])
                gsum += q["G_rgb_sum"]
                gpx += q["G_px"]
            for pr in json.loads(str(z["proprio_probe"])):
                for n in probe:
                    probe[n].append(pr[n])
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
        d["proprio_probe"] = {n: None if not v else {
            "n": len(v), "bit_equal": int(sum(x["bit_equal"] for x in v)),
            "max_abs_median": float(np.median([x["max_abs"] for x in v])),
            "max_abs_max": float(np.max([x["max_abs"] for x in v])),
            "gripper_dim_max_abs_median": float(np.median([x["gripper_dim_max_abs"] for x in v]))} for n, v in probe.items()}
        if a.aa_root:
            r1 = recs(a.aa_root, b, "C0")
            common = sorted(set(r0) & set(r1))
            if common:
                eq = [bool(np.array_equal(r0[k]["env_actions"], r1[k]["env_actions"])) for k in common]
                mx = [float(np.abs(r0[k]["raw_actions"].astype(np.float64) - r1[k]["raw_actions"]).max())
                      if r0[k]["raw_actions"].shape == r1[k]["raw_actions"].shape else float("inf") for k in common]
                flips = [k for k in common if bool(r0[k]["success"]) != bool(r1[k]["success"])]
                d["aa"] = {"n": len(common), "actions_bit_equal": int(sum(eq)), "raw_action_max_abs_diff": float(max(mx)),
                           "success_flips": len(flips), "flipped_keys": flips,
                           "init_state_equal": int(sum(np.array_equal(r0[k]["init_actor_poses"], r1[k]["init_actor_poses"])
                                                       for k in common))}
        res[b] = d
        flags[b] = {"blacken_weak": d["blacken_weak"], "lag_weak": lag_weak, "L": L_final}
    for p, o in ((a.out, res), (a.flags, flags)):
        with open(p, "w") as f:
            json.dump(o, f, ensure_ascii=False, indent=1)
    print(json.dumps(res, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
