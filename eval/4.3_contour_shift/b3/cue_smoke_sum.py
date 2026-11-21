"""冒烟汇总（校准键，盲：不读成败字段）：锚点逐位比对、闭环记录完整性与统计量、吞吐（查询部分待 v2）。
  python cue_smoke_sum.py --closed out_cue_smoke --out cue_smoke_summary.json
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import cue_cmp  # noqa: E402
import cue_spec as CS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--closed", default="out_cue_smoke")
    ap.add_argument("--out", default="cue_smoke_summary.json")
    a = ap.parse_args()
    res = {"anchor": {}, "closed": {}, "query": {}}
    for b in CS.BLOCKS:
        for c, ref in (("C0", "../b2/out_calib"), ("C3", "../b2/out_smoke_v11"), ("C4", "../b2/out_smoke_v11")):
            n, eq, dif = 0, 0, []
            for p in sorted(glob.glob(os.path.join(a.closed, b, c, "*.npz"))):
                k = os.path.basename(p)[:-4]
                rp = os.path.join(ref, b, c, k + ".npz")
                if os.path.exists(rp):
                    n += 1
                    e = cue_cmp.cmp_one(rp, p)
                    eq += e
                    if not e:
                        dif.append(k)
            res["anchor"][f"{b}/{c}"] = {"ref": ref, "compared": n, "identical": eq, "differ": dif}
        for c in CS.CELL_ORDER:
            rows = []
            for p in sorted(glob.glob(os.path.join(a.closed, b, c, "*.npz"))):
                z = np.load(p)
                T = int(z["steps"])
                ok_shape = z["st_tcp_pose"].shape[0] == T + 1 and z["q_step"].shape[0] == int(z["n_queries"])
                lc, lcc = z["q_line_centroid"], z["q_line_centroid_cur"]
                inv = z["q_frac_inview"]
                full = (inv == 1.0) & np.isfinite(lc[:, 0]) & np.isfinite(lcc[:, 0])
                sh = list(z["shift_env"])
                cerr = None
                if c.startswith("S") and full.any():
                    d = lc[full] - lcc[full] - np.array(sh, float)
                    cerr = float(np.abs(d).max())
                rows.append({"key": str(z["key"]), "shape_ok": bool(ok_shape), "sec": float(z["sec"]), "gpu": str(z["gpu_name"]),
                             "code_sha_cue_closed": json.loads(str(z["code_sha"]))["cue_closed.py"][:12],
                             "n_queries": int(z["n_queries"]), "changed_out_E_max": int(z["q_changed_out_E_px"].max()),
                             "line_vanish_frames": int(((z["q_line_px_224"] == 0) & (z["q_G_px"] > 0)).sum()),
                             "inview_lt08_frames": int((inv < 0.8).sum()),
                             "drawn_centroid_minus_cur_minus_shift_maxabs_inview1": cerr,
                             "tcp_in_G6_frac": float(np.mean(z["q_tcp_in_G6"][z["q_tcp_in_G6"] >= 0])) if (z["q_tcp_in_G6"] >= 0).any() else None,
                             "obj_on_mask_frac": float(np.mean(z["q_obj_on_mask"][z["q_obj_on_mask"] >= 0])) if (z["q_obj_on_mask"] >= 0).any() else None,
                             "frame0": str(z["frame0_sha256"])[:12], "seeds_sha": str(hash(z["seeds"].tobytes()))})
            res["closed"][f"{b}/{c}"] = rows
        # 同一键九格：第 0 帧哈希、种子、初始状态
        same = {}
        for k in sorted({r["key"] for c in CS.CELL_ORDER for r in res["closed"][f"{b}/{c}"]}):
            zs = [np.load(os.path.join(a.closed, b, c, k + ".npz")) for c in CS.CELL_ORDER
                  if os.path.exists(os.path.join(a.closed, b, c, k + ".npz"))]
            same[k] = {"n_cells": len(zs),
                       "frame0_same": len({str(z["frame0_sha256"]) for z in zs}) == 1,
                       "seeds_same": all(np.array_equal(zs[0]["seeds"], z["seeds"]) for z in zs),
                       "init_same": all(np.array_equal(zs[0]["init_actor_poses"], z["init_actor_poses"]) and
                                        np.array_equal(zs[0]["init_robot_qpos"], z["init_robot_qpos"]) for z in zs)}
        res["closed"][f"{b}/same_key"] = same
    json.dump(res, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(res["anchor"], indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
