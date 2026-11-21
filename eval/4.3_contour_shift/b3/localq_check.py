"""局部响应工具的冒烟/冻结前核查汇总（不读任何成败）：
  python localq_check.py --root out_localq_smoke --blocks B1 B2
报：两道核对；零平移后处理动作与记录 env_actions 是否相同；投影核对（末端参考点投影与 G 掩膜质心之差，输入像素）；
指令平移和（换到世界坐标）与实际执行记录动作后末端位移的夹角余弦（核对坐标系换算）；自然动作幅度 |r(0)|；
各画面视野内比例、与目标物外扩一圈相交的线像素、输入里可见的线像素；闭合前查询数；每键耗时、每次查询耗时。"""
import argparse
import glob
import json
import os

import numpy as np


def q(v, ps=(5, 50, 95)):
    v = [x for x in v if x is not None]
    return None if not v else [round(float(np.percentile(v, p)), 3) for p in ps]


ap = argparse.ArgumentParser()
ap.add_argument("--root", required=True)
ap.add_argument("--blocks", nargs="+", required=True)
a = ap.parse_args()
out = {}
for b in a.blocks:
    rows = []
    for p in sorted(glob.glob(os.path.join(a.root, b, "*.npz"))):
        z = np.load(p)
        meta = json.load(open(p[:-4] + ".json"))
        rows.append((z, meta))
    d = {"n_keys": len(rows), "check1": [bool(z["check1_success_steps_equal"]) for z, _ in rows],
         "check1_available": [bool(z["check1_available"]) for z, _ in rows],
         "check2_n_equal_over_n": [f"{int(z['check2_n_equal'])}/{int(z['n_queries'])}" for z, _ in rows],
         "zero_exec_equal_record": sum(m["zero_exec_equal_record"] for _, ms in rows for m in ms),
         "n_queries": sum(len(ms) for _, ms in rows), "preclose_queries_per_key": [sum(m["pre_close"] for m in ms) for _, ms in rows],
         "gpu": sorted({str(z["gpu_name"]) for z, _ in rows}), "sec_per_key": [round(float(z["sec"]), 1) for z, _ in rows]}
    d["sec_per_query"] = round(sum(float(z["sec"]) for z, _ in rows) / max(1, d["n_queries"]), 3)
    proj, cosv, r0 = [], [], []
    views = {}
    for z, ms in rows:
        for m in ms:
            if m["G_centroid_in"] is not None:
                proj.append(float(np.hypot(m["tcp_uv_in"][0] - m["G_centroid_in"][0], m["tcp_uv_in"][1] - m["G_centroid_in"][1])))
            dw = np.asarray(m["views"]["zero"]["dw_world_m"])
            act = np.asarray(m["tcp_after_exec_world"]) - np.asarray(m["tcp_world"])
            if np.linalg.norm(dw) > 5e-3 and np.linalg.norm(act) > 5e-3:
                cosv.append(float(dw @ act / np.linalg.norm(dw) / np.linalg.norm(act)))
            r0.append(float(np.hypot(*m["views"]["zero"]["r_in"])))
            for n, v in m["views"].items():
                e = views.setdefault(n, {"inview": [], "ring": [], "vis224": [], "protected": []})
                e["inview"].append(v["frac_inview"])
                e["ring"].append(v["line_ring_px"])
                e["vis224"].append(v["line_px_224"])
                e["protected"].append(v["frac_protected"])
    d["proj_tcp_minus_G_centroid_in_px_p5_50_95"] = q(proj)
    d["cos_cmd_vs_actual_tcp_move_p5_50_95"] = q(cosv)
    d["n_cos"] = len(cosv)
    d["abs_r0_in_px_p5_50_95"] = q(r0)
    d["views"] = {n: {"inview": q(e["inview"]), "frac_inview_lt_0.8": round(sum(1 for x in e["inview"] if x is not None and x < 0.8) / len(e["inview"]), 3),
                      "ring_px": q(e["ring"]), "vis224": q(e["vis224"]), "vis224_zero_frac": round(sum(1 for x in e["vis224"] if x == 0) / len(e["vis224"]), 3),
                      "protected": q(e["protected"])} for n, e in views.items()}
    out[b] = d
print(json.dumps(out, indent=1, ensure_ascii=False))
