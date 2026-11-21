"""第 0 步行为描述：逐回合特征提取（只读 out_formal 的本地拷贝，不改任何东西）。

用法：python3 -I feat.py <out_formal 目录> <输出 feats.json>
每条记录一行特征；配对比较在 summarize.py 里做。
动作语义（报告里给了代码出处）：
  env_actions[:, 0:3] 平移增量（米，基座朝向的坐标轴，原点在末端）
  env_actions[:, 3:6] 旋转增量（轴角，弧度，范数上限 pi/2）
  env_actions[:, 6]   夹爪：WidowX 绝对指令 +1 张开 / -1 闭合；
                      Google Robot 相对指令，正（约 +1）闭合、负（约 -1）张开，适配器带 15 步粘滞闭合。
"""
import glob
import json
import os
import sys

import numpy as np

ROOT, OUT = sys.argv[1], sys.argv[2]
BLOCKS = ["B1", "B2", "F1", "F2"]
CELLS = ["C0", "C1", "C3", "C4", "C5"]


def grip_state(g, body):
    """逐步的指令夹爪状态：1 = 闭合，0 = 张开。"""
    if body == "widowx":
        return (g < 0).astype(int)
    st, cur = np.zeros(len(g), int), 0
    for i, x in enumerate(g):
        if x > 0.5:
            cur = 1
        elif x < -0.5:
            cur = 0
        st[i] = cur
    return st


def first(idx):
    return int(idx[0]) if len(idx) else None


rows = []
for b in BLOCKS:
    for c in CELLS:
        for f in sorted(glob.glob(os.path.join(ROOT, b, c, "*.npz"))):
            z = np.load(f, allow_pickle=False)
            body = "widowx" if "widowx" in str(z["robot_uid"]).lower() else "google"
            ea = np.asarray(z["env_actions"], np.float64)
            T = len(ea)
            dp = ea[:, :3]
            s = np.linalg.norm(dp, axis=1)
            P = np.cumsum(dp, axis=0)
            rot = np.linalg.norm(ea[:, 3:6], axis=1)
            gs = grip_state(ea[:, 6], body)
            close_on = np.where(np.diff(np.r_[0, gs]) == 1)[0]
            open_again = np.where(np.diff(np.r_[0, gs]) == -1)[0]
            qs = json.loads(str(z["qstats"]))
            qstep = np.array([q["step"] for q in qs])
            Gpx = np.array([q["G_px"] for q in qs], float)
            Rpx = np.array([q["R_px"] for q in qs], float)
            Apx = np.array([q["arm_px"] for q in qs], float)
            # 当前与 L 步前「真实可见夹爪」掩膜的 IoU（任何格都按真实渲染算），缺失或两帧都空为 nan
            iou = np.array([np.nan if q.get("iou_lag_mask") is None else q["iou_lag_mask"] for q in qs], float)
            H, W = [int(x) for x in z["img_hw"]]
            rows.append(dict(
                block=b, cell=c, key=str(z["key"]), body=body, T=T, qe=int(z["query_every"]),
                hz=float(z["control_freq"]), lag=int(z["lag_steps"]),
                success=bool(z["success"]), success_ever=bool(z["success_ever"]),
                first_success=first(np.where(np.asarray(z["success_steps"]))[0]),
                path=float(s.sum()), net=float(np.linalg.norm(P[-1])),
                rot_path=float(rot.sum()),
                step_mag=s.tolist(), P=P.tolist(), gs=gs.tolist(),
                first_close=first(close_on), n_close=int(len(close_on)), n_reopen=int(len(open_again)),
                frac_closed=float(gs.mean()),
                ea=ea.tolist(),
                q_step=qstep.tolist(), G_px=Gpx.tolist(), R_px=Rpx.tolist(), arm_px=Apx.tolist(),
                iou_lag_mask=[None if np.isnan(x) else float(x) for x in iou],
                img_px=H * W,
            ))
json.dump(rows, open(OUT, "w"))
print("records", len(rows))
