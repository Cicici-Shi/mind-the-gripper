"""回放结果的逐回合特征（真实状态）。用法：python3 -I replay_feat.py <回放根> <记录根 out_formal> <输出 json>

阶段（每回合取「最远到过」的一段；定义在看回放结果之前写好，阈值见下）：
  三块共用：第 2 段 = 手指碰到过任务物，或任务物被推动 >= 3 cm，但 check_grasp 从未为真。
  B1/B2 放置：0 没接近（末端离任务物最近距离 > NEAR）；1 接近但手指没碰到；2 碰到但没夹住（check_grasp 从未为真）；
             3 夹住但任务物水平移动 < 3 cm；4 夹住并移动但最后一步没放到位；5 成功（最后一步）
  F1 抓取：0 没接近；1 接近没碰；2 碰到没夹住；3 夹住没抬起（任务物最高抬升 < 2 cm）；4 抬起过但最后一步不算成功；5 成功
  F2 移近：0 没接近；1 接近没碰；2 碰到没夹住；3 夹住但 moved_correct_obj 从未为真；4 移动过正确物体但最后一步没成功；5 成功
接近阈值 NEAR：末端点（WidowX ee_gripper_link，Google link_gripper_tcp）到任务物中心 8 cm。任务物尺寸 5 到 15 cm，
中心距离 8 cm 只是粗代理；同时给出最近距离的分布。
"""
import glob
import json
import os
import sys

import numpy as np

RP, REC, OUT = sys.argv[1], sys.argv[2], sys.argv[3]
NEAR = 0.08
rows = []
for f in sorted(glob.glob(os.path.join(RP, "*", "*", "*.npz"))):
    z = np.load(f)
    b, c, key = str(z["block"]), str(z["cell"]), str(z["key"])
    r = np.load(os.path.join(REC, b, c, key + ".npz"))
    names = json.loads(str(z["actor_names"]))
    si = names.index(str(z["src_name"]))
    ti = names.index(str(z["tgt_name"])) if str(z["tgt_name"]) in names else None
    tcp = z["tcp"][:, :3]
    A = z["actors"]
    src = A[:, si, :3]
    d = np.linalg.norm(tcp - src, axis=1)
    contact = [bool(x[0] or x[1]) if x is not None else False for x in json.loads(str(z["contact_src"]))]
    contact = np.array(contact)
    both = np.array([bool(x[0] and x[1]) if x is not None else False for x in json.loads(str(z["contact_src"]))])
    grasp = np.asarray(z["grasp_src"], bool)
    lift = src[:, 2] - src[0, 2]
    xy = np.linalg.norm(src[:, :2] - src[0, :2], axis=1)
    infos = json.loads(str(z["infos"]))
    succ = np.asarray(z["success_steps"], bool)
    hz = float(r["control_freq"])
    moved_correct = any(i.get("moved_correct_obj", False) for i in infos)
    st = 0
    if d.min() <= NEAR:
        st = 1
    # 第 2 段含「没用手指碰到、但任务物被推动 >= 3 cm」：check_contact_fingers 只看手指连杆，
    # 校验集里有 F2 回合任务物被移动而手指从未接触（看 40 条校验回放后补的，全量结果之前）
    if contact.any() or xy.max() >= 0.03:
        st = max(st, 2)
    if grasp.any():
        st = 3
        if b in ("B1", "B2") and xy[grasp.argmax():].max() >= 0.03:
            st = 4
        if b == "F1" and lift.max() >= 0.02:
            st = 4
        if b == "F2" and moved_correct:
            st = 4
    if succ[-1]:
        st = 5
    # 末端真实路径（逐步），与指令积分对比
    dp = np.diff(tcp, axis=0)
    ea = np.asarray(r["env_actions"])
    first = lambda m: (int(np.argmax(m)) / hz) if np.any(m) else None
    gw = np.asarray(z["grip_width"])
    T = len(ea)
    rows.append(dict(
        block=b, cell=c, key=key, success=bool(succ[-1]), match=bool(z["match_success"]), stage=st,
        dmin=float(d.min()), t_dmin=float(np.argmin(d) / hz), d_end=float(d[-1]),
        t_near=first(d <= NEAR), contact=bool(contact.any()), both_contact=bool(both.any()), t_contact=first(contact),
        grasp=bool(grasp.any()), t_grasp=first(grasp), grasp_frac=float(grasp.mean()),
        grasp_end=bool(grasp[-1]), lift_max=float(lift.max()), xy_max=float(xy.max()), xy_end=float(xy[-1]),
        moved_correct=bool(moved_correct),
        src_tgt_end=None if ti is None else float(np.linalg.norm(A[-1, si, :2] - A[-1, ti, :2])),
        tcp_path=float(np.linalg.norm(dp, axis=1).sum()), tcp_net=float(np.linalg.norm(tcp[-1] - tcp[0])),
        cmd_path=float(np.linalg.norm(ea[:, :3], axis=1).sum()), cmd_net=float(np.linalg.norm(ea[:, :3].sum(0))),
        tcp_path_late=float(np.linalg.norm(dp[T // 2:], axis=1).sum()),
        tcp_path_early=float(np.linalg.norm(dp[: T // 3], axis=1).sum()),
        cmd_path_early=float(np.linalg.norm(ea[: T // 3, :3], axis=1).sum()),
        pushed_no_grasp=bool((not grasp.any()) and xy.max() >= 0.03),
        tcp_z_min_rel_src0=float(tcp[:, 2].min() - src[0, 2]),
        grip_width_min=float(gw.min()), grip_width_end=float(gw[-1]),
        grip_closed_frac=float((gw < 0.5).mean()),
        d_series=d.tolist(), grasp_series=grasp.astype(int).tolist(), hz=hz,
    ))
json.dump(rows, open(OUT, "w"))
print("records", len(rows), "match", sum(x["match"] for x in rows))
