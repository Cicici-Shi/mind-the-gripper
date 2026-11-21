"""回放结果的逐回合特征（真实状态），CogACT 与 open-pi-zero 的回放都能用（后者的回放没有 init_qpos_match 等核对字段，缺的就跳过）。
用法：python feat_any.py <回放根> <记录根 out_formal> <输出 json>

阶段定义照抄第一阶段 bench2_simpler/step0/replay_feat.py（open-pi-zero 回放用的那一份），阈值不变：
  NEAR = 8 cm；第 2 段 = 手指碰到过任务物或任务物被推动 >= 3 cm 但 check_grasp 从未为真；
  B1/B2：3 夹住但水平移动 < 3 cm；4 夹住并移动但最后没成；F1：3 夹住没抬起（< 2 cm）；F2：3 夹住但 moved_correct_obj 从未为真；5 最后一步成功。
相对那一份新增（都是事后描述用的量）：
  首次闭合指令步 t_c：WidowX 是 env_actions 第 7 维 < 0（绝对指令，-1 闭合）；Google Robot 是第 7 维 > 0.5
  （CogACT 适配器给的是相对指令，上一步张开度减这一步张开度，+1 为闭合，带 15 步粘滞）。
  首次物理闭合步 t_phys：get_gripper_closedness 比回合初值高出 0.3 的第一步。
  闭合处（执行闭合指令之前的状态）末端相对任务物的位移向量、水平距离、三维距离，以及此前任务物已被推动的距离。
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
    cs = json.loads(str(z["contact_src"]))
    contact = np.array([bool(x[0] or x[1]) if x is not None else False for x in cs])
    both = np.array([bool(x[0] and x[1]) if x is not None else False for x in cs])
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
    ea = np.asarray(r["env_actions"])
    T = len(ea)
    g = ea[:, 6]
    widowx = "widowx" in str(z["robot_uid"]).lower()
    close_cmd = (g < 0) if widowx else (g > 0.5)
    open_cmd = (g > 0) if widowx else (g < -0.5)
    tc = int(np.argmax(close_cmd)) if close_cmd.any() else None
    gc = np.asarray(z["grip_closed"])
    phys = gc > gc[0] + 0.3
    tp = int(np.argmax(phys)) if phys.any() else None
    dp = np.diff(tcp, axis=0)
    row = dict(
        block=b, cell=c, key=key, success=bool(succ[-1]), success_ever=bool(succ.any()), match=bool(z["match_success"]),
        init_ok=all(bool(z[k]) for k in ("init_actor_match", "init_qpos_match", "episode_id_match", "freq_match") if k in z.files),
        stage=st, T=T, hz=hz,
        dmin=float(d.min()), t_dmin=int(np.argmin(d)), d_end=float(d[-1]),
        t_near=int(np.argmax(d <= NEAR)) if (d <= NEAR).any() else None,
        contact=bool(contact.any()), both_contact=bool(both.any()),
        t_contact=int(np.argmax(contact)) if contact.any() else None,
        grasp=bool(grasp.any()), t_grasp=int(np.argmax(grasp)) if grasp.any() else None, grasp_end=bool(grasp[-1]),
        lift_max=float(lift.max()), xy_max=float(xy.max()), xy_end=float(xy[-1]), moved_correct=bool(moved_correct),
        src_tgt_end=None if ti is None else float(np.linalg.norm(A[-1, si, :2] - A[-1, ti, :2])),
        tcp_path=float(np.linalg.norm(dp, axis=1).sum()),
        tcp_path_early=float(np.linalg.norm(dp[: T // 3], axis=1).sum()),
        tcp_path_late=float(np.linalg.norm(dp[T // 2:], axis=1).sum()),
        cmd_path_early=float(np.linalg.norm(ea[: T // 3, :3], axis=1).sum()),
        pushed_no_grasp=bool((not grasp.any()) and xy.max() >= 0.03),
        tcp_z_min_rel_src0=float(tcp[:, 2].min() - src[0, 2]),
        n_close_cmd=int(close_cmd.sum()), n_open_cmd=int(open_cmd.sum()),
        n_close_onsets=int((np.diff(np.concatenate([[0], close_cmd.astype(int)])) == 1).sum()),
        t_close=tc, t_phys=tp, grip_closed_max=float(gc.max()), grip_closed_end=float(gc[-1]), grip_closed_0=float(gc[0]),
        src0=src[0].tolist(), tcp0=tcp[0].tolist(),
    )
    if tc is not None:
        off = tcp[tc] - src[tc]
        row.update(off_close=off.tolist(), dxy_close=float(np.linalg.norm(off[:2])), d_close=float(np.linalg.norm(off)),
                   src_moved_before_close=float(xy[tc]), contact_before_close=bool(contact[: tc + 1].any()),
                   off0_close=(tcp[tc] - src[0]).tolist())
    rows.append(row)
json.dump(rows, open(OUT, "w"))
print("records", len(rows), "match", sum(x["match"] for x in rows), "init_ok", sum(x["init_ok"] for x in rows))
