"""b3 参考轨迹回放（草案 protocol_cue_draft_v0 §3.3 与 §4 第 5 项；不跑策略、不读成败以外的东西）。
参考轨迹 = 第一阶段校准键清单的 C0 记录（b2/out_calib，只读），按记录的 env_actions 回放取状态。
校准键上 C0 允许看成败：这里只读 success_steps 来定「首次成功之前」的前缀，不用于任何参数选择以外的事。

第 1 遍（--pass 1，不渲染任何新格）：
  每个查询帧：G 的几何、224 可见像素、触边、运动学；对每个 --lags 的 L，按策略实际看到的滞后源帧
  src = max(t - L, 0)（不足 L 时用第 0 帧），记掩膜 IoU、轮廓线 IoU、配准平移（|dx|,|dy| <= 图宽/4）与残差。
  每个控制步：手指关节、指尖间距、TCP 位姿、保护物体位姿、指垫与保护物体是否接触；完整仿真状态快照的 sha256。
  核对：逐查询 G_px 与原记录相同（回放与原轨迹一致）。
第 2 遍（--pass 2 --params p.json）：同样回放，每个查询帧额外做模板渲染（q_open）、偏移、重排的统计；
  每个控制步的状态快照哈希与第 1 遍逐一比较（草案 §4 第 5 项），模板渲染并复原后重渲染与原图逐像素比较。
输出 <out>/<块>/pass<1|2>/<键>.json（不存画面）。
"""
import argparse
import glob
import hashlib
import json
import os
import sys

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import numpy as np  # noqa: E402

import keys as K  # noqa: E402
import views as V  # noqa: E402


def snap_sha(u):
    s = V._phys_snapshot(u)
    h = hashlib.sha256()
    for k in sorted(s):
        h.update(k.encode())
        h.update(np.ascontiguousarray(s[k]).tobytes())
    return h.hexdigest()


def contact_flag(u, finger_names, prot_names):
    for c in u._scene.get_contacts():
        n0, n1 = c.actor0.name, c.actor1.name
        if (n0 in finger_names and n1 in prot_names) or (n1 in finger_names and n0 in prot_names):
            if any(np.linalg.norm(p.impulse) > 0 for p in c.points) or any(p.separation <= 0 for p in c.points):
                return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--ref", required=True, help="参考 C0 记录根目录（b2/out_calib）")
    ap.add_argument("--out", required=True)
    ap.add_argument("--pass", dest="pas", type=int, choices=[1, 2], required=True)
    ap.add_argument("--lags", default="")
    ap.add_argument("--params", default="", help="第 2 遍：{块: {q_open, q_closed, shifts: {名: [dx, dy]}, copies: {名: [dx, dy]}}}")
    a = ap.parse_args()
    import simpler_env
    from eval_views import reset_key, check_env_src
    check_env_src()
    blk = K.BLOCKS[a.block]
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    calib = {k["key"] for k in K.make_keys(a.block, "calib")}
    assert all(k["key"] in calib for k in keys), "只许回放校准键"
    lags = [int(x) for x in a.lags.split(",") if x.strip()]
    prm = json.load(open(a.params))[a.block] if a.pas == 2 else None
    odir = os.path.join(a.out, a.block, f"pass{a.pas}")
    os.makedirs(odir, exist_ok=True)
    envs = {}
    for k in keys:
        z = np.load(os.path.join(a.ref, a.block, "C0", k["key"] + ".npz"))
        acts = z["env_actions"]
        succ = z["success_steps"]
        ref_gpx = [q["G_px"] for q in json.loads(str(z["qstats"]))]
        qe, L0, lw = int(z["query_every"]), int(z["lag_steps"]), int(z["line_width"])
        if k["task"] not in envs:
            envs[k["task"]] = simpler_env.make(k["task"])
        env = envs[k["task"]]
        u = env.unwrapped
        obs, info = reset_key(env, k)
        vm = V.ViewMaker3(env, "C0", blk["body"], L0, lw, qe, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
        vm.setup3(a.block, k["cfg_id"], k["trial"], {}, (), None, key=k["key"])
        vm.on_reset3()
        vm.strict_restore = False  # 回放核对时不中断：不等的情况记进 tpl_restore_all_equal 与 tpl_restore_maxdiff
        vm.check_restore(obs)
        assert vm.restore_maxdiff == 0
        links = {l.name: l for l in u.agent.robot.get_links()}
        ft = [links[n] for n in V.FINGERTIP_LINKS[blk["body"]]]
        finger_names = set(V.G_LINKS[blk["body"]])
        prot = [x for x in u.get_actors() if x.name in vm.p_names]
        H, W = obs["image"][vm.cam]["rgb"].shape[:2]
        frames, steps = [], []
        hashes = [snap_sha(u)]
        tpl_pix_eq, tpl_n = True, 0
        prevD = {}
        cam = vm.cam

        def step_log():
            q = np.asarray(u.agent.robot.get_qpos(), np.float64)
            return dict(finger_q=q[vm.fj_idx].tolist(), tip_dist=float(np.linalg.norm(ft[0].pose.p - ft[1].pose.p)),
                        closedness=float(u.agent.gripper_closedness), tcp_p=np.asarray(vm.tcp.pose.p).tolist(),
                        tcp_q=np.asarray(vm.tcp.pose.q).tolist(),
                        obj_p=[np.concatenate([o.pose.p, o.pose.q]).tolist() for o in prot],
                        contact=contact_flag(u, finger_names, set(vm.p_names)))
        steps.append(step_log())
        for i in range(len(acts)):
            if i % qe == 0:
                rgb = obs["image"][cam]["rgb"]
                seg = obs["image"][cam]["Segmentation"][..., 1]
                G = np.isin(seg, vm.gids)
                R = np.isin(seg, vm.robot_ids)
                vm.hist[i] = G.copy()
                vm.kin[i] = vm._kin_now()
                vm.R_cur = R
                f = {"step": i, "q": i // qe, "G_px": int(G.sum()), "G_px_224": int(V.mask_to_model(G).sum()),
                     "arm_px": int((R & ~G).sum()), "geo": V.mask_geom(G), "edge": V.edge_touch(G), "kin": vm.kin[i]}
                if a.pas == 1:
                    O = V.outline(G, lw)
                    lg = {}
                    for L in lags:
                        src = max(i - L, 0)
                        Gp = vm.hist[src]
                        d = {"src": src, "iou_mask": V.iou(G, Gp), "iou_outline": V.iou(O, V.outline(Gp, lw)),
                             "geo_src": V.mask_geom(Gp)}
                        if G.any() and Gp.any():
                            dx, dy, bi = V.reg_shift(Gp, G, W // 4)
                            d.update(reg_dx=dx, reg_dy=dy, reg_iou=bi)
                        lg[str(L)] = d
                    f["lag"] = lg
                else:
                    img_h, seg_h = vm.render_hidden()
                    P_full = np.isin(seg_h, vm.pids)
                    O = V.outline(G, lw)
                    D = O & ~P_full
                    imgc = img_h.copy()
                    imgc[D] = 0
                    f["C4"] = {"O_px": int(O.sum()), "D_px": int(D.sum()),
                               "frac_protected": None if O.sum() == 0 else float((O & P_full).sum()) / float(O.sum()),
                               "line_px_224": vm._vis224(imgc, img_h), "G_vis_after_hide_px": int(np.isin(seg_h, vm.gids).sum())}
                    cells = {"T": ("T", {"tpl": prm["q_open"]}), "T_closed": ("T", {"tpl": prm["q_closed"]})}
                    for nm, (dx, dy) in prm["shifts"].items():
                        cells[nm] = ("S", {"mode": "dxdy", "dxdy": [dx, dy]})
                    for nm, (dx, dy) in prm.get("copies", {}).items():
                        cells[nm] = ("SC", {"mode": "dxdy", "dxdy": [dx, dy]})
                    cells["X_query"] = ("X", {"seed_mode": "query"})
                    cells["X_episode"] = ("X", {"seed_mode": "episode"})
                    f["C4"]["iou_prev"] = V.iou(D, prevD["C4"]) if "C4" in prevD else None
                    prevD["C4"] = D
                    for nm, (c, v) in cells.items():
                        d = {}
                        _, Dn = vm._draw_new(c, v, G, i, img_h, seg_h, d)
                        # 与上一查询帧同格画线的 IoU（时间连续性；重排两种种子做法的对比用）
                        d["iou_prev"] = V.iou(Dn, prevD[nm]) if nm in prevD else None
                        prevD[nm] = Dn
                        if c == "T":
                            d["cur_tip_dist"] = steps[-1]["tip_dist"]
                            tpl_n += 1
                            # 复原后重渲染与原图逐像素相同
                            again = V.rgb_of(u.get_obs()["image"][cam])
                            d["rerender_equal"] = bool(np.array_equal(again, rgb))
                            tpl_pix_eq &= d["rerender_equal"]
                        f[nm] = d
                frames.append(f)
            obs, r, success, truncated, info = env.step(acts[i])
            hashes.append(snap_sha(u))
            steps.append(step_log())
            if truncated:
                break
        gpx = [f["G_px"] for f in frames]
        rec = {"key": k["key"], "task": k["task"], "H": H, "W": W, "qe": qe, "L0": L0, "line_width": lw,
               "max_steps": int(z["max_episode_steps"]), "n_steps": len(acts),
               "first_success_idx": int(np.argmax(succ)) if succ.any() else None,
               "grip_actions": acts[:, -1].tolist(), "replay_gpx_equal": gpx == ref_gpx,
               "finger_open_limit": vm.f_open.tolist(), "finger_closed_limit": vm.f_closed.tolist(),
               "finger_q0": vm.f_q0.tolist(), "frames": frames, "steps": steps, "state_sha": hashes,
               "tpl_renders": vm.n_tpl_render, "tpl_restore_all_equal": vm.tpl_restore_all_equal,
               "tpl_restore_maxdiff": vm.tpl_restore_maxdiff, "tpl_rerender_all_equal": tpl_pix_eq}
        if a.pas == 2:
            p1 = json.load(open(os.path.join(a.out, a.block, "pass1", k["key"] + ".json")))
            rec["state_sha_equal_pass1"] = p1["state_sha"] == hashes
            rec["n_state_sha_equal"] = sum(x == y for x, y in zip(p1["state_sha"], hashes))
        json.dump(rec, open(os.path.join(odir, k["key"] + ".json"), "w"))
        print(f"[replay] {a.block} pass{a.pas} {k['key']} gpx_equal {rec['replay_gpx_equal']}"
              + (f" state_equal {rec['state_sha_equal_pass1']} tpl {tpl_n} rerender {tpl_pix_eq}" if a.pas == 2 else ""), flush=True)


if __name__ == "__main__":
    main()
