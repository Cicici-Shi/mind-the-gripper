"""§10 实测值（protocol_cue_widowx.md v1 §10；v1 时的测量脚本，v3.1 只引用它的结果（§14 已有的实测值之一），参数名已跟随 cue_spec 改名；第四部分第 1、2、10 条）：在校准键原图格参考轨迹上量，不跑策略。
参考轨迹 = 第一阶段校准记录 b2/out_calib/<块>/C0（只读；校准键原图格允许看成败，这里只用 env_actions 与首次闭合步）。
  python cue_ref10.py replay --block B1 --keys ../b2/keysets/calib10_B1.json --ref ../b2/out_calib --out out_ref10
  python cue_ref10.py summarize --root out_ref10 --out cue_ref10.json
每个查询帧（回放，渲染）：
  * 参考夹爪宽度：G 包围盒宽 x 224/W（与 cue_calib.py 的 gripper_visible_width 同一定义，全部查询帧、掩膜非空）；
  * 各平移/副本画面（水平 ±34、±69，竖直 ±26、±51 及减半的 ±13、±26，副本 ±69）：视野内比例、被保护截掉比例、
    压在手臂上比例、线在输入里是否消失（可见像素 0 且掩膜非空）；
  * 操纵核对：平移路径零平移的图与 C4 逐像素相同；视野内比例为 1 的帧上平移轮廓质心减原轮廓质心 = 名义平移量；
  * 任务物图像平移路径：零位移与 C4 逐像素相同与否、差异像素数，±69 的视野内比例；
  * 投影核对：末端投影点落在 G 外扩 6 像素内；任务物原点投影落在（隐藏夹爪后的）任务物掩膜上；
  * 厘米换算：任务物初始高度水平面上，射线与平面求交，图像向右 1 个环境像素对应的桌面方向与长度；
  * 手到物方向（图像）与水平轴的夹角；
  * 指令平移坐标系核对：零画面（原图记录动作）指令平移和换世界坐标后，与实际执行 4 步后末端位移的方向余弦；
  * 接触与位移（定「接触前」）。
"""
import argparse
import glob
import json
import math
import os
import sys

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import numpy as np  # noqa: E402

import cue_spec as CS  # noqa: E402

SHIFTS = {"h-b": (-69, 0), "h-a": (-34, 0), "h+a": (34, 0), "h+b": (69, 0),
          "v-b": (0, -51), "v-a": (0, -26), "v+a": (0, 26), "v+b": (0, 51),
          "vh-b": (0, -26), "vh-a": (0, -13), "vh+a": (0, 13), "vh+b": (0, 26)}
COPIES = {"copy-b": (-69, 0), "copy+b": (69, 0)}


def project(p, intr, extr):
    pc = extr[:3, :3] @ np.asarray(p, float) + extr[:3, 3]
    uvw = intr @ pc
    return uvw[:2] / uvw[2]


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def ray_plane(u, v, intr, extr, z0):
    """像素 (u, v) 的视线与水平面 z = z0 的交点（世界坐标）。extr：世界 -> 相机（OpenCV）。"""
    Rm, t = extr[:3, :3], extr[:3, 3]
    d_cam = np.linalg.inv(intr) @ np.array([u, v, 1.0])
    c = -Rm.T @ t
    d = Rm.T @ d_cam
    lam = (z0 - c[2]) / d[2]
    return c + lam * d


def shift_img(img, dx, dy):
    """整数平移图像（越界丢弃，空位补 0）。"""
    out = np.zeros_like(img)
    H, W = img.shape[:2]
    if abs(dx) >= W or abs(dy) >= H:
        return out
    out[max(dy, 0):H + min(dy, 0), max(dx, 0):W + min(dx, 0)] = img[max(-dy, 0):H - max(dy, 0), max(-dx, 0):W - max(dx, 0)]
    return out


def centroid(M):
    ys, xs = np.nonzero(M)
    return None if len(xs) == 0 else (float(xs.mean()), float(ys.mean()))


def replay(a):
    import simpler_env
    import keys as K
    import views as V
    from eval_views import reset_key, check_env_src
    check_env_src()
    blk = K.BLOCKS[a.block]
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    calib = {k["key"] for k in K.make_keys(a.block, "calib")}
    assert all(k["key"] in calib for k in keys), "只许回放校准键"
    odir = os.path.join(a.out, a.block)
    os.makedirs(odir, exist_ok=True)
    env = simpler_env.make(blk["tasks"][0])
    for k in keys:
        z = np.load(os.path.join(a.ref, a.block, "C0", k["key"] + ".npz"))
        acts = z["env_actions"]
        g = acts[:, -1]
        fc = int(np.nonzero(g < 0)[0][0]) if (g < 0).any() else None
        qe, lw = int(z["query_every"]), int(z["line_width"])
        obs, info = reset_key(env, k)
        u = env.unwrapped
        vm = V.ViewMaker3(env, "C4", blk["body"], int(z["lag_steps"]), lw, qe, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
        vm.setup3(a.block, k["cfg_id"], k["trial"], {}, (), None, key=k["key"])
        vm.on_reset3()
        vm.check_restore(obs)
        assert vm.restore_maxdiff == 0
        cam = vm.cam
        tobj = [x for x in u.get_actors() if x.name == CS.TARGET[a.block]][0]
        tid = np.array([tobj.id], np.int32)
        obj_vbs = list(tobj.get_visual_bodies())
        p0 = np.asarray(tobj.pose.p, float).copy()
        Rb = quat2mat(np.asarray(u.agent.robot.pose.q, float))
        H, W = obs["image"][cam]["rgb"].shape[:2]
        contact_hist = []
        frames = []
        ref_gpx = [s["G_px"] for s in json.loads(str(z["qstats"]))]
        for i in range(len(acts)):
            c = u.agent.check_contact_fingers(tobj)
            contact_hist.append(bool(c[0] or c[1]))
            if i % qe == 0:
                rgb = obs["image"][cam]["rgb"]
                seg = obs["image"][cam]["Segmentation"][..., 1]
                img_c4, _ = vm.make(obs, i)
                G = vm.hist[i]
                vm.R_cur = np.isin(seg, vm.robot_ids)
                img_h, seg_h = vm.render_hidden()
                P_full = np.isin(seg_h, vm.pids)
                T = np.isin(seg_h, tid)
                O = V.outline(G, lw)
                cp = obs["camera_param"][cam]
                intr, extr = np.asarray(cp["intrinsic_cv"], float), np.asarray(cp["extrinsic_cv"], float)
                tcp = np.asarray(vm.tcp.pose.p, float)
                uv, ut = project(tcp, intr, extr), project(tobj.pose.p, intr, extr)
                gm = V.mask_geom(G)
                f = {"step": i, "q": i // qe, "pre_close": fc is None or i <= fc, "G_px": int(G.sum()),
                     "width_in": None if gm is None else (gm["x1"] - gm["x0"]) * 224.0 / W,
                     "contact_now": contact_hist[-1], "never_contact": not any(contact_hist),
                     "grasp": bool(u.agent.check_grasp(tobj)), "obj_disp": float(np.linalg.norm(np.asarray(tobj.pose.p) - p0)),
                     "tcp_uv": uv.tolist(), "obj_uv": ut.tolist()}
                D6 = V._dilate_sq(G, CS.PROJ_DIL_PX)
                x, y = int(round(uv[0])), int(round(uv[1]))
                f["tcp_in_G6"] = None if not G.any() else bool(0 <= x < W and 0 <= y < H and D6[y, x])
                x, y = int(round(ut[0])), int(round(ut[1]))
                f["obj_on_mask"] = None if not T.any() else bool(0 <= x < W and 0 <= y < H and T[y, x])
                # 厘米换算：任务物初始高度平面，从任务物投影点向右 1 环境像素
                pa = ray_plane(ut[0], ut[1], intr, extr, p0[2])
                pb = ray_plane(ut[0] + 1.0, ut[1], intr, extr, p0[2])
                dvec = pb - pa
                f["cm_per_env_px"] = float(np.linalg.norm(dvec[:2]) * 100)
                f["right_dir_xy"] = (dvec[:2] / max(np.linalg.norm(dvec[:2]), 1e-12)).tolist()
                hv = ut - uv
                f["hand_to_obj_angle_deg"] = float(math.degrees(math.atan2(hv[1], hv[0])))  # 相对 +u 轴，v 下正
                # 平移/副本
                st = {}
                for nm, dxdy in list(SHIFTS.items()) + list(COPIES.items()):
                    d = {}
                    img, draw = vm._draw_new("SC" if nm in COPIES else "S", {"mode": "dxdy", "dxdy": list(dxdy)}, G, i, img_h, seg_h, d)
                    st[nm] = {"inview": d.get("frac_inview"), "prot": d.get("frac_protected"), "arm": d.get("frac_on_arm"),
                              "vanish": bool(G.any() and d["line_px_224"] == 0), "l224": d["line_px_224"]}
                    if nm in SHIFTS and d.get("frac_inview") == 1.0:
                        c0, c1 = centroid(O), centroid(V.shift_mask(O, *dxdy))
                        st[nm]["cshift_err"] = [c1[0] - c0[0] - dxdy[0], c1[1] - c0[1] - dxdy[1]]
                f["shift"] = st
                d0 = {}
                img_s0, _ = vm._draw_new("S", {"mode": "dxdy", "dxdy": [0, 0]}, G, i, img_h, seg_h, d0)
                f["s0_equal_c4"] = bool(np.array_equal(img_s0, img_c4))
                f["c4_line_vanish"] = bool(G.any() and vm._vis224(img_c4, img_h) == 0)
                # 任务物图像平移路径
                vm._set_vis(0.0)
                for vb in obj_vbs:
                    vb.set_visibility(0.0)
                try:
                    base = V.rgb_of(u.get_obs()["image"][cam])
                finally:
                    vm._set_vis(1.0)
                    for vb in obj_vbs:
                        vb.set_visibility(1.0)
                z0 = base.copy()
                z0[T] = img_h[T]
                z0[O & ~P_full] = 0
                dm = np.any(z0 != img_c4, axis=2)
                f["obj_zero_equal"] = bool(not dm.any())
                f["obj_zero_ndiff"] = int(dm.sum())
                f["obj_zero_ndiff_outT"] = int((dm & ~T).sum())
                f["obj_zero_maxdiff"] = int(np.abs(z0.astype(int) - img_c4.astype(int)).max())
                f["obj_T_px"] = int(T.sum())
                for s_ in (-69, 69):
                    Ts = V.shift_mask(T, s_, 0)
                    f[f"obj{s_:+d}_inview"] = None if not T.any() else float(Ts.sum()) / float(T.sum())
                # 指令坐标系核对
                chunk = acts[i:i + qe]
                f["cmd_world"] = (Rb @ chunk[:, :3].sum(0)).tolist()
                f["_tcp_before"] = tcp.tolist()
                frames.append(f)
            obs, *_ = env.step(acts[i])
            if (i + 1) % qe == 0 or i + 1 == len(acts):
                fr = frames[-1]
                ex = np.asarray(vm.tcp.pose.p, float) - np.asarray(fr["_tcp_before"])
                cw = np.asarray(fr["cmd_world"])
                fr["exec_cos"] = None if np.linalg.norm(cw) < 1e-6 or np.linalg.norm(ex) < 1e-6 else \
                    float(cw @ ex / np.linalg.norm(cw) / np.linalg.norm(ex))
                fr["exec_norm_m"], fr["cmd_norm_m"] = float(np.linalg.norm(ex)), float(np.linalg.norm(cw))
        gpx = [f["G_px"] for f in frames]
        rec = {"key": k["key"], "block": a.block, "first_close_step": fc, "q_c": None if fc is None else fc // qe,
               "replay_gpx_equal": gpx == ref_gpx, "frames": frames}
        json.dump(rec, open(os.path.join(odir, k["key"] + ".json"), "w"))
        print(f"[ref10] {a.block} {k['key']} gpx_equal {rec['replay_gpx_equal']}", flush=True)


def qt(v):
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if not len(v):
        return None
    return {"n": int(len(v)), "median": round(float(np.median(v)), 4), "p5": round(float(np.percentile(v, 5)), 4),
            "p25": round(float(np.percentile(v, 25)), 4), "p75": round(float(np.percentile(v, 75)), 4),
            "p95": round(float(np.percentile(v, 95)), 4), "mean": round(float(v.mean()), 4)}


def frac(xs):
    xs = [x for x in xs if x is not None]
    return None if not xs else round(float(np.mean(xs)), 4)


def summarize(a):
    out = {}
    for bdir in sorted(glob.glob(os.path.join(a.root, "B*"))):
        b = os.path.basename(bdir)
        eps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(bdir, "*.json")))]
        allf = [f for e in eps for f in e["frames"]]
        pre = [f for f in allf if f["pre_close"]]
        r = {"n_keys": len(eps), "n_frames": len(allf), "n_preclose_frames": len(pre),
             "replay_gpx_equal": sum(e["replay_gpx_equal"] for e in eps),
             "gripper_width_input_px": qt([f["width_in"] for f in allf if f["G_px"] > 0])}
        w = r["gripper_width_input_px"]["median"]
        r["a_over_width"], r["b_over_width"] = round(CS.AMP_IN["a"] / w, 4), round(CS.AMP_IN["b"] / w, 4)
        r["shift_stats_preclose"] = {}
        for nm in list(SHIFTS) + list(COPIES):
            fs = [f["shift"][nm] for f in pre if f["G_px"] > 0]
            r["shift_stats_preclose"][nm] = {
                "out_of_view_frame_frac": frac([x["inview"] < CS.PARAMS["inview_thr"] for x in fs if x["inview"] is not None]),
                "protected_frac_median": qt([x["prot"] for x in fs])["median"] if qt([x["prot"] for x in fs]) else None,
                "protected_frac_mean": qt([x["prot"] for x in fs])["mean"] if qt([x["prot"] for x in fs]) else None,
                "on_arm_frac_mean": qt([x["arm"] for x in fs])["mean"] if qt([x["arm"] for x in fs]) else None,
                "line_vanish_frame_frac": frac([x["vanish"] for x in fs]), "n": len(fs)}
        r["c4_line_vanish_frac_preclose"] = frac([f["c4_line_vanish"] for f in pre if f["G_px"] > 0])
        r["s0_equal_c4"] = [sum(f["s0_equal_c4"] for f in allf), len(allf)]
        errs = [abs(e) for f in allf for nm in SHIFTS if "cshift_err" in f["shift"][nm] for e in f["shift"][nm]["cshift_err"]]
        r["centroid_shift_err_max_px"] = None if not errs else max(errs)
        r["centroid_shift_n_frames"] = sum(1 for f in allf for nm in SHIFTS if "cshift_err" in f["shift"][nm])
        r["obj_zero_equal"] = [sum(f["obj_zero_equal"] for f in allf), len(allf)]
        r["obj_zero_ndiff"] = qt([f["obj_zero_ndiff"] for f in allf])
        r["obj_zero_ndiff_outT"] = qt([f["obj_zero_ndiff_outT"] for f in allf])
        r["obj_zero_maxdiff_max"] = max(f["obj_zero_maxdiff"] for f in allf)
        r["obj_shift_out_of_view_frac_preclose"] = {s: frac([f[f"obj{s}_inview"] < CS.PARAMS["inview_thr"] for f in pre
                                                              if f[f"obj{s}_inview"] is not None]) for s in ("-69", "+69")}
        r["proj_tcp_in_G6_frac"] = frac([f["tcp_in_G6"] for f in allf])
        r["proj_obj_on_mask_frac"] = frac([f["obj_on_mask"] for f in allf])
        cm = [f["cm_per_env_px"] for f in allf]
        r["cm_per_env_px"] = qt(cm)
        r["cm_per_input_px_h"] = round(float(np.median(cm)) * CS.W_ENV / 224.0, 4)
        r["cm_at_a_b"] = [round(float(np.median(cm)) * CS.AMP_ENV["a"], 3), round(float(np.median(cm)) * CS.AMP_ENV["b"], 3)]
        r["exec_cos"] = qt([f.get("exec_cos") for f in allf if f.get("cmd_norm_m", 0) > 0.005])
        # 手到物方向与水平轴的夹角（折到 0 到 90 度）：q=0 与 q_c-2、q_c-1
        def fold(d):
            d = abs(d) % 180
            return min(d, 180 - d)
        a0 = [fold(e["frames"][0]["hand_to_obj_angle_deg"]) for e in eps]
        am = [fold(f["hand_to_obj_angle_deg"]) for e in eps if e["q_c"] is not None for f in e["frames"]
              if f["q"] - e["q_c"] in (-2, -1)]
        r["hand_obj_angle_fold_q0"], r["hand_obj_angle_fold_main"] = qt(a0), qt(am)
        # 响应矩阵：每键 rel ∈ {-2,-1} 且接触前的状态上，h、v 两对在 a、b 两档都在视野内（两侧 >= 0.8）
        for tag, vb in (("full", ("v-a", "v+a", "v-b", "v+b")), ("halved", ("vh-a", "vh+a", "vh-b", "vh+b"))):
            n_state, n_key = 0, 0
            for e in eps:
                if e["q_c"] is None:
                    continue
                ok_key = False
                for f in e["frames"]:
                    if f["q"] - e["q_c"] not in (-2, -1):
                        continue
                    if not (f["never_contact"] and not f["grasp"] and f["obj_disp"] < CS.PARAMS["precontact_disp_m"]):
                        continue
                    need = ("h-a", "h+a", "h-b", "h+b") + vb
                    if all((f["shift"][nm]["inview"] or 0) >= CS.PARAMS["inview_thr"] for nm in need):
                        n_state += 1
                        ok_key = True
                n_key += ok_key
            r[f"matrix_{tag}"] = {"states": n_state, "keys": n_key, "keys_with_close": sum(e["q_c"] is not None for e in eps),
                                  "projected_keys_of_120": round(120.0 * n_key / max(len(eps), 1), 1)}
        r["main_state_precontact"] = sum(1 for e in eps if e["q_c"] is not None for f in e["frames"]
                                         if f["q"] - e["q_c"] in (-2, -1) and f["never_contact"] and not f["grasp"]
                                         and f["obj_disp"] < CS.PARAMS["precontact_disp_m"])
        r["main_state_total"] = sum(1 for e in eps if e["q_c"] is not None for f in e["frames"] if f["q"] - e["q_c"] in (-2, -1))
        out[b] = r
    json.dump(out, open(a.out, "w"), indent=1, ensure_ascii=False)
    print(json.dumps(out, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["replay", "summarize"])
    ap.add_argument("--block")
    ap.add_argument("--keys")
    ap.add_argument("--ref", default="../b2/out_calib")
    ap.add_argument("--out")
    ap.add_argument("--root")
    a = ap.parse_args()
    replay(a) if a.mode == "replay" else summarize(a)
