"""v3 §3 任务物图像平移合成的三条放弃条件（冻结前，校准键原图格参考帧；只量，不改采集代码）。
  python cue_obcheck.py replay --block B1 --keys ../b2/keysets/calib10_B1.json --ref ../b2/out_calib --out out_obcheck
  python cue_obcheck.py summarize --root out_obcheck
合成（v3 §3 逐条）：T = 隐藏夹爪图分割里的任务物像素；底图 = 夹爪与任务物都隐藏；把隐藏夹爪图在 T 上的像素水平平移 ±69 贴上，
越界丢弃，落在机器人像素（隐藏夹爪后分割图里的机器人连杆）上的不贴；保护集合 = 原保护 - T 并 实际贴上的像素；再画当前轮廓。
  条件一：零位移输出与 C4 逐像素相同（全部查询帧）。
  条件二：隐藏任务物只改变 T 之内的像素（底图与隐藏夹爪图在 T 外的差异像素数，全部查询帧）。
  条件三：贴上率 = 实际贴上像素 / |T|；首次闭合之前（step <= 首次闭合步）的帧里，贴上率 < 0.8 的帧占比 <= 5%，正负两侧各自算。
v3.1 §17 第 9 条补量（同一次回放，结果写进同一个 OBCHECK.json）：
  贴上去的像素落在目标容器掩膜上的帧占比（按块、按侧；容器 = cue_spec.CONTAINER，取隐藏夹爪后的分割图）；
  末端投影点落在夹爪掩膜包围盒之内的帧占比；物体静置时的水平漂移（从未接触、未夹住的步上，任务物原点相对第 0 步的水平位移）；
  相机外参在回合内是否逐次相同；半档 ±17 环境像素的出视野、被保护截掉、压在手臂上、线消失四项（首次闭合之前），
  以及半档的零平移逐像素相同（S 路径 dxdy=[0,0] 对 C4）与截断前质心核对。
"""
import argparse
import glob
import json
import os
import sys

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
import numpy as np  # noqa: E402

import cue_spec as CS  # noqa: E402


def shift_img(img, dx):
    out = np.zeros_like(img)
    W = img.shape[1]
    out[:, max(dx, 0):W + min(dx, 0)] = img[:, max(-dx, 0):W - max(dx, 0)]
    return out


def synth(base, img_h, seg_h, T, robot, P_full, O, dx):
    Ts = np.zeros_like(T)
    W = T.shape[1]
    Ts[:, max(dx, 0):W + min(dx, 0)] = T[:, max(-dx, 0):W - max(dx, 0)]
    pasted = Ts & ~robot
    img = base.copy()
    img[pasted] = shift_img(img_h, dx)[pasted]
    P = (P_full & ~T) | pasted
    img[O & ~P] = 0
    return img, pasted


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
    b = CS.AMP_ENV["b"]
    h = CS.AMP_ENV["h"]
    for k in keys:
        z = np.load(os.path.join(a.ref, a.block, "C0", k["key"] + ".npz"))
        acts = z["env_actions"]
        g = acts[:, -1]
        fc = int(np.nonzero(g < 0)[0][0]) if (g < 0).any() else None
        qe, lw = int(z["query_every"]), int(z["line_width"])
        obs, _ = reset_key(env, k)
        u = env.unwrapped
        vm = V.ViewMaker3(env, "C4", blk["body"], int(z["lag_steps"]), lw, qe, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
        vm.setup3(a.block, k["cfg_id"], k["trial"], {}, (), None, key=k["key"])
        vm.on_reset3()
        vm.check_restore(obs)
        assert vm.restore_maxdiff == 0
        cam = vm.cam
        tobj = [x for x in u.get_actors() if x.name == CS.TARGET[a.block]][0]
        tid = np.array([tobj.id], np.int32)
        cont = [x for x in u.get_actors() if x.name == CS.CONTAINER[a.block]]
        assert len(cont) == 1, CS.CONTAINER[a.block]
        cid = np.array([cont[0].id], np.int32)
        p0 = np.asarray(tobj.pose.p, float).copy()
        contact_hist, grasp_hist, drift = [], [], []
        extr0 = None
        vbs = list(tobj.get_visual_bodies())
        frames = []
        ref_gpx = [s["G_px"] for s in json.loads(str(z["qstats"]))]
        for i in range(len(acts)):
            c = u.agent.check_contact_fingers(tobj)
            contact_hist.append(bool(c[0] or c[1]))
            grasp_hist.append(bool(u.agent.check_grasp(tobj)))
            if not any(contact_hist) and not any(grasp_hist):
                drift.append(float(np.linalg.norm(np.asarray(tobj.pose.p, float)[:2] - p0[:2])))
            if i % qe == 0:
                img_c4, _ = vm.make(obs, i)
                G = vm.hist[i]
                img_h, seg_h = vm.render_hidden()
                P_full = np.isin(seg_h, vm.pids)
                T = np.isin(seg_h, tid)
                robot = np.isin(seg_h, vm.robot_ids)
                O = V.outline(G, lw)
                vm._set_vis(0.0)
                for vb in vbs:
                    vb.set_visibility(0.0)
                try:
                    base = V.rgb_of(u.get_obs()["image"][cam])
                finally:
                    vm._set_vis(1.0)
                    for vb in vbs:
                        vb.set_visibility(1.0)
                z0, _ = synth(base, img_h, seg_h, T, robot, P_full, O, 0)
                dT = np.any(base != img_h, axis=2)
                f = {"step": i, "pre_close": fc is None or i <= fc, "T_px": int(T.sum()), "G_px": int(G.sum()),
                     "c1_equal": bool(np.array_equal(z0, img_c4)), "c2_diff_outT": int((dT & ~T).sum()),
                     "T_on_robot_px": int((T & robot).sum())}
                f["robot_px"] = int(robot.sum())
                cp = obs["camera_param"][cam]
                intr, extr = np.asarray(cp["intrinsic_cv"], float), np.asarray(cp["extrinsic_cv"], float)
                if extr0 is None:
                    extr0, intr0 = extr.copy(), intr.copy()
                f["extr_same_as_q0"] = bool(np.array_equal(extr, extr0) and np.array_equal(intr, intr0))
                pc = extr[:3, :3] @ np.asarray(vm.tcp.pose.p, float) + extr[:3, 3]
                uvw = intr @ pc
                uv = uvw[:2] / uvw[2]
                gm = V.mask_geom(G)
                f["tcp_in_Gbbox"] = None if gm is None else bool(gm["x0"] <= uv[0] < gm["x1"] and gm["y0"] <= uv[1] < gm["y1"])
                Cm = np.isin(seg_h, cid)
                f["container_px"] = int(Cm.sum())
                # 半档 ±17：四项统计、零平移逐像素、截断前质心
                d0 = {}
                img_s0, _ = vm._draw_new("S", {"mode": "dxdy", "dxdy": [0, 0]}, G, i, img_h, seg_h, d0)
                f["s0_equal_c4"] = bool(np.array_equal(img_s0, img_c4))
                vm.R_cur = np.isin(obs["image"][cam]["Segmentation"][..., 1], vm.robot_ids)
                for s_ in (-h, h):
                    d = {}
                    vm._draw_new("S", {"mode": "dxdy", "dxdy": [s_, 0]}, G, i, img_h, seg_h, d)
                    e = {"inview": d.get("frac_inview"), "prot": d.get("frac_protected"), "arm": d.get("frac_on_arm"),
                         "vanish": bool(G.any() and d["line_px_224"] == 0)}
                    if d.get("frac_inview") == 1.0:
                        ys, xs = np.nonzero(O)
                        ys2, xs2 = np.nonzero(V.shift_mask(O, s_, 0))
                        e["cshift_err"] = max(abs(xs2.mean() - xs.mean() - s_), abs(ys2.mean() - ys.mean()))
                    f[f"half{s_:+d}"] = e
                rob20 = V._dilate_sq(robot, 20)
                for s in (-b, b):
                    _, pasted = synth(base, img_h, seg_h, T, robot, P_full, O, s)
                    Tsh = np.zeros_like(T)
                    Tsh[:, max(s, 0):T.shape[1] + min(s, 0)] = T[:, max(-s, 0):T.shape[1] - max(s, 0)]
                    f[f"near20{s:+d}"] = int((Tsh & rob20).sum())
                    f[f"on_container_px{s:+d}"] = int((pasted & Cm).sum())
                    Ts_n = int(np.roll(T, 0).sum())
                    f[f"rate{s:+d}"] = None if not T.any() else float(pasted.sum()) / float(T.sum())
                    f[f"inview{s:+d}"] = None if not T.any() else float(
                        T[:, max(-s, 0):T.shape[1] - max(s, 0)].sum()) / float(Ts_n)
                frames.append(f)
            obs, *_ = env.step(acts[i])
        rec = {"key": k["key"], "first_close_step": fc, "drift_max_m": max(drift) if drift else None, "n_drift_steps": len(drift), "gpx_equal": [f["G_px"] for f in frames] == ref_gpx, "frames": frames}
        json.dump(rec, open(os.path.join(odir, k["key"] + ".json"), "w"))
        print(f"[ob] {a.block} {k['key']} gpx_equal {rec['gpx_equal']}", flush=True)


def summarize(a):
    out = {}
    for bd in sorted(glob.glob(os.path.join(a.root, "B*"))):
        eps = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(bd, "*.json")))]
        fr = [f for e in eps for f in e["frames"]]
        pre = [f for f in fr if f["pre_close"] and f["T_px"] > 0]
        r = {"keys": len(eps), "gpx_equal": sum(e["gpx_equal"] for e in eps), "frames": len(fr), "pre_frames": len(pre),
             "c1_equal": [sum(f["c1_equal"] for f in fr), len(fr)], "c2_frames_with_outT_diff": sum(f["c2_diff_outT"] > 0 for f in fr),
             "c2_outT_diff_max": max(f["c2_diff_outT"] for f in fr), "T_empty_frames": sum(f["T_px"] == 0 for f in fr)}
        for s in ("-69", "+69"):
            v = np.array([f["rate" + s] for f in pre], float)
            r["rate" + s] = {"median": round(float(np.median(v)), 4), "p5": round(float(np.percentile(v, 5)), 4),
                             "min": round(float(v.min()), 4), "frac_lt_0.8": round(float((v < 0.8).mean()), 4),
                             "inview_lt_0.8": round(float((np.array([f["inview" + s] for f in pre]) < 0.8).mean()), 4)}
            r["c3_pass" + s] = r["rate" + s]["frac_lt_0.8"] <= CS.PARAMS["flag_frac"]
            r["on_container_frame_frac" + s] = {"all": round(float(np.mean([f["on_container_px" + s] > 0 for f in fr if "on_container_px" + s in f])), 4)
                                                if all("on_container_px" + s in f for f in fr) else None,
                                                "pre_close": round(float(np.mean([f["on_container_px" + s] > 0 for f in pre if "on_container_px" + s in f])), 4)
                                                if all("on_container_px" + s in f for f in pre) else None}
        if all("tcp_in_Gbbox" in f for f in fr):
            v = [f["tcp_in_Gbbox"] for f in fr if f["tcp_in_Gbbox"] is not None]
            r["tcp_in_Gbbox_frac"] = round(float(np.mean(v)), 4)
            r["extr_constant_keys"] = sum(all(f["extr_same_as_q0"] for f in e["frames"]) for e in eps)
            dr = [e["drift_max_m"] for e in eps if e["drift_max_m"] is not None]
            r["rest_drift_xy_m"] = {"max_over_keys": max(dr), "median_of_key_max": float(np.median(dr)), "steps_median": float(np.median([e["n_drift_steps"] for e in eps]))}
            r["half_s0_equal_c4"] = [sum(f["s0_equal_c4"] for f in fr), len(fr)]
            prg = [f for f in fr if f["pre_close"] and f["G_px"] > 0]
            for s_ in ("-17", "+17"):
                hs = [f["half" + s_] for f in prg]
                r["half" + s_] = {"out_of_view_frame_frac": round(float(np.mean([x["inview"] < CS.PARAMS["inview_thr"] for x in hs if x["inview"] is not None])), 4),
                                  "protected_frac_mean": round(float(np.nanmean([np.nan if x["prot"] is None else x["prot"] for x in hs])), 4),
                                  "on_arm_frac_mean": round(float(np.nanmean([np.nan if x["arm"] is None else x["arm"] for x in hs])), 4),
                                  "line_vanish_frame_frac": round(float(np.mean([x["vanish"] for x in hs])), 4), "n": len(hs)}
            ce = [f["half" + s_]["cshift_err"] for f in fr for s_ in ("-17", "+17") if "cshift_err" in f["half" + s_]]
            r["half_centroid_err_max"] = max(ce) if ce else None
            r["half_centroid_n"] = len(ce)
        r["c1_pass"] = r["c1_equal"][0] == r["c1_equal"][1]
        r["c2_pass"] = r["c2_frames_with_outT_diff"] == 0
        r["abandon_OB"] = not (r["c1_pass"] and r["c2_pass"] and r["c3_pass-69"] and r["c3_pass+69"])
        out[os.path.basename(bd)] = r
    json.dump(out, open(os.path.join(a.root, "OBCHECK.json"), "w"), indent=1)
    print(json.dumps(out, indent=1))


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
