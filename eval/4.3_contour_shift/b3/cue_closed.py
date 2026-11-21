"""WidowX 轮廓位置响应协议 v3.1（protocol_cue_widowx_v3_1_final.md §3、§4）的闭环采集：一个进程跑一个（块，格）的一批键。

评测循环与第一阶段 b2/eval_views.py 逐行相同（reset、预处理、种子、推理、后处理、执行前 4 步、截断即止）；
差别只有两处，都不改物理：
  1. 画面（§3 十三格）：C0、C3、C4 走第一阶段原函数 views.ViewMaker.make（与 b2/views.py 前 215 行逐字相同）；
     S(±h、±a、±b) 与 SC(±b) 走 views.ViewMaker3._draw_new 的 "S" / "SC" 分支，dxdy 模式，水平整数平移；
     OB(±b) 用 cue_obcheck.synth（§3 任务物图像平移合成六步，与冻结前放行条件实测用的是同一个函数）。
  2. 记录：§4 要求的逐步状态（T+1 行）与逐查询统计量（字段见 record_formats_v31，st_*、q_*）。
     逐步状态只读不写：连杆位姿、关节角、actor 位姿、agent.check_contact_fingers / check_grasp、gripper_closedness。
     C3、C4 为了算线的统计量另做一次「隐藏夹爪渲染」，只改可视性并立即恢复；C0 不另渲染（任务物掩膜取可见分割图）。
键（§4）：校准键（试次 9）随时可跑（冒烟）。其余键要求 b3/FROZEN_cue_widowx 存在、CODE_FILES 的 sha256 与清单一致，且：
  锚点键（清单 anchor_keysets）只许 C0、C3、C4 三格，输出根目录只能是 out_cue_anchor 或 out_cue_anchor_post；
  主键（清单 keysets）不许写进锚点目录；两种键集以外的非校准键一律拒绝。日志不打印成败与步数（盲采集）。
"""
import argparse
import hashlib
import json
import os
import socket
import subprocess
import sys
import time
import traceback

B3 = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, B3)
OPZ = os.path.expanduser("$PROJ_ROOT/bench2_simpler/open-pi-zero")
import numpy as np  # noqa: E402

import cue_spec as CS  # noqa: E402
import keys as K  # noqa: E402
import views as V  # noqa: E402

CODE_FILES = ("cue_closed.py", "views.py", "eval_views.py", "keys.py", "cue_spec.py", "cue_obcheck.py")
FROZEN = os.path.join(B3, "FROZEN_cue_widowx")


def sha_file(p):
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def project(p_world, intr, extr):
    pc = extr[:3, :3] @ np.asarray(p_world, float) + extr[:3, 3]
    uvw = intr @ pc
    return uvw[:2] / uvw[2]


def centroid(M):
    ys, xs = np.nonzero(M)
    return [float("nan"), float("nan")] if len(xs) == 0 else [float(xs.mean()), float(ys.mean())]


def pix_in(M, uv):
    """投影点（四舍五入到像素）是否落在掩膜里；掩膜空 -1，出画 0。"""
    if not M.any():
        return -1
    x, y = int(round(uv[0])), int(round(uv[1]))
    H, W = M.shape
    return int(0 <= x < W and 0 <= y < H and bool(M[y, x]))


def edge_touch_any(M):
    return int(bool(M[:, 0].any() or M[:, -1].any() or M[0].any() or M[-1].any()))


def key_ok_bridge(k, block):
    """键的 reset 规格与 keys.py 的 Bridge 定义一致（episode_id 0..23、env_seed 由（块，episode）定）。键集本身是参数。"""
    ep = int(k["reset"]["episode_id"])
    return (k["block"] == block and 0 <= ep < K.BRIDGE_EPISODES and k["cfg_id"] == f"ep{ep:02d}"
            and k["cluster"] == f"ep{ep:02d}" and int(k["env_seed"]) == K.bridge_env_seed(block, ep)
            and k["key"] == f"{block}__ep{ep:02d}__t{int(k['trial'])}")


def check_manifest_for_keys(keys, block, code_files, cell, out_root, manifest=FROZEN):
    """返回清单 sha256（全是校准键时为空串）。规则见文件头。"""
    calib = {k["key"] for k in K.make_keys(block, "calib")}
    for k in keys:
        assert key_ok_bridge(k, block), f"键规格不符 {k['key']}"
    out_base = os.path.basename(os.path.normpath(out_root))
    if all(k["key"] in calib for k in keys):
        assert out_base not in CS.ANCHOR_DIRS, "锚点目录只放锚点键"
        return ""
    assert os.path.exists(manifest), "非校准键需要冻结清单 FROZEN_cue_widowx，现在拒绝运行"
    man = json.load(open(manifest))
    for f in code_files:
        assert man["code_sha256"].get(f) == sha_file(os.path.join(B3, f)), f"{f} 与冻结清单不符"
    sets = {}
    for kind in ("keysets", "anchor_keysets"):
        ks = man[kind][block]
        kp = os.path.join(B3, ks["path"])
        assert sha_file(kp) == ks["sha256"], f"{kind} 键集文件与冻结清单不符"
        sets[kind] = {k["key"] for k in json.load(open(kp))}
    for k in keys:
        if k["key"] in calib:
            assert out_base not in CS.ANCHOR_DIRS, "锚点目录只放锚点键"
        elif k["key"] in sets["anchor_keysets"]:
            assert cell in CS.ANCHOR_CELLS, f"锚点键只许跑 {CS.ANCHOR_CELLS}，拒绝 {cell}"
            assert out_base in CS.ANCHOR_DIRS, f"锚点键只许写进 {CS.ANCHOR_DIRS}，拒绝 {out_base}"
        elif k["key"] in sets["keysets"]:
            assert out_base not in CS.ANCHOR_DIRS, "主键不许写进锚点目录"
        else:
            raise AssertionError(f"键不在冻结的主键集或锚点键集里：{k['key']}")
    return sha_file(manifest)


def check_model(manifest=FROZEN):
    """协调方 11-14 复核建议改第 3 条：启动时核模型与清单 model 字段，不符拒跑（只在非校准键、清单存在时）。
    权重 11.8 GB，算 sha256 在计算节点约 40 秒，同卡 4 个进程同时读还会更慢，而每个（块，格）分片都要起一次进程；
    所以这里核「文件大小 + mtime_ns + open-pi-zero 提交号 + 两个配置/适配器文件的 sha256」，
    完整的权重 sha256 由冻结时在登录节点算一次写进清单（model.ckpt_sha256）。改权重而不改大小和 mtime 需要刻意为之。"""
    import subprocess as sp
    m = json.load(open(manifest))["model"]
    ck = os.path.realpath(os.path.expanduser(m["ckpt_path"]))
    st = os.stat(ck)
    assert st.st_size == int(m["ckpt_bytes"]), f"权重大小 {st.st_size} 与清单 {m['ckpt_bytes']} 不符"
    assert st.st_mtime_ns == int(m["ckpt_mtime_ns"]), f"权重 mtime_ns {st.st_mtime_ns} 与清单 {m['ckpt_mtime_ns']} 不符"
    head = sp.run(["git", "-C", OPZ, "rev-parse", "HEAD"], capture_output=True, text=True, timeout=30).stdout.strip()
    assert head == m["open_pi_zero_commit"], f"open-pi-zero 提交号 {head} 与清单 {m['open_pi_zero_commit']} 不符"
    assert sha_file(os.path.join(OPZ, "config/eval/bridge.yaml")) == m["open_pi_zero_cfg_bridge_yaml_sha256"], "bridge.yaml 与清单不符"
    assert sha_file(os.path.join(OPZ, "src/agent/env_adapter/simpler.py")) == m["open_pi_zero_env_adapter_simpler_py_sha256"], "适配器与清单不符"
    return ck


def main():
    from eval_views import reset_key, check_env_src, check_reset_spec, scene_state, jsonable
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=list(CS.BLOCKS))
    ap.add_argument("--cell", required=True, choices=CS.CELL_ORDER)
    ap.add_argument("--keys", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-done", action="store_true")
    a = ap.parse_args()

    blk = dict(K.BLOCKS[a.block])
    kind, shift = CS.CELLS[a.cell]
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    man_sha = check_manifest_for_keys(keys, a.block, CODE_FILES, a.cell, a.out)
    if man_sha:  # 非校准键：清单存在且已核过代码哈希
        ck_real = check_model()
        assert ck_real == os.path.realpath(os.path.expanduser(blk["ckpt"])), "keys.py 的权重路径与清单 model.ckpt_path 不符"
    odir = os.path.join(a.out, a.block, a.cell)
    edir = os.path.join(odir, "_errors")
    os.makedirs(edir, exist_ok=True)

    def done(k):
        p = os.path.join(odir, k["key"] + ".npz")
        if not os.path.exists(p):
            return False
        try:
            with np.load(p) as z:
                return bool(z["complete"])
        except Exception:
            return False
    todo = [k for k in keys if not (a.skip_done and done(k))]
    print(f"[cue] block {a.block} cell {a.cell} keys {len(keys)} todo {len(todo)}", flush=True)
    if not todo:
        return

    import torch
    import hydra
    from omegaconf import OmegaConf
    import simpler_env
    from cue_obcheck import synth
    check_env_src()
    cwd0 = os.getcwd()
    os.chdir(OPZ)
    sys.path.insert(0, OPZ)
    from src.model.vla.pizero import PiZeroInference
    cfg = OmegaConf.load(blk["cfg"])
    ckpt = os.path.expanduser(blk["ckpt"])
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)
    try:
        smi = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception as e:
        smi = f"nvidia-smi failed: {e}"
    t0 = time.time()
    model = PiZeroInference(cfg, use_ddp=False)
    data = torch.load(ckpt, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict({k.replace("_orig_mod.", ""): v for k, v in data["model"].items()}, strict=True)
    del data
    model.freeze_all_weights()
    model.to(dtype)
    model.to(dev)
    model.eval()
    print(f"[cue] model loaded {time.time() - t0:.1f}s gpu {gpu_name}", flush=True)
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    os.chdir(cwd0)
    act_steps = int(cfg.act_steps)
    assert act_steps == CS.QUERY_EVERY
    envs = {}
    code_sha = {f: sha_file(os.path.join(B3, f)) for f in CODE_FILES}

    def infer(inp, s):  # 与 b2/eval_views.infer 逐行相同
        cm, vp, pp, apos = model.build_causal_mask_and_position_ids(inp["attention_mask"], dtype=dtype)
        itpm, am = model.split_full_mask_into_submasks(cm)
        x = {"input_ids": inp["input_ids"], "pixel_values": inp["pixel_values"].to(dtype),
             "image_text_proprio_mask": itpm, "action_mask": am, "vlm_position_ids": vp,
             "proprio_position_ids": pp, "action_position_ids": apos, "proprios": inp["proprios"].to(dtype)}
        x = {kk: v.to(dev) for kk, v in x.items()}
        torch.manual_seed(s)
        torch.cuda.manual_seed_all(s)
        with torch.inference_mode():
            return model(**x)[0].float().cpu().numpy()

    for k in todo:
        te = time.time()
        try:
            if k["task"] not in envs:
                envs[k["task"]] = simpler_env.make(k["task"])
            env = envs[k["task"]]
            u = env.unwrapped
            cam = V.cam_name(u.robot_uid)
            max_steps = int(env.spec.max_episode_steps)
            ctrl_hz = float(u._control_freq)
            L = V.lag_for(act_steps, ctrl_hz)
            obs, info = reset_key(env, k)
            reset_info = json.dumps(jsonable(info))
            got_ep, got_urdf = check_reset_spec(env, info, k)
            adapter.reset()
            H, W = obs["image"][cam]["rgb"].shape[:2]
            lw = V.line_width_for(W)
            assert lw == CS.LINE_WIDTH and (W, H) == (CS.W_ENV, CS.H_ENV)
            base_cell = a.cell if a.cell in V.CELLS else "C3"
            vm = V.ViewMaker3(env, base_cell, blk["body"], L, lw, act_steps, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
            var = {"mode": "dxdy", "dxdy": list(shift)}
            vm.setup3(a.block, k["cfg_id"], k["trial"], var if kind in ("S", "SC") else {}, (), None, key=k["key"])
            if kind == "OB":
                vm.cell = "C3"  # 父类按 C3 构造；OB 不走 make
            vm.on_reset3()
            if kind != "C0":
                vm.check_restore(obs)
                assert vm.restore_maxdiff == 0, f"隐藏后恢复与原图不同 maxdiff {vm.restore_maxdiff}"
            st0 = scene_state(env)
            rgb0 = np.ascontiguousarray(obs["image"][cam]["rgb"])
            frame0_sha = hashlib.sha256(rgb0.tobytes()).hexdigest()
            acts = u.get_actors()
            anames = [x.name for x in acts]
            tobj = [x for x in acts if x.name == CS.TARGET[a.block]]
            assert len(tobj) == 1, CS.TARGET[a.block]
            tobj = tobj[0]
            src = getattr(u, "episode_source_obj", None)
            assert src is not None and src.name == tobj.name, "任务物与 episode_source_obj 不符"
            tid = np.array([tobj.id], np.int32)
            cont = [x for x in acts if x.name == CS.CONTAINER[a.block]]
            assert len(cont) == 1, CS.CONTAINER[a.block]
            cont = cont[0]
            cid = np.array([cont.id], np.int32)
            obj_vbs = list(tobj.get_visual_bodies())
            assert obj_vbs, "任务物没有可视体"
            ag = u.agent
            links = {l.name: l for l in ag.robot.get_links()}
            ft = [links[n] for n in V.FINGERTIP_LINKS[blk["body"]]]

            def snap():
                qp = np.asarray(ag.robot.get_qpos(), np.float64)
                c = ag.check_contact_fingers(tobj)
                return (np.concatenate([vm.tcp.pose.p, vm.tcp.pose.q]).astype(np.float64), qp[vm.fj_idx].copy(),
                        float(np.linalg.norm(np.asarray(ft[0].pose.p) - np.asarray(ft[1].pose.p))),
                        float(ag.gripper_closedness),
                        np.array([np.concatenate([x.pose.p, x.pose.q]) for x in acts], np.float64),
                        [bool(c[0]), bool(c[1])], bool(ag.check_grasp(tobj)))
            S = [snap()]
            instr0 = env.get_language_instruction()
            instr = instr0
            seeds, raw_actions, env_actions, qstats, succ_steps = [], [], [], [], []
            Q = {n: [] for n in ("step", "intr", "extr", "lc", "lc_cur", "gc", "gpx", "inview", "prot", "arm", "ring",
                                 "l224", "chg", "tcpG", "objM", "edge", "tcpB", "cov", "covpx",
                                 "ob_rate", "ob_off", "ob_arm", "ob_cont", "ob_cut", "ob_T")}
            step, q, truncated = 0, 0, False
            while True:
                rgb = obs["image"][cam]["rgb"]
                seg = obs["image"][cam]["Segmentation"][..., 1]
                cp = obs["camera_param"][cam]
                intr, extr = np.asarray(cp["intrinsic_cv"], float), np.asarray(cp["extrinsic_cv"], float)
                if kind in ("C0", "C3", "C4"):
                    img, stt = vm.make3(obs, step)  # make3 对 C0..C5 直接调用父类 make（第一阶段原函数）
                    G = vm.hist[int(step)]
                    seg_h = None
                    if kind == "C4":
                        _, seg_h = vm.render_hidden()
                else:
                    # 与 ViewMaker3.make3 新格分支相同，只是留下画线区与分割图做统计
                    G = np.isin(seg, vm.gids)
                    vm.hist[int(step)] = G.copy()
                    vm.kin[int(step)] = vm._kin_now()
                    vm.R_cur = np.isin(seg, vm.robot_ids)
                    stt = {"step": int(step), "G_px": int(G.sum())}
                    img_h, seg_h = vm.render_hidden()
                    stt["G_vis_after_hide_px"] = int(np.isin(seg_h, vm.gids).sum())
                    if kind == "OB":
                        # §3 任务物图像平移合成：cue_obcheck.synth（底图 = 夹爪与任务物都隐藏，只改可视性，取图后立即恢复）
                        P_full = np.isin(seg_h, vm.pids)
                        T_ob = np.isin(seg_h, tid)
                        robot_h = np.isin(seg_h, vm.robot_ids)
                        O_ob = V.outline(G, lw)
                        vm._set_vis(0.0)
                        for vb in obj_vbs:
                            vb.set_visibility(0.0)
                        try:
                            base = V.rgb_of(u.get_obs()["image"][cam])
                        finally:
                            vm._set_vis(1.0)
                            for vb in obj_vbs:
                                vb.set_visibility(1.0)
                        dx = int(shift[0])
                        img, pasted = synth(base, img_h, seg_h, T_ob, robot_h, P_full, O_ob, dx)
                        img_pre, _ = synth(base, img_h, seg_h, T_ob, robot_h, P_full, np.zeros_like(O_ob), dx)  # 贴图后、画线前
                        P_ob = (P_full & ~T_ob) | pasted
                        draw = O_ob & ~P_ob
                        Ts = V.shift_mask(T_ob, dx, 0)
                        nT = int(T_ob.sum())
                        stt.update(line_px_224=vm._vis224(img, img_pre), O_px=int(O_ob.sum()), O_drawn_px=int(draw.sum()),
                                   O_protected_px=int((O_ob & P_ob).sum()), ob_T_px=nT, ob_pasted_px=int(pasted.sum()))
                        E = G | draw | T_ob | pasted
                    else:
                        img, draw = vm._draw_new(kind, vm.var, G, step, img_h, seg_h, stt)
                        E = G | draw
                    diff = np.any(img != rgb, axis=2)
                    stt["changed_px"] = int(diff.sum())
                    stt["changed_out_E_px"] = int((diff & ~E).sum())
                # ---- 逐查询统计量（只读，不改 img）
                T_m = np.isin(seg if seg_h is None else seg_h, tid)
                lc = lc_cur = [float("nan")] * 2
                inview = prot = arm = float("nan")
                ring, l224, edge = -1, -1, 0
                cov, covpx = float("nan"), -1
                ob = dict(rate=float("nan"), off=float("nan"), arm=float("nan"), cont=-1, cut=float("nan"), T=-1)
                obj_mask, obj_pt = T_m, None
                if kind == "OB":
                    R_vis = np.isin(seg, vm.robot_ids)
                    lc_cur = centroid(draw)
                    lc = lc_cur
                    inview = 1.0 if O_ob.any() else float("nan")
                    prot = float((O_ob & P_ob).sum()) / O_ob.sum() if O_ob.any() else float("nan")
                    arm = float((draw & R_vis & ~G).sum()) / draw.sum() if draw.any() else float("nan")
                    l224 = int(stt["line_px_224"])
                    ringm = V._dilate_sq(pasted, CS.RING_PX) & ~pasted  # 解读：外圈围着策略看到的（贴上去的）任务物
                    ring = int((draw & ringm).sum())
                    edge = edge_touch_any(draw)
                    if nT:
                        ob = dict(rate=float(pasted.sum()) / nT, off=1.0 - float(Ts.sum()) / nT,
                                  arm=float((Ts & robot_h).sum()) / nT, cont=int((pasted & np.isin(seg_h, cid)).sum()),
                                  cut=float((O_ob & pasted).sum()) / O_ob.sum() if O_ob.any() else float("nan"), T=nT)
                    else:
                        ob["T"] = 0
                    obj_mask, obj_pt = pasted, np.array([float(shift[0]), 0.0])
                if kind in ("C4", "S", "SC"):
                    P_full = np.isin(seg_h, vm.pids)
                    O_cur = V.outline(G, lw)
                    D_cur = O_cur & ~P_full
                    lc_cur = centroid(D_cur)
                    if kind == "C4":
                        draw_line = D_cur
                        inview = 1.0 if O_cur.any() else float("nan")
                        prot = float((O_cur & P_full).sum()) / O_cur.sum() if O_cur.any() else float("nan")
                        R_cur = np.isin(seg, vm.robot_ids)
                        arm = float((D_cur & R_cur & ~G).sum()) / D_cur.sum() if D_cur.any() else float("nan")
                        l224 = int(stt["line_px_224"])
                        draw_all = D_cur
                    else:
                        Os = V.shift_mask(O_cur, shift[0], shift[1])
                        draw_line = Os & ~P_full
                        inview = stt.get("frac_inview")
                        prot = stt.get("frac_protected")
                        if kind == "SC":
                            cov = float(stt["copy_overlap_with_cur"]) if stt.get("copy_overlap_with_cur") is not None else float("nan")
                            covpx = int((Os & O_cur).sum())
                        arm = stt.get("frac_on_arm")
                        l224 = int(stt["line_px_224"])
                        draw_all = draw
                    inview = float("nan") if inview is None else float(inview)
                    prot = float("nan") if prot is None else float(prot)
                    arm = float("nan") if arm is None else float(arm)
                    lc = centroid(draw_line)
                    ringm = V._dilate_sq(T_m, CS.RING_PX) & ~T_m
                    ring = int((draw_line & ringm).sum())
                    edge = edge_touch_any(draw_all)
                tcp_uv = project(vm.tcp.pose.p, intr, extr)
                obj_uv = project(tobj.pose.p, intr, extr)
                gg = V.mask_geom(G)
                tcp_bb = -1 if gg is None else int(gg["x0"] <= tcp_uv[0] < gg["x1"] and gg["y0"] <= tcp_uv[1] < gg["y1"])
                for nm, val in (("step", step), ("intr", intr), ("extr", extr), ("lc", lc), ("lc_cur", lc_cur),
                                ("gc", centroid(G)), ("gpx", int(G.sum())), ("inview", inview), ("prot", prot),
                                ("arm", arm), ("ring", ring), ("l224", l224), ("chg", int(stt["changed_out_E_px"])),
                                ("tcpG", pix_in(V._dilate_sq(G, CS.PROJ_DIL_PX), tcp_uv)),
                                ("objM", pix_in(obj_mask, obj_uv if obj_pt is None else obj_uv + obj_pt)), ("edge", edge),
                                ("tcpB", tcp_bb), ("cov", cov), ("covpx", covpx), ("ob_rate", ob["rate"]), ("ob_off", ob["off"]),
                                ("ob_arm", ob["arm"]), ("ob_cont", ob["cont"]), ("ob_cut", ob["cut"]), ("ob_T", ob["T"])):
                    Q[nm].append(val)
                pobs = dict(obs)
                pobs["image"] = {cam: {"rgb": img}}
                s = K.noise_seed(a.block, k["cfg_id"], k["trial"], q)
                seeds.append(s)
                inp = adapter.preprocess(env, pobs, instr)
                ra = infer(inp, s)
                raw_actions.append(ra)
                qstats.append(stt)
                for ea in adapter.postprocess(ra)[:act_steps]:
                    obs, r, success, truncated, info = env.step(ea)
                    succ_steps.append(bool(success))
                    env_actions.append(np.asarray(ea, np.float64))
                    S.append(snap())
                    step += 1
                    if truncated:
                        break
                new = env.get_language_instruction()
                if new != instr:
                    instr = new
                q += 1
                if truncated:
                    break
            ea_arr = np.array(env_actions, np.float64)
            g = ea_arr[:, -1]
            fc = int(np.nonzero(g < 0)[0][0]) if (g < 0).any() else -1
            st_succ = np.array([-1] + [int(x) for x in succ_steps], np.int8)
            rec = dict(
                complete=np.bool_(True), block=a.block, cell=a.cell, cell_kind=kind, shift_env=np.array(shift, np.int32),
                key=k["key"], cfg_id=k["cfg_id"], cluster=k["cluster"], task=k["task"], trial=np.int32(k["trial"]),
                env_seed=np.int64(k["env_seed"]), reset_spec=json.dumps(k["reset"]), robot_uid=u.robot_uid,
                actual_episode_id=np.int32(got_ep), actual_urdf_version=got_urdf,
                n_queries=np.int32(q), seeds=np.array(seeds, np.uint64), raw_actions=np.array(raw_actions, np.float32),
                env_actions=ea_arr, instruction=instr0, instruction_final=instr, qstats=json.dumps(qstats),
                gpu_name=gpu_name, nvidia_smi=smi, host=socket.gethostname(), lag_steps=np.int32(L), line_width=np.int32(lw),
                query_every=np.int32(act_steps), max_episode_steps=np.int32(max_steps), control_freq=np.float64(ctrl_hz),
                img_hw=np.array([H, W], np.int32), g_links=json.dumps([l.name for l in vm.g_links]), g_ids=vm.gids,
                p_names=json.dumps(vm.p_names), restore_maxdiff=np.int32(-1 if vm.restore_maxdiff is None else vm.restore_maxdiff),
                reset_info=reset_info, code_sha=json.dumps(code_sha), manifest_sha256=man_sha,
                frame0_sha256=frame0_sha, vk_icd_filenames=os.environ.get("VK_ICD_FILENAMES", ""),
                ckpt=ckpt, dtype="bf16", torch_compile=np.bool_(False),
                success=np.bool_(succ_steps[-1]), success_ever=np.bool_(any(succ_steps)),
                success_steps=np.array(succ_steps, np.bool_), steps=np.int32(step),
                success_steps_sha256=hashlib.sha256(np.array(succ_steps, np.bool_).tobytes()).hexdigest(),
                first_close_step=np.int32(fc), task_obj_name=tobj.name, task_obj_index=np.int32(anames.index(tobj.name)),
                robot_base_pose=np.concatenate([ag.robot.pose.p, ag.robot.pose.q]).astype(np.float64),
                st_tcp_pose=np.array([x[0] for x in S]), st_finger_q=np.array([x[1] for x in S]),
                st_grip_width=np.array([x[2] for x in S]), st_closedness=np.array([x[3] for x in S]),
                st_actor_poses=np.array([x[4] for x in S]), st_actor_names=json.dumps(anames),
                st_contact=np.array([x[5] for x in S], np.bool_), st_grasp=np.array([x[6] for x in S], np.bool_),
                st_success=st_succ,
                q_step=np.array(Q["step"], np.int32), q_intrinsic=np.array(Q["intr"]), q_extrinsic=np.array(Q["extr"]),
                q_line_centroid=np.array(Q["lc"], np.float64), q_line_centroid_cur=np.array(Q["lc_cur"], np.float64),
                q_G_centroid=np.array(Q["gc"], np.float64), q_G_px=np.array(Q["gpx"], np.int32),
                q_frac_inview=np.array(Q["inview"], np.float64), q_frac_protected=np.array(Q["prot"], np.float64),
                q_frac_on_arm=np.array(Q["arm"], np.float64), q_ring_px=np.array(Q["ring"], np.int32),
                q_line_px_224=np.array(Q["l224"], np.int32), q_changed_out_E_px=np.array(Q["chg"], np.int32),
                q_tcp_in_G6=np.array(Q["tcpG"], np.int8), q_obj_on_mask=np.array(Q["objM"], np.int8),
                q_edge_touch=np.array(Q["edge"], np.int8), q_tcp_in_Gbbox=np.array(Q["tcpB"], np.int8),
                q_copy_overlap=np.array(Q["cov"], np.float64), q_copy_overlap_px=np.array(Q["covpx"], np.int32),
                q_ob_paste_rate=np.array(Q["ob_rate"], np.float64), q_ob_lost_offscreen=np.array(Q["ob_off"], np.float64),
                q_ob_lost_arm=np.array(Q["ob_arm"], np.float64), q_ob_on_container_px=np.array(Q["ob_cont"], np.int32),
                q_ob_line_cut_frac=np.array(Q["ob_cut"], np.float64), q_ob_T_px=np.array(Q["ob_T"], np.int32),
                st_obj_disp_xy=np.linalg.norm(np.array([x[4] for x in S])[:, anames.index(tobj.name), :2]
                                              - S[0][4][anames.index(tobj.name), :2], axis=1),
                container_name=cont.name, container_index=np.int32(anames.index(cont.name)),
                sec=np.float64(time.time() - te), **st0)
            assert len(S) == step + 1 and len(Q["step"]) == q
            tmp = os.path.join(odir, f".{k['key']}.tmp.npz")
            np.savez_compressed(tmp, **rec)
            os.replace(tmp, os.path.join(odir, k["key"] + ".npz"))
            ep = os.path.join(edir, k["key"] + ".txt")
            if os.path.exists(ep):
                os.replace(ep, ep + ".resolved")
            print(f"[cue] done {k['key']} {time.time() - te:.1f}s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            with open(os.path.join(edir, k["key"] + ".txt"), "w") as f:
                f.write(tb)
            print(f"[cue] ERROR {k['key']} (see _errors)", flush=True)
            print(tb, file=sys.stderr, flush=True)
    print(f"[cue] finished peak_vram_gb {torch.cuda.max_memory_reserved() / 1024 ** 3:.2f}", flush=True)


if __name__ == "__main__":
    main()
