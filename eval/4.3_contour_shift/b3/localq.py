"""局部响应检验的采集工具（协议 protocol_cue_local.md v0，未冻结）。
同一状态只把画回的轮廓画偏，向策略各查询一次；不执行这些查询的动作，轨迹按状态来源记录的动作原样回放。

状态来源：<src>/<块>/<格>/<键>.npz（正式运行为 b2/out_formal/<块>/C4，只读；冒烟为本目录 eval_views 在校准键上跑出的 C4）。
每个键：
  1. reset（与第一阶段同一函数），按记录的 env_actions 逐步回放；
  2. 在每个查询时刻 t（t % 查询间隔 == 0）：
     - 零平移画面用第一阶段原函数 views.ViewMaker.make（C4）生成，其余画面用 ViewMaker3._draw_new（S 平移、SC 副本），
       都在同一张擦除底图上画线，保护名单截断同第一阶段；
     - 每个画面以记录里该次查询的噪声种子、同一份本体状态输入，向策略查询一次，存原始输出整块与后处理后将被执行的前
       act_steps 步（平移、旋转、夹爪）；
     - 存相机内外参、末端参考点（ee_gripper_link，即控制器的 ee_link）与目标物的世界坐标、它们及「末端 + 指令平移和」
       在策略输入图（横乘 224/W、竖乘 224/H）上的投影；画面的视野内比例、被保护截掉比例、与目标物外扩 6 像素一圈相交的
       线像素数、线在输入里的可见像素数；该查询是否在记录里第一次闭合指令之前（含当步）；
     - 另存该查询之后实际执行 act_steps 步的末端位移（只用零平移的记录动作），用来核对指令平移的坐标系。
  3. 核对一：回放的逐步成功标志与记录相同（记录带 success_steps 时直接比，盲记录比 success_steps_sha256；本工具不打印成败）。
     核对二：零平移画面的原始输出与记录 raw_actions[q] 逐位相同，种子与记录 seeds[q] 相同。
输出：<out>/<块>/<键>.npz（数值数组）与 <键>.json（逐查询元数据）。不存画面；--sheet-queries 时另存拼好的小图。
正式键需要 b3/FROZEN_localq 存在才允许运行（现在不存在）。
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

import keys as K  # noqa: E402
import views as V  # noqa: E402

# 目标物（与目标物外扩一圈相交的线像素数；投影）：每块写死
TARGET = {"B1": "eggplant", "B2": "bridge_spoon_generated_modified"}
RING_PX = 6
# 默认画面（协议 §1；幅度在策略输入尺度上定义）：名字 -> (类型, 轴, 输入像素)
DEFAULT_VIEWS = [("zero", "zero", None, 0), ("h-b", "S", "h", -24), ("h-a", "S", "h", -12), ("h+a", "S", "h", 12),
                 ("h+b", "S", "h", 24), ("v-b", "S", "v", -24), ("v+b", "S", "v", 24), ("copy-b", "SC", "h", -24),
                 ("copy+b", "SC", "h", 24)]


def env_px(axis, s_in, W, H):
    """输入尺度的平移换回环境像素：水平乘 W/224，竖直乘 H/224，四舍五入。"""
    return int(round(s_in * (W if axis == "h" else H) / 224.0))


def project(p_world, intr, extr):
    pc = extr[:3, :3] @ np.asarray(p_world, float) + extr[:3, 3]
    uvw = intr @ pc
    return uvw[:2] / uvw[2]


def quat2mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def bool_sha(a):
    return hashlib.sha256(np.asarray(a, np.bool_).tobytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--block", required=True, choices=["B1", "B2"])
    ap.add_argument("--keys", required=True)
    ap.add_argument("--src", required=True, help="状态来源根目录")
    ap.add_argument("--src-cell", default="C4")
    ap.add_argument("--out", required=True)
    ap.add_argument("--views", default="", help="JSON [[名字, 类型 zero|S|SC, 轴 h|v, 输入像素], ...]；默认协议 §1 的九个")
    ap.add_argument("--queries", default="all", choices=["all", "preclose"],
                    help="all：全部查询照采九个画面（协议 §1：其余照采只作描述）；preclose：闭合后只查零平移")
    ap.add_argument("--skip-done", action="store_true")
    ap.add_argument("--sheet-queries", default="")
    a = ap.parse_args()
    views = [tuple(x) for x in json.loads(a.views)] if a.views else DEFAULT_VIEWS
    assert views[0][1] == "zero", "第一个画面必须是零平移"
    blk = dict(K.BLOCKS[a.block])
    keys = [k for k in json.load(open(a.keys)) if k["block"] == a.block]
    formal = {k["key"] for k in K.make_keys(a.block, "formal")}
    if any(k["key"] in formal for k in keys):
        assert os.path.exists(os.path.join(B3, "FROZEN_localq")), "正式键需要冻结清单 FROZEN_localq，现在拒绝运行"
    odir = os.path.join(a.out, a.block)
    edir = os.path.join(odir, "_errors")
    os.makedirs(edir, exist_ok=True)
    todo = [k for k in keys if not (a.skip_done and os.path.exists(os.path.join(odir, k["key"] + ".npz")))]
    print(f"[lq] {a.block} keys {len(keys)} todo {len(todo)} views {[v[0] for v in views]}", flush=True)
    if not todo:
        return
    import torch
    import hydra
    from omegaconf import OmegaConf
    import simpler_env
    from eval_views import reset_key, check_env_src, check_reset_spec
    check_env_src()
    cwd0 = os.getcwd()
    os.chdir(OPZ)
    sys.path.insert(0, OPZ)
    from src.model.vla.pizero import PiZeroInference
    cfg = OmegaConf.load(blk["cfg"])
    dev, dtype = torch.device("cuda:0"), torch.bfloat16
    gpu_name = torch.cuda.get_device_name(0)
    ckpt = os.path.expanduser(blk["ckpt"])
    model = PiZeroInference(cfg, use_ddp=False)
    data = torch.load(ckpt, weights_only=True, map_location="cpu", mmap=True)
    model.load_state_dict({kk.replace("_orig_mod.", ""): v for kk, v in data["model"].items()}, strict=True)
    del data
    model.freeze_all_weights()
    model.to(dtype)
    model.to(dev)
    model.eval()
    adapter = hydra.utils.instantiate(cfg.env.adapter)
    os.chdir(cwd0)
    act_steps = int(cfg.act_steps)
    code_sha = {f: hashlib.sha256(open(os.path.join(B3, f), "rb").read()).hexdigest()
                for f in ("localq.py", "views.py", "eval_views.py", "keys.py")}

    def infer(inp, s):  # 与 eval_views.infer 逐行相同
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

    envs = {}
    sheet_q = {int(x) for x in a.sheet_queries.split(",") if x.strip()}
    for k in todo:
        te = time.time()
        try:
            z = np.load(os.path.join(a.src, a.block, a.src_cell, k["key"] + ".npz"))
            rec_env_actions, rec_raw, rec_seeds = z["env_actions"], z["raw_actions"], z["seeds"]
            assert str(z["cell"]) == a.src_cell and str(z["key"]) == k["key"]
            if "success_steps" in z.files:
                rec_succ_sha = bool_sha(z["success_steps"])
            else:
                rec_succ_sha = str(z["success_steps_sha256"]) if "success_steps_sha256" in z.files else None
            if k["task"] not in envs:
                envs[k["task"]] = simpler_env.make(k["task"])
            env = envs[k["task"]]
            u = env.unwrapped
            cam = V.cam_name(u.robot_uid)
            obs, info = reset_key(env, k)
            check_reset_spec(env, info, k)
            adapter.reset()
            assert np.array_equal(np.asarray(u.agent.robot.get_qpos(), np.float64), z["init_robot_qpos"]), "reset 初始状态与记录不同"
            H, W = obs["image"][cam]["rgb"].shape[:2]
            L, lw = int(z["lag_steps"]), int(z["line_width"])
            vm = V.ViewMaker3(env, "C4", blk["body"], L, lw, act_steps, K.PROTECT[a.block], K.PROTECT_EXPECT_N[a.block])
            vm.setup3(a.block, k["cfg_id"], k["trial"], {}, (), None, key=k["key"])
            vm.on_reset3()
            vm.check_restore(obs)
            assert vm.restore_maxdiff == 0
            tgt = [x for x in u.get_actors() if x.name == TARGET[a.block]]
            assert len(tgt) == 1, TARGET[a.block]
            tgt = tgt[0]
            g = np.asarray(rec_env_actions[:, -1])
            c_idx = int(np.nonzero(g < 0)[0][0]) if (g < 0).any() else None  # WidowX：-1 为闭合（REPORT_bench2_step0 第二部分）
            instr = env.get_language_instruction()
            n_q = int(len(rec_raw))
            nv = len(views)
            raw_out = np.zeros((n_q, nv, rec_raw.shape[1], rec_raw.shape[2]), np.float32)
            exe = np.zeros((n_q, nv, act_steps, 7), np.float64)
            meta, succ, tcp_hist = [], [], {}
            zero_eq = []
            step, q = 0, 0
            while step < len(rec_env_actions):
                assert step % act_steps == 0
                s = int(rec_seeds[q])
                assert s == K.noise_seed(a.block, k["cfg_id"], k["trial"], q), "记录种子与公式不符"
                img0, st0 = vm.make(obs, step)  # 第一阶段的 C4 原函数
                G = vm.hist[int(step)]
                vm.R_cur = np.isin(obs["image"][cam]["Segmentation"][..., 1], vm.robot_ids)
                img_h, seg_h = vm.render_hidden()
                T = np.isin(seg_h, np.array([tgt.id], np.int32))
                ring = V._dilate_sq(T, RING_PX) & ~T
                cp = obs["camera_param"][cam]
                intr, extr = np.asarray(cp["intrinsic_cv"], float), np.asarray(cp["extrinsic_cv"], float)
                tcp = np.asarray(vm.tcp.pose.p, float)
                root = u.agent.robot.pose
                Rb = quat2mat(np.asarray(root.q, float))
                sx, sy = 224.0 / W, 224.0 / H
                m = {"q": q, "step": step, "seed": s, "pre_close": bool(c_idx is None or step <= c_idx), "first_close_step": c_idx,
                     "intrinsic_cv": intr.tolist(), "extrinsic_cv": extr.tolist(), "tcp_world": tcp.tolist(),
                     "root_q": np.asarray(root.q, float).tolist(), "target_world": np.asarray(tgt.pose.p, float).tolist(),
                     "G_px": int(G.sum()), "views": {}}
                uv = project(tcp, intr, extr)
                m["tcp_uv_in"] = [uv[0] * sx, uv[1] * sy]
                ut = project(tgt.pose.p, intr, extr)
                m["target_uv_in"] = [ut[0] * sx, ut[1] * sy]
                gg = V.mask_geom(G)
                m["G_centroid_in"] = None if gg is None else [gg["cx"] * sx, gg["cy"] * sy]
                pobs = dict(obs)
                inp0 = None
                # --queries preclose：闭合之后的查询只查零平移（仍做核对二），其余画面不查
                run_views = views if (a.queries == "all" or m["pre_close"]) else views[:1]
                m["views_run"] = [v[0] for v in run_views]
                for vi, (name, typ, axis, s_in) in enumerate(run_views):
                    d = {"type": typ, "axis": axis, "s_in": s_in}
                    if typ == "zero":
                        img = img0
                        O = V.outline(G, lw)
                        draw = O & ~np.isin(seg_h, vm.pids)
                        d.update(dxdy=[0, 0], frac_inview=1.0 if O.any() else None,
                                 frac_protected=None if not O.any() else float(st0.get("O_protected_px", 0)) / float(O.sum()))
                    else:
                        dx = env_px("h", s_in, W, H) if axis == "h" else 0
                        dy = env_px("v", s_in, W, H) if axis == "v" else 0
                        stt = {}
                        img, draw = vm._draw_new(typ, {"mode": "dxdy", "dxdy": [dx, dy]}, G, step, img_h, seg_h, stt)
                        d.update(dxdy=[dx, dy], frac_inview=stt.get("frac_inview"), frac_protected=stt.get("frac_protected"),
                                 frac_on_arm=stt.get("frac_on_arm"))
                    d["line_ring_px"] = int((draw & ring).sum())
                    d["line_px_224"] = vm._vis224(img, img_h)
                    pobs["image"] = {cam: {"rgb": img}}
                    inp = adapter.preprocess(env, pobs, instr)
                    if inp0 is None:
                        inp0 = inp
                    else:  # 本体状态输入各画面相同
                        assert torch.equal(inp["proprios"], inp0["proprios"]) and torch.equal(inp["input_ids"], inp0["input_ids"])
                    ra = infer(inp, s)
                    raw_out[q, vi] = ra
                    ea = np.asarray(adapter.postprocess(ra)[:act_steps], np.float64)
                    exe[q, vi, :len(ea)] = ea
                    dt = ea[:, :3].sum(0)  # 平移增量之和（控制器坐标）
                    dw = Rb @ dt  # 换到世界坐标（控制器按基座朝向的轴，见 REPORT_bench2_step0 第二部分；投影用世界坐标）
                    up = project(tcp + dw, intr, extr)
                    d["r_in"] = [(up[0] - uv[0]) * sx, (up[1] - uv[1]) * sy]
                    d["dt_ctrl_m"] = dt.tolist()
                    d["dw_world_m"] = dw.tolist()
                    d["gripper"] = ea[:, -1].tolist()
                    m["views"][name] = d
                    if q in sheet_q:
                        m.setdefault("_imgs", []).append((f"{name} r=({d['r_in'][0]:.1f},{d['r_in'][1]:.1f})", img))
                zero_eq.append(bool(np.array_equal(raw_out[q, 0], rec_raw[q])))
                m["zero_bit_equal"] = zero_eq[-1]
                nst = len(rec_env_actions[step:step + act_steps])
                m["zero_exec_equal_record"] = bool(np.array_equal(exe[q, 0, :nst], rec_env_actions[step:step + nst]))
                if "_imgs" in m:
                    from eval_views import save_sheet
                    save_sheet(os.path.join(odir, "sheets", f"{k['key']}_q{q:03d}"), m.pop("_imgs"), ncol=3)
                tcp_hist[step] = tcp
                for i in range(act_steps):
                    if step >= len(rec_env_actions):
                        break
                    obs, r, success, truncated, info = env.step(rec_env_actions[step])
                    succ.append(bool(success))
                    step += 1
                m["tcp_after_exec_world"] = np.asarray(vm.tcp.pose.p, float).tolist()  # 实际执行记录动作后的末端
                meta.append(m)
                q += 1
            check1 = (rec_succ_sha is not None) and bool_sha(succ) == rec_succ_sha
            rec = dict(key=k["key"], block=a.block, cfg_id=k["cfg_id"], cluster=k["cluster"], trial=np.int32(k["trial"]),
                       src=os.path.abspath(os.path.join(a.src, a.block, a.src_cell)), src_code_sha=str(z["code_sha"]),
                       views=json.dumps(views), raw_out=raw_out, exec_actions=exe, rec_raw=rec_raw, n_queries=np.int32(n_q),
                       check1_success_steps_equal=np.bool_(check1), check1_available=np.bool_(rec_succ_sha is not None),
                       check2_zero_bit_equal_all=np.bool_(all(zero_eq)), check2_n_equal=np.int32(sum(zero_eq)),
                       img_hw=np.array([H, W], np.int32), act_steps=np.int32(act_steps), first_close_step=np.int32(-1 if c_idx is None else c_idx),
                       gpu_name=gpu_name, host=socket.gethostname(), code_sha=json.dumps(code_sha),
                       vk_icd_filenames=os.environ.get("VK_ICD_FILENAMES", ""), sec=np.float64(time.time() - te))
            np.savez_compressed(os.path.join(odir, f".{k['key']}.tmp.npz"), **rec)
            os.replace(os.path.join(odir, f".{k['key']}.tmp.npz"), os.path.join(odir, k["key"] + ".npz"))
            json.dump(meta, open(os.path.join(odir, k["key"] + ".json"), "w"))
            print(f"[lq] done {k['key']} queries {n_q} check1 {check1} check2 {sum(zero_eq)}/{n_q} {time.time() - te:.1f}s", flush=True)
        except Exception:
            tb = traceback.format_exc()
            open(os.path.join(edir, k["key"] + ".txt"), "w").write(tb)
            print(f"[lq] ERROR {k['key']}", flush=True)
            print(tb, file=sys.stderr, flush=True)


if __name__ == "__main__":
    main()
