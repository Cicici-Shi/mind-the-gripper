"""cue_readout.py（连同它内部调用的 cue_gate）的合成数据测试（§17 第 13 条）。只写临时目录里的合成记录（真实记录格式的 npz），
不碰真实目录。运行：python3 test_cue_readout.py

合成几何：相机在世界 z = 1 处朝下（外参 R = diag(1,-1,-1)，t = [0,0,1]），fx = fy = 500，cx = 320，cy = 240；任务物与末端都在 z = 0，
于是输入图 u = 175·x + 112（世界 x 每 1/175 米 = 1 个输入像素），每输入像素 0.5714 cm。
逐键：任务物第 0 行在原点；末端第 0 行在 x = −0.05；第 t >= 1 行末端相对任务物原点的水平图像偏移 = 符号 × s̃ × (g(t) + σ·n_键)，
g(t) 分段：第 1 到 4 行 q1，第 5 到 8 行 q2，第 9 到 13 行 ref，第 14 行起 tc（默认闭合步 14、t_ref 10）。闭合后任务物按 kb 平移。
n_键 是对称格点的一个固定排列（中位数恰为 0），全部格共用，所以同档 O(x) 的噪声逐键相同。"""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import cue_gate as G  # noqa: E402
import cue_readout as R  # noqa: E402

PX = 175.0
T, QE, TREF, TC = 24, 4, 10, 14
NQ = T // QE
NK = 96
ST = {"a": 11.9, "b": 24.15, "h": 5.95}
CELLS = G.CELL_ORDER
SHIFT = {"C0": 0, "C3": 0, "C4": 0, "S-b": -69, "S-a": -34, "S+a": 34, "S+b": 69, "SC-b": -69, "SC+b": 69,
         "OB-b": -69, "OB+b": 69, "S-h": -17, "S+h": 17}
KIND = {c: (c if c in ("C0", "C3", "C4") else c.split("-")[0].split("+")[0]) for c in CELLS}
GROUP = {"S-b": "b", "S+b": "b", "S-a": "a", "S+a": "a", "S-h": "h", "S+h": "h", "SC-b": "SC", "SC+b": "SC", "OB-b": "OB", "OB+b": "OB"}
SIGN = {c: (1 if "+" in c else -1) for c in GROUP}
STG = {"a": ST["a"], "b": ST["b"], "h": ST["h"], "SC": ST["b"], "OB": ST["b"]}
# 清单 code_sha256：读数会逐个核登记的文件（相对 b3），所以合成清单只登记这两个，取测试时的真实哈希（改坏副本里也自洽）
CODE = {f: hashlib.sha256(open(os.path.join(os.path.dirname(os.path.abspath(R.__file__)), f), "rb").read()).hexdigest()
        for f in ("cue_readout.py", "cue_gate.py")}
GPU = "NVIDIA A100-PCIE-40GB"
ICD = "/usr/share/vulkan/icd.d/nvidia_icd.json"
K_INT = np.array([[500.0, 0, 320], [0, 500.0, 240], [0, 0, 1]])
EXT = np.array([[1.0, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 1], [0, 0, 0, 1]])
NOISE = np.random.RandomState(7).permutation(np.linspace(-np.sqrt(3), np.sqrt(3), NK))
ROWS_SEEN = set()
CASES_SEEN = set()


def ep_of(i):
    return i % 24


def CL(n):
    """前 n 个 episode 的键为 1，其余 0（按簇变化，区间宽）。"""
    return lambda i: 1 if ep_of(i) < n else 0


def NOTCL(n):
    return lambda i: 0 if ep_of(i) < n else 1


def first_n(n, val, other):
    """键序号 < n 的取 val（键按外层试次、内层 episode 排，前 24 个是试次 5 的全部 episode）。"""
    return lambda i: val if i < n else other


def V(f, *a):
    return f(*a) if callable(f) else f


DEF_SUCC = {"C0": 1, "C3": 0, "C4": 1, "S-b": 1, "S-a": 1, "S+a": 1, "S+b": 1, "SC-b": 1, "SC+b": 1, "OB-b": 0, "OB+b": 0, "S-h": 1, "S+h": 1}
DEF_KH = {"a": dict(q1=-0.5, q2=-0.8, ref=-0.9, tc=-1.0), "b": dict(q1=-0.5, q2=-0.8, ref=-0.9, tc=-1.0),
          "h": dict(q1=-0.5, q2=-0.8, ref=-0.9, tc=-1.0),
          "SC": dict(q1=-0.25, q2=-0.4, ref=-0.45, tc=-ST["a"] / ST["b"]),   # 默认 δ_off 恰为 0 偏
          "OB": dict(q1=0.2, q2=0.3, ref=0.4, tc=0.5)}


def bspec(**kw):
    s = dict(succ={}, tc={}, kh={}, kb={}, sigma=0.1, gcmd=-0.4, contact_from={}, q={}, rec={}, pre_disp={})
    for k, v in kw.items():
        s[k] = v
    return s


def kh_of(spec, g):
    d = dict(DEF_KH[g])
    d.update(spec["kh"].get(g, {}))
    return d


def seg(d, t):
    if t == 0:
        return 0.0
    if t <= 4:
        return d["q1"]
    if t <= 8:
        return d["q2"]
    if t <= 13:
        return d["ref"]
    return d["tc"]


def make_keys(block):
    return [dict(key=f"{block}__ep{e:02d}__t{t}", block=block, task="widowx_task", cfg_id=f"ep{e:02d}", trial=t,
                 env_seed=1000 * e + t, reset={"episode_id": e}, cluster=f"ep{e:02d}") for t in (5, 6, 7, 8) for e in range(24)]


def write_json(p, obj):
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, ensure_ascii=False, indent=1)
    return hashlib.sha256(open(p, "rb").read()).hexdigest()


def deep_update(d, u):
    for k, v in u.items():
        if isinstance(v, dict) and isinstance(d.get(k), dict):
            deep_update(d[k], v)
        else:
            d[k] = v


def record(block, cell, i, kd, spec, man_sha):
    g = GROUP.get(cell)
    sg = SIGN.get(cell, 0)
    tc = int(V(spec["tc"].get(cell, TC), i))
    succ = int(V(spec["succ"].get(cell, DEF_SUCC[cell]), i))
    sigma = spec["sigma"]
    tcp = np.zeros((T + 1, 7))
    tcp[:, 3] = 1.0
    obj = np.zeros((T + 1, 7))
    obj[:, 3] = 1.0
    kb = spec["kb"].get(g, 0.0) if g else 0.0
    for t in range(T + 1):
        dev = 0.0
        if g:
            d = kh_of(spec, g)
            dev = sg * STG[g] * (seg(d, t) + (sigma * NOISE[i] if t >= 1 else 0.0))
        ox = sg * STG[g] * kb / PX if (g and tc >= 0 and t >= tc) else 0.0
        pd = spec["pre_disp"].get(cell)
        if pd is not None and t >= pd[0]:
            ox += pd[1]                       # 接触前任务物自己漂了（米，不碰手指）
        obj[t, 0] = ox
        tcp[t, 0] = -0.05 if t == 0 else dev / PX
    contact = np.zeros((T + 1, 2), bool)
    cf = V(spec["contact_from"].get(cell, (tc + 1) if tc >= 0 else None), i)
    if cf is not None:
        contact[int(cf):] = True
    grasp = np.zeros(T + 1, bool)
    if tc >= 0:
        grasp[tc + 2:] = True
    disp = np.abs(obj[:, 0] - obj[0, 0])
    ea = np.zeros((T, 7))
    ea[:, 6] = 1.0
    if tc >= 0:
        ea[tc:, 6] = -1.0
    if g in ("a", "b", "h"):
        ea[:QE, 0] = sg * spec["gcmd"] * ST[g] / PX / QE
    else:
        ea[:QE, 0] = {"C0": 0.001, "C3": 0.004, "C4": 0.002}.get(cell, 0.0005)
    is_line = cell not in ("C0", "C3")
    nanq = np.full(NQ, np.nan)
    negq = -np.ones(NQ)
    qf = dict(
        q_step=np.arange(NQ) * QE, q_intrinsic=np.repeat(K_INT[None], NQ, 0), q_extrinsic=np.repeat(EXT[None], NQ, 0),
        q_line_centroid=np.zeros((NQ, 2)), q_line_centroid_cur=np.zeros((NQ, 2)), q_G_centroid=np.zeros((NQ, 2)),
        q_G_px=np.full(NQ, 100), q_frac_inview=np.ones(NQ) if is_line else nanq,
        q_frac_protected=np.full(NQ, 0.05) if is_line else nanq, q_frac_on_arm=np.full(NQ, 0.03) if is_line else nanq,
        q_ring_px=np.array([0, 0, 3, 3, 3, 3]) if is_line else negq, q_line_px_224=np.full(NQ, 50) if is_line else negq,
        q_changed_out_E_px=np.zeros(NQ), q_tcp_in_G6=np.ones(NQ, np.int8), q_tcp_in_Gbbox=np.ones(NQ, np.int8),
        q_obj_on_mask=np.ones(NQ, np.int8), q_edge_touch=np.zeros(NQ, np.int8),
        q_copy_overlap=np.full(NQ, 0.6) if g == "SC" else nanq, q_copy_overlap_px=np.full(NQ, 40) if g == "SC" else negq,
        q_ob_paste_rate=np.ones(NQ) if g == "OB" else nanq, q_ob_lost_offscreen=np.zeros(NQ) if g == "OB" else nanq,
        q_ob_lost_arm=np.full(NQ, 0.01) if g == "OB" else nanq, q_ob_on_container_px=np.zeros(NQ) if g == "OB" else negq,
        q_ob_line_cut_frac=np.zeros(NQ) if g == "OB" else nanq, q_ob_T_px=np.full(NQ, 300) if g == "OB" else negq)
    for f, v in spec["q"].get(cell, {}).items():
        qf[f] = np.array([V(v, i, q) for q in range(NQ)], dtype=np.asarray(qf[f]).dtype)
    ss = np.zeros(T, bool)
    ss[-1] = bool(succ)
    st_succ = np.zeros(T + 1, np.int8)
    st_succ[0] = -1
    st_succ[-1] = succ
    names = ["eggplant", "basket"]
    r = dict(complete=True, block=block, cell=cell, cell_kind=KIND[cell], key=kd["key"], cfg_id=kd["cfg_id"], cluster=kd["cluster"],
             task=kd["task"], trial=kd["trial"], env_seed=kd["env_seed"], code_sha=json.dumps(CODE), manifest_sha256=man_sha,
             shift_env=np.array([SHIFT[cell], 0], np.int32), line_width=8, lag_steps=0, query_every=QE, max_episode_steps=T,
             control_freq=5, gpu_name=GPU, host="node1", vk_icd_filenames=ICD,
             seeds=np.array([1000 * i + q for q in range(NQ)], np.uint64), raw_actions=np.zeros((NQ, 4, 7), np.float32),
             env_actions=ea, frame0_sha256=f"f0_{kd['key']}", init_robot_qpos=np.arange(8.0), init_robot_pose=np.zeros(7),
             init_actor_poses=obj[0:1].repeat(2, 0), init_actor_names=json.dumps(names), robot_base_pose=np.array([0, 0, 0, 1.0, 0, 0, 0]),
             n_queries=NQ, steps=T, success=bool(succ), success_ever=bool(succ), success_steps=ss, first_close_step=np.int32(tc),
             task_obj_name="eggplant", task_obj_index=0, container_name="basket", container_index=1,
             st_tcp_pose=tcp, st_finger_q=np.zeros((T + 1, 2)), st_grip_width=np.zeros(T + 1), st_closedness=np.zeros(T + 1),
             st_actor_poses=np.stack([obj, np.zeros((T + 1, 7))], 1), st_actor_names=json.dumps(names), st_contact=contact,
             st_grasp=grasp, st_success=st_succ, st_obj_disp_xy=disp, **qf)
    for f, v in spec["rec"].get((cell, i), {}).items():
        r[f] = v
    return r


def build(tmp, specs, man_over=None, B=200, anchor=True, init=True, init_over=None, anchor_over=None, stage1=True):
    """写一套两块的合成数据。返回 (root, manifest_path)。采后的两份比对文件的修改时间设成晚于全部记录（§4 的先后）。"""
    root = os.path.join(tmp, "out_cue_closed")
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "SYNTHETIC_DATA"), "w").write("synthetic\n")
    os.makedirs(os.path.join(root, "formal_state"), exist_ok=True)
    if stage1:
        st = {b: {c: {"n_ref": 10, "equal": 10, "differ": [], "ref_missing": [], "missing": [],
                      "fields": ["raw_actions", "env_actions", "frame0_sha256"]} for c in G.ANCHOR_CELLS} for b in R.BLOCKS}
        if isinstance(stage1, dict):
            deep_update(st, stage1)
        write_json(os.path.join(root, "formal_state", "STAGE1_OK"), st)
    keys = {b: make_keys(b) for b in R.BLOCKS}
    man = {"protocol_sha256": R.PROTOCOL_SHA256, "code_sha256": dict(CODE), "keysets": {}, "anchor_keysets": {},
           "cells": {c: {"kind": KIND[c], "shift_env": [SHIFT[c], 0]} for c in CELLS}, "cell_order": list(CELLS),
           "anchor_cells": ["C0", "C3", "C4"], "tref_file": "tref_cue_widowx_v31.json",
           "s_tilde": dict(ST),
           "params": {"sr_thr_pts": 10, "min_keys": 77, "inview_thr": 0.8, "paste_loss_thr": 0.2, "flag_frac": 0.05,
                      "precontact_disp_m": 0.01, "ring_px": 6, "kappa_q": [1, 2], "curve_q": [1, 2, 3, 4, 5],
                      "bootstrap": {"B": B, "seed": 0, "ci": [2.5, 97.5]},
                      "localq_q0": {b: {"a": {"est": -0.3, "ci": [-0.4, -0.2]}, "b": {"est": -0.3, "ci": [-0.4, -0.2]}} for b in R.BLOCKS}},
           "reachability": {}, "line_width": 8, "query_every": QE, "vk_icd_filenames": ICD,
           "gpu_whitelist": ["NVIDIA A100-PCIE-40GB"]}
    for b in R.BLOCKS:
        h = write_json(os.path.join(tmp, "keysets", f"cue_{b}.json"), keys[b])
        man["keysets"][b] = {"path": f"keysets/cue_{b}.json", "sha256": h, "n": NK}
        ah = write_json(os.path.join(tmp, "keysets", f"anchor_{b}.json"), [{"key": f"{b}__ep{e:02d}__t0"} for e in range(10)])
        man["anchor_keysets"][b] = {"path": f"keysets/anchor_{b}.json", "sha256": ah, "n": 10}
    man["tref_sha256"] = write_json(os.path.join(tmp, "tref_cue_widowx_v31.json"), {b: {k["key"]: TREF for k in keys[b]} for b in R.BLOCKS})
    if man_over:
        deep_update(man, man_over)
    mp = os.path.join(tmp, "SYNTHETIC_MANIFEST.json")
    man_sha = write_json(mp, man)
    for b in R.BLOCKS:
        sp = specs.get(b, bspec())
        for c in CELLS:
            os.makedirs(os.path.join(root, b, c), exist_ok=True)
            for i, kd in enumerate(keys[b]):
                np.savez(os.path.join(root, b, c, kd["key"] + ".npz"), **record(b, c, i, kd, sp, man_sha))
    if anchor:
        a = {"ok": True, "per": {b: {c: {"n_ref": 10, "equal": 10, "differ": [], "ref_missing": [], "missing": [],
                                         "fields": ["raw_actions", "env_actions", "frame0_sha256"]} for c in G.ANCHOR_CELLS} for b in R.BLOCKS}}
        if anchor_over:
            deep_update(a, anchor_over)
        write_json(os.path.join(tmp, "out_cue_anchor_post", "ANCHOR_POST.json"), a)
    if init:
        d = {"ok_frame0": {b: True for b in R.BLOCKS}, "ok_init": {b: True for b in R.BLOCKS},
             "per": {b: {c: {"n": NK, "frame0_equal": NK, "init_equal": NK, "differ_frame0": [], "differ_init": [], "missing": []}
                         for c in CELLS} for b in R.BLOCKS}}
        if init_over:
            deep_update(d, init_over)
        write_json(os.path.join(root, "INIT_CMP.json"), d)
    later(os.path.join(tmp, "out_cue_anchor_post", "ANCHOR_POST.json"), os.path.join(root, "INIT_CMP.json"))
    return root, mp


def later(*paths, dt=60.0):
    """把文件的修改时间设成现在之后 dt 秒（晚于刚写完的全部记录）；dt 为负则早于。"""
    t = time.time() + dt
    for p in paths:
        if os.path.exists(p):
            os.utime(p, (t, t))


def run(root, mp, *extra):
    return R.main(["--root", root, "--manifest", mp] + list(extra))


def rows(res, b):
    return res["blocks"][b]["decision"]["rows"]


def sent(res, b, g):
    s = res["blocks"][b]["decision"]["sent"].get(g)
    return None if s is None else s["write"] + " || " + " || ".join(s["modifiers"])


def wording_ok(tc, res):
    """§9：可写句、修饰语、合成句里不出现禁用词。"""
    for b in R.BLOCKS:
        dec = res["blocks"][b]["decision"]
        for g, s in dec["sent"].items():
            ROWS_SEEN.add(s["row"])
            for w in R.FORBIDDEN_IN_WRITE:
                tc.assertNotIn(w, s["write"], f"{b} {s['row']} 可写句含禁用词 {w}")
                for m in s["modifiers"]:
                    tc.assertNotIn(w, m, f"{b} {s['row']} 修饰语含禁用词 {w}")
        for v in dec.get("synthesis", {}).values():
            for x in (v if isinstance(v, list) else [v]):
                for w in R.FORBIDDEN_IN_WRITE:
                    if x:
                        tc.assertNotIn(w, x, f"{b} 合成句含禁用词 {w}")


class Base(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="cue_v31_test_")

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def go(self, specs, *extra, **kw):
        root, mp = build(self.tmp, specs, **kw)
        res = run(root, mp, *extra)
        wording_ok(self, res)
        return res


# ====================================================================== 判定表各行与第 13 条情形
class TestRows(Base):
    def test_default_R3_K1_A1_P1_and_R0_block(self):
        """B1：R3，F 只照报，K1（剂量为负、闭合时刻不对称），A1 并 K1 的附句，P1。B2：R0，其余各组不落行。"""
        res = self.go({"B1": bspec(tc={"S+b": 16}), "B2": bspec(succ={"C4": 0})})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual(r1["R"], "R3")
        self.assertIsNone(r1["F"])
        self.assertIn("R 行是 R3：副本格只照报三个配对差与 δ_off，不接任何句子", res["blocks"]["B1"]["decision"]["notes"])
        self.assertIsNone(r1["H"])
        self.assertEqual(r1["K"], "K1")
        self.assertIn("偏移随幅度增大", sent(res, "B1", "K"))
        self.assertIn("两档都有", sent(res, "B1", "K"))
        self.assertIn("闭合时刻不对称：t_c(+b) − t_c(−b) 中位数 +2.0 步", sent(res, "B1", "K"))
        self.assertIn("简化递推模型的粗略预测并排", sent(res, "B1", "K"))
        self.assertEqual(r1["A"], "A1")
        self.assertIn("两种平移引起方向相反的偏移", sent(res, "B1", "A"))
        self.assertEqual(r1["P"], "P1")
        self.assertEqual(r2["R"], "R0")
        self.assertTrue(all(r2[g] is None for g in "HFKAP"))
        self.assertIn(R.NUMBERS_ONLY["r0"], res["blocks"]["B2"]["decision"]["notes"])
        CASES_SEEN.update({"R0 其余不落行", "R3 时 F 只照报"})

    def test_R1_H1_F2_K2_A2_P2__R1_H2_F1_K3_A3_P3(self):
        b1 = bspec(succ={"S-a": 0, "S+a": 0, "S-h": 0, "S+h": 0, "S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 0},
                   kb={"b": -1.0}, kh={"OB": dict(q1=0, q2=0, ref=0, tc=0), "b": dict(q2=0.0)})
        b2 = bspec(succ={"S-a": 0, "S+a": 0, "S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 1},
                   kh={"b": dict(q1=0.0, tc=0.0), "OB": dict(tc=-0.5)}, kb={"b": 1.0})
        res = self.go({"B1": b1, "B2": b2})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual((r1["R"], r1["H"], r1["F"], r1["K"], r1["A"], r1["P"]), ("R1", "H1", "F2", "K2", "A2", "P2"))
        self.assertIn("这半句只成上界", sent(res, "B1", "R"))
        self.assertIn("这个比较没有分辨力", sent(res, "B1", "F"))
        self.assertIn("物体在闭合前朝同一方向被移动", sent(res, "B1", "K"))
        self.assertIn("终点偏移可能经由闭合时刻或推动物体形成", sent(res, "B1", "P"))
        syn = res["blocks"]["B1"]["decision"]["synthesis"]
        self.assertTrue(syn["丙"].startswith("平移轮廓削弱恢复"))
        self.assertIn(R.F2_CAVEAT, syn["丙"])
        self.assertEqual((r2["R"], r2["H"], r2["F"], r2["K"], r2["A"], r2["P"]), ("R1", "H2", "F1", "K3", "A3", "P3"))
        f = sent(res, "B2", "F")
        self.assertIn(R.MOD["F1_DSRB"], f)
        self.assertIn("留着正确轮廓，多画一条线后恢复也下降", f)
        self.assertIn("这部分偏移来自物体被移动", sent(res, "B2", "K"))
        self.assertIn("下降出现在 6 到 12 输入像素之间", sent(res, "B2", "H"))
        CASES_SEEN.add("H 只在 R1 落行（R1 一侧）")

    def test_R1_mods_and_F1_DF_not(self):
        """R1：大档不是降低时写「未见随幅度加重」；F1：D_F 下界高于 −10 时写未达到预设的 10 点。R2 时 H 不落行。"""
        b1 = bspec(succ={"S-a": 0, "S+a": 0, "SC-b": 1, "SC+b": 1, "S-b": NOTCL(4), "S+b": 1})
        b2 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 1, "SC+b": 1})
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual(rows(res, "B1")["R"], "R1")
        self.assertIn("未见随幅度加重", sent(res, "B1", "R"))
        self.assertIn("方向差 SR(S(+b)) − SR(S(−b)) = +16.7", sent(res, "B1", "R"))
        r2 = rows(res, "B2")
        self.assertEqual((r2["R"], r2["F"], r2["H"]), ("R2", "F1", None))
        self.assertIn("留着正确轮廓时的下降未达到预设的 10 点", sent(res, "B2", "F"))
        CASES_SEEN.add("H 只在 R1 落行（R2 不落）")

    def test_F3_doff_zero_and_positive__K4_with_A1_and_A2(self):
        b1 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 0}, kh={"b": dict(tc=0.0)})
        b2 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 0}, kh={"SC": dict(tc=0.0), "b": dict(tc=0.0),
                                                                          "OB": dict(tc=0.0)})
        res = self.go({"B1": b1, "B2": b2})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual((r1["R"], r1["F"], r1["K"], r1["A"]), ("R2", "F3", "K4", "A1"))
        self.assertIn("没有判出副本格里手被带得更远", sent(res, "B1", "F"))
        self.assertIn("含 0", sent(res, "B1", "F"))
        self.assertIn(R.MOD["K4_A1"], sent(res, "B1", "K"))
        self.assertEqual((r2["R"], r2["F"], r2["K"], r2["A"]), ("R2", "F3", "K4", "A2"))
        self.assertIn("为正", sent(res, "B2", "F"))
        self.assertIn(R.MOD["K4_NOT_A1"], sent(res, "B2", "K"))
        syn = res["blocks"]["B2"]["decision"]["synthesis"]
        self.assertTrue(syn["丙"].startswith("平移轮廓削弱恢复"))
        self.assertIn(R.SELECTIVE_F3, syn["选择性证据"])
        CASES_SEEN.update({"F3 δ_off 含 0", "F3 δ_off 为正", "A2 时 K4 的附句"})

    def test_F3_doff_na__F3_with_F1_kept__K5__K6_neg(self):
        b1 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 0}, tc={"SC+b": first_n(25, -1, TC)}, kh={"b": dict(tc=1.0)})
        b2 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 1}, kb={"b": -2.0})
        res = self.go({"B1": b1, "B2": b2})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual((r1["R"], r1["F"], r1["K"]), ("R2", "F3", "K5"))
        self.assertIn("四格都闭合的键不足（71 < 77），δ_off 算不出", sent(res, "B1", "F"))
        self.assertEqual((r2["R"], r2["F"], r2["K"]), ("R2", "F3", "K6"))
        self.assertIn("同时：画面里同样有一条偏移 24 像素的线，留着正确轮廓时恢复更高", sent(res, "B2", "F"))
        self.assertIn("两项发现都保留", sent(res, "B2", "F"))
        CASES_SEEN.update({"F3 δ_off 算不出", "F1 与 F3 同时满足落 F3", "K6 手项为负"})

    def test_F_generic_modifier__K6_zero__F4_natural__H3__K0_n(self):
        b1 = bspec(succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 1}, kh={"SC": dict(tc=-1.0), "b": dict(tc=0.0)}, kb={"b": -1.0})
        b2 = bspec(succ={"S-a": 0, "S+a": 0, "S-h": NOTCL(2), "S+h": NOTCL(2)},
                   tc={"S+a": first_n(20, -1, TC), "S+b": first_n(20, -1, TC)})
        res = self.go({"B1": b1, "B2": b2})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual((r1["R"], r1["F"], r1["K"]), ("R2", "F1", "K6"))
        self.assertIn("副本比平移小档更低，同时手被带得更远", sent(res, "B1", "F"))
        self.assertEqual((r2["R"], r2["H"], r2["F"], r2["K"]), ("R1", "H3", "F4", "K0"))
        self.assertIn("成对样本 76 键 < 77", sent(res, "B2", "K"))
        CASES_SEEN.update({"F 通用修饰语", "K6 手项含 0", "K0 来由：键数"})

    def test_R4_and_R_order__F_report_only_R4__K_ref_modifier(self):
        """B2：D_SR(a)「降低」与 D_SR(b) 升高同时满足，按 R0、R4、R1 的顺序落 R4。"""
        b1 = bspec(succ={"C4": CL(16)}, kh={"b": dict(ref=0.0)})
        b2 = bspec(succ={"C4": CL(16), "S-a": 0, "S+a": 0})
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual(rows(res, "B1")["R"], "R4")
        self.assertIsNone(rows(res, "B1")["F"])
        self.assertIn("R 行是 R4：副本格只照报三个配对差与 δ_off，不接任何句子", res["blocks"]["B1"]["decision"]["notes"])
        self.assertEqual(rows(res, "B1")["K"], "K1")
        self.assertIn("固定参考时刻未见闭合处那样的偏移", sent(res, "B1", "K"))
        est2 = res["blocks"]["B2"]["estimate"]["sr"]
        self.assertEqual(R.lv_down(est2["D_SR_a"], 10), "降低")      # R1 的条件也满足
        self.assertEqual(rows(res, "B2")["R"], "R4")
        CASES_SEEN.update({"R 组落行顺序", "R4 时 F 只照报"})

    def test_R5__K0_flag__P_none(self):
        b1 = bspec(succ={"S-a": NOTCL(2), "S+a": NOTCL(2)})
        fl = {"q_frac_inview": 0.5}
        b2 = bspec(q={"S-a": fl, "S+a": fl, "S-b": fl, "S+b": fl})
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual(rows(res, "B1")["R"], "R5")
        self.assertIsNone(rows(res, "B1")["F"])
        self.assertIn("R 行是 R5：副本格只照报三个配对差与 δ_off，不接任何句子", res["blocks"]["B1"]["decision"]["notes"])
        r2 = rows(res, "B2")
        self.assertEqual(r2["K"], "K0")
        self.assertIn("被标「出视野」", sent(res, "B2", "K"))
        self.assertIsNone(r2["P"])
        self.assertTrue(any("替补之后 κ_1 仍不可判" in n for n in res["blocks"]["B2"]["decision"]["notes"]))
        CASES_SEEN.update({"R5 时 F 只照报", "K0 来由：标记"})

    def test_P_substitution__kappa2_na__K_a_substitution__A0_n(self):
        b1 = bspec(contact_from={"S+b": 6})
        b2 = bspec(contact_from={"S+b": 6, "S+a": 6}, tc={"S+b": first_n(20, -1, TC), "OB+b": first_n(20, -1, TC)})
        res = self.go({"B1": b1, "B2": b2})
        d1, d2 = res["blocks"]["B1"]["decision"], res["blocks"]["B2"]["decision"]
        self.assertEqual(rows(res, "B1")["P"], "P1")
        self.assertEqual(d1["P_tier"], "a")
        self.assertIn("整组换成 a 档", sent(res, "B1", "P"))
        self.assertIn("κ_1(a)", sent(res, "B1", "P"))
        self.assertEqual(rows(res, "B2")["P"], "P2")
        self.assertEqual(d2["P_tier"], "b")
        self.assertIn("视同 κ_2 不为负", sent(res, "B2", "P"))
        self.assertIn("合格键 0 < 77", sent(res, "B2", "P"))
        self.assertEqual(rows(res, "B2")["K"], "K1")
        self.assertEqual(d2["K_tier"], "a")
        self.assertIn("b 档不可判（成对样本 76 键 < 77），各行条件里的 b 换成 a", sent(res, "B2", "K"))
        self.assertIn("（12 输入像素档）", sent(res, "B2", "K"))
        self.assertEqual(rows(res, "B2")["A"], "A0")
        self.assertIn("成对样本 76 键 < 77", sent(res, "B2", "A"))
        CASES_SEEN.update({"P 先替补再判", "κ_2 不可判视同不为负", "K a 档替补", "A0 来由：键数"})

    def test_H3_flag__A0_flag__F4_sc_flag__negative_boundary(self):
        """B2 的 σ = 0：k^H_b 的区间恰为 [0, 0]，上界恰为 0 不算为负，所以落 K3 而不是 K1。"""
        fl = {"q_frac_inview": 0.5}
        b1 = bspec(succ={"S-a": 0, "S+a": 0}, q={"S-h": fl, "S+h": fl, "OB+b": {"q_ob_lost_offscreen": 0.5}})
        b2 = bspec(succ={"S-a": 0, "S+a": 0}, q={"SC+b": {"q_line_px_224": 0}}, sigma=0.0,
                   kh={"b": dict(tc=0.0)}, kb={"b": 1.0})
        res = self.go({"B1": b1, "B2": b2})
        r1, r2 = rows(res, "B1"), rows(res, "B2")
        self.assertEqual((r1["R"], r1["H"], r1["A"]), ("R1", "H3", "A0"))
        self.assertIn("S±h 被标「出视野」", sent(res, "B1", "H"))
        self.assertIn("任务物平移格被标「合成不足」", sent(res, "B1", "A"))
        self.assertEqual((r2["R"], r2["F"]), ("R1", "F4"))
        self.assertIn("副本格被标记（「轮廓操纵不足」）", sent(res, "B2", "F"))
        kH = res["blocks"]["B2"]["estimate"]["K"]["b"]["kH"]
        self.assertEqual((kH["lo"], kH["hi"]), (0.0, 0.0))
        self.assertEqual(r2["K"], "K3")
        CASES_SEEN.update({"半档被标记落 H3", "A0 来由：合成不足", "「为负」边界"})


# ====================================================================== 处置、护栏、门槛、bootstrap、参数
class TestGuards(Base):
    def test_second_read_refused(self):
        root, mp = build(self.tmp, {})
        run(root, mp)
        txt = os.path.join(root, R.OUT_NAME + ".txt")
        before = open(txt, "rb").read()
        with self.assertRaises(R.Refuse):
            run(root, mp)
        self.assertEqual(open(txt, "rb").read(), before)
        CASES_SEEN.add("拒绝第二次读数")

    def test_keyset_hash_mismatch_refused(self):
        root, mp = build(self.tmp, {})
        kp = os.path.join(self.tmp, "keysets", "cue_B1.json")
        open(kp, "a").write(" ")
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("sha256", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(root, R.OUT_NAME + ".txt")))
        CASES_SEEN.add("键清单哈希不符拒读")

    def test_protocol_hash_and_real_root_guards(self):
        root, mp = build(self.tmp, {}, man_over={"protocol_sha256": "5af593e2e78f2807f93d40e2a7c0c45cbb281a998a66b78880c96a99eb9f1aa0"})
        with self.assertRaises(R.Refuse) as cm:          # 旧协议哈希（§18 之前）也拒
            run(root, mp)
        self.assertIn("protocol_sha256", str(cm.exception))
        self.assertEqual(R.PROTOCOL_SHA256, "a1ce8fdd34ec855c129591105db694e61bd5150fee48bdb35f8a5930f4ba7e2d")
        os.remove(os.path.join(root, "SYNTHETIC_DATA"))
        with self.assertRaises(R.Refuse):
            run(root, mp)

    def test_incomplete_refused(self):
        root, mp = build(self.tmp, {})
        k = make_keys("B1")[3]["key"]
        os.remove(os.path.join(root, "B1", "S+a", k + ".npz"))
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("未采完", str(cm.exception))
        os.makedirs(os.path.join(root, "B1", "S+a", "_errors"))
        open(os.path.join(root, "B1", "S+a", "_errors", k + ".txt"), "w").write("boom")
        write_json(os.path.join(root, "formal_state", "submits.json"), {"B1": {"S+a": {k: 1}}})
        with self.assertRaises(R.Refuse):
            run(root, mp)

    def test_anchor_and_init_disposition(self):
        root, mp = build(self.tmp, {"B1": bspec(q={"C4": {"q_line_px_224": 0}})}, anchor=False,
                         init_over={"ok_init": {"B2": False}})
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("锚点", str(cm.exception))
        write_json(os.path.join(self.tmp, "out_cue_anchor_post", "ANCHOR_POST.json"),
                   {"ok": True, "per": {b: {c: {"n_ref": 10, "equal": 10, "differ": [], "ref_missing": [], "missing": []}
                                            for c in G.ANCHOR_CELLS} for b in R.BLOCKS}})
        later(os.path.join(self.tmp, "out_cue_anchor_post", "ANCHOR_POST.json"))
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("B2 新键初始比对不过", str(cm.exception))
        res = run(root, mp, "--investigated-unresolved")
        d1, d2 = res["blocks"]["B1"]["decision"], res["blocks"]["B2"]["decision"]
        self.assertTrue(d1["numbers_only"] and d2["numbers_only"])
        self.assertIn(R.NUMBERS_ONLY["c4"], d1["notes"])
        self.assertNotIn(R.NUMBERS_ONLY["init"], d1["notes"])
        self.assertIn(R.NUMBERS_ONLY["init"], d2["notes"])
        self.assertTrue(all(v is None for v in d1["rows"].values()))

    def test_anchor_unresolved_numbers_only(self):
        root, mp = build(self.tmp, {}, anchor_over={"ok": False})
        with self.assertRaises(R.Refuse):
            run(root, mp)
        res = run(root, mp, "--investigated-unresolved")
        for b in R.BLOCKS:
            self.assertIn(R.NUMBERS_ONLY["anchor"], res["blocks"][b]["decision"]["notes"])
            self.assertTrue(all(v is None for v in rows(res, b).values()))

    def test_gate_hard_items_exclude_whole_key(self):
        """硬门槛 1 到 4 各触发一次：该键十三格一起剔；读数只用剩下的键（簇大小不等），bootstrap 逐键核对。"""
        keys = make_keys("B1")
        bad = {"code": 1, "init": 30, "seed": 55, "white": 60, "gpu": 70, "gpu2": 80, "errrec": 90}
        rec = {("C3", bad["code"]): {"code_sha": json.dumps({"cue_closed.py": "z" * 64})},
               ("S+a", bad["init"]): {"init_robot_qpos": np.arange(8.0) + 1},
               ("OB-b", bad["seed"]): {"seeds": np.arange(NQ, dtype=np.uint64) + 999},
               ("C0", bad["gpu"]): {"gpu_name": "NVIDIA A100 MIG 1g.5gb"},
               ("SC+b", bad["gpu2"]): {"gpu_name": "NVIDIA A100-PCIE-80GB"}}
        for c in CELLS:                                   # 十三格同一张卡、不是 MIG，只是不在白名单
            rec[(c, bad["white"])] = {"gpu_name": "NVIDIA A100-PCIE-80GB"}
        b1 = bspec(rec=rec, succ={"S-b": CL(9), "S+b": CL(15)})
        root, mp = build(self.tmp, {"B1": b1})
        k_err = keys[bad["errrec"]]["key"]
        os.makedirs(os.path.join(root, "B1", "S-h", "_errors"))
        open(os.path.join(root, "B1", "S-h", "_errors", k_err + ".txt"), "w").write("boom")
        k_miss = make_keys("B2")[5]["key"]
        os.remove(os.path.join(root, "B2", "OB+b", k_miss + ".npz"))
        os.makedirs(os.path.join(root, "B2", "OB+b", "_errors"))
        open(os.path.join(root, "B2", "OB+b", "_errors", k_miss + ".txt"), "w").write("boom")
        write_json(os.path.join(root, "formal_state", "submits.json"), {"B2": {"OB+b": {k_miss: 2}}})
        res = run(root, mp)
        g1 = res["blocks"]["B1"]["gate"]
        want = sorted(keys[i]["key"] for i in bad.values())
        self.assertEqual(g1["excluded_keys"], want)
        reasons = g1["fail_reasons"]
        self.assertTrue(any(x.startswith("1:C3") for x in reasons[keys[bad["code"]]["key"]]))
        self.assertTrue(any(x.startswith("2:S+a") for x in reasons[keys[bad["init"]]["key"]]))
        self.assertTrue(any(x.startswith("3:OB-b") for x in reasons[keys[bad["seed"]]["key"]]))
        self.assertTrue(any(x.startswith("4:C0") for x in reasons[keys[bad["gpu"]]["key"]]))
        self.assertTrue(any(x.startswith("4:各格显卡型号不同") for x in reasons[keys[bad["gpu2"]]["key"]]))
        rw = reasons[keys[bad["white"]]["key"]]
        self.assertTrue(rw and all("不在白名单" in x for x in rw), rw)
        self.assertTrue(any(x.startswith("1:S-h") for x in reasons[k_err]))
        self.assertEqual(res["blocks"]["B2"]["gate"]["excluded_keys"], [k_miss])
        est = res["blocks"]["B1"]["estimate"]
        self.assertEqual(est["n_kept"], NK - len(bad))
        # ---- 逐键显式重采样核对 D_SR(b) 与 k_b（剔键后簇大小不等）
        kept = [k for k in keys if k["key"] not in set(want)]
        idx = [keys.index(k) for k in kept]
        clus = [k["cluster"] for k in kept]
        self.assertGreater(len(set(np.unique(clus, return_counts=True)[1])), 1)
        sb = lambda c: np.array([V(b1["succ"].get(c, DEF_SUCC[c]), i) for i in idx], float)  # noqa: E731
        dsr = ((sb("S+b") + sb("S-b")) / 2 - sb("C4")) * 100
        lo, hi = explicit(dsr, clus, 200, 0, np.mean)
        e = est["sr"]["D_SR_b"]
        self.assertAlmostEqual(e["point"], float(dsr.mean()), 9)
        self.assertAlmostEqual(e["lo"], lo, 9)
        self.assertAlmostEqual(e["hi"], hi, 9)
        ob = []
        for k in kept:
            u = {}
            for c in ("S+b", "S-b"):
                z = np.load(os.path.join(root, "B1", c, k["key"] + ".npz"))
                t = int(z["first_close_step"])
                u[c] = (z["st_tcp_pose"][t, 0] - z["st_actor_poses"][t, 0, 0]) * PX
            ob.append((u["S+b"] - u["S-b"]) / 2 / ST["b"])
        ob = np.array(ob)
        lo, hi = explicit(ob, clus, 200, 0, np.median)
        k = est["K"]["b"]["k"]
        self.assertAlmostEqual(k["point"], float(np.median(ob)), 9)
        self.assertAlmostEqual(k["lo"], lo, 9)
        self.assertAlmostEqual(k["hi"], hi, 9)
        CASES_SEEN.update({"门槛硬项按键剔", "bootstrap 逐键核对（剔键后）"})

    def test_params_read_from_manifest(self):
        """清单里改 10 点线、77 键与 s̃_b：落行与增益跟着变（不是脚本里写死的数）。"""
        b1 = bspec(succ={"C4": CL(12)})
        res = self.go({"B1": b1, "B2": bspec()}, man_over={"params": {"sr_thr_pts": 60, "min_keys": 97},
                                                            "s_tilde": {"b": 2 * ST["b"]}})
        sr1 = res["blocks"]["B1"]["estimate"]["sr"]["Rec0"]
        self.assertAlmostEqual(sr1["point"], 50.0, 9)
        self.assertGreater(sr1["lo"], 0)
        self.assertEqual(rows(res, "B1")["R"], "R0")           # 10 点线时是「提高」，60 点线时不是
        self.assertEqual(rows(res, "B2")["K"], "K0")
        self.assertEqual(rows(res, "B2")["A"], "A0")
        self.assertIn("< 97", sent(res, "B2", "K"))
        kb = res["blocks"]["B2"]["estimate"]["K"]["b"]["k"]["point"]
        self.assertAlmostEqual(kb, -0.5, 9)
        self.assertEqual(res["params"]["thr"], 60.0)
        CASES_SEEN.add("参数从清单读")

    def test_full_B_from_manifest(self):
        """用清单原值 B = 10000。"""
        res = self.go({}, B=10000)
        self.assertEqual(res["params"]["B"], 10000)
        self.assertEqual(json.load(open(os.path.join(self.tmp, "out_cue_closed", R.OUT_NAME + ".json")))["params"]["B"], 10000)
        self.assertEqual([rows(res, b)["R"] for b in R.BLOCKS], ["R3", "R3"])
        self.assertEqual([rows(res, b)["K"] for b in R.BLOCKS], ["K1", "K1"])
        self.assertFalse(res["both_blocks"]["labels_differ"])
        CASES_SEEN.add("B = 10000 原值")


class TestNewGuards(Base):
    """审查后新加的护栏：真实数据下比对文件只认默认路径；STAGE1_OK；采后文件的先后；读数锁。"""

    def test_real_root_only_default_cmp_paths(self):
        root, mp = build(self.tmp, {})
        os.remove(os.path.join(root, "SYNTHETIC_DATA"))
        fake = os.path.join(self.tmp, "fake_ok.json")
        write_json(fake, {"ok": True})
        for opt in ("--anchor-post", "--init-cmp"):
            with self.assertRaises(R.Refuse) as cm:
                R.main(["--root", root, opt, fake])
            self.assertIn(f"{opt} 只许取默认路径", str(cm.exception))
        with self.assertRaises(R.Refuse) as cm:            # 显式给默认路径可以，往下走到 root 钉死才拒
            R.main(["--root", root, "--anchor-post", R.default_anchor_post(os.path.abspath(root))])
        self.assertIn("--root 只能是", str(cm.exception))
        CASES_SEEN.add("真实数据只认默认比对路径")

    def test_real_root_pinned(self):
        root, mp = build(self.tmp, {})
        os.remove(os.path.join(root, "SYNTHETIC_DATA"))
        with self.assertRaises(R.Refuse) as cm:
            R.main(["--root", root])
        self.assertIn("--root 只能是", str(cm.exception))
        self.assertEqual(R.REAL_STATE, os.path.join(os.path.dirname(os.path.abspath(R.__file__)), "formal_state"))
        CASES_SEEN.add("root 钉死")

    def test_stage1_content_checked(self):
        for bad in ({"B2": {"C3": {"equal": 9}}}, {"B1": {"C4": {"differ": ["B1__ep03__t0"]}}}, {"B1": {"C0": {"n_ref": 9, "equal": 9}}}):
            root, mp = build(self.tmp, {}, stage1=bad)
            with self.assertRaises(R.Refuse) as cm:
                run(root, mp)
            self.assertIn("内容不过", str(cm.exception))
            shutil.rmtree(self.tmp)
            os.makedirs(self.tmp)
        CASES_SEEN.add("STAGE1_OK 核内容")

    def test_all_registered_code_hashes_checked(self):
        root, mp = build(self.tmp, {}, man_over={"code_sha256": {"cue_cmp.py": "0" * 64}})
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("cue_cmp.py", str(cm.exception))
        CASES_SEEN.add("全部代码哈希")

    def test_stage1_ok_required(self):
        root, mp = build(self.tmp, {}, stage1=False)
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("STAGE1_OK", str(cm.exception))
        self.assertFalse(os.path.exists(os.path.join(root, "formal_state", R.OUT_NAME + ".lock")))
        CASES_SEEN.add("STAGE1_OK 必须存在")

    def test_post_files_must_be_later_than_records(self):
        root, mp = build(self.tmp, {})
        ap = os.path.join(self.tmp, "out_cue_anchor_post", "ANCHOR_POST.json")
        ic = os.path.join(root, "INIT_CMP.json")
        rec = os.path.join(root, "B2", "S+b", make_keys("B2")[40]["key"] + ".npz")
        for p in (ap, ic):
            later(p, dt=-3600)
            with self.assertRaises(R.Refuse) as cm:
                run(root, mp)
            self.assertIn("修改时间不晚于", str(cm.exception))
            later(p)
        npz = os.path.join(self.tmp, "out_cue_anchor_post", "B1", "C0", "B1__ep00__t0.npz")
        os.makedirs(os.path.dirname(npz))
        np.savez(npz, x=np.zeros(1))
        later(npz, dt=-3600)
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn("B1__ep00__t0.npz", str(cm.exception))
        later(npz)
        later(rec, dt=600)                                    # 有一条主键记录晚于采后比对（采后又补采）
        with self.assertRaises(R.Refuse) as cm:
            run(root, mp)
        self.assertIn(rec, str(cm.exception))
        later(rec, dt=-600)
        res = run(root, mp)                                    # 先后都对上之后正常读
        self.assertEqual(rows(res, "B1")["R"], "R3")
        CASES_SEEN.add("采后文件晚于全量")

    def test_lock_refuses_reread(self):
        root, mp = build(self.tmp, {})
        res = run(root, mp)
        lp = os.path.join(root, "formal_state", R.OUT_NAME + ".lock")
        self.assertEqual(res["lock"], lp)
        self.assertEqual(open(lp).read().strip(), os.path.realpath(root))
        for f in (".txt", ".json"):
            os.remove(os.path.join(root, R.OUT_NAME + f))
        with self.assertRaises(R.Refuse) as cm:              # 删掉输出也不能重读
            run(root, mp)
        self.assertIn("读数锁", str(cm.exception))
        CASES_SEEN.add("读数锁")


class TestReviewGaps(Base):
    """审查方改坏实验里测试没抓住的 10 处（M01、M07、M09、M10、M11、M12、M14、M16、M19、M22、M23），每处一个用例。"""

    def test_M12_selective_needs_P1__M16_both_tiers(self):
        b1 = bspec(succ={"S-b": 0, "S+b": 0})                                      # R2、K1、P1
        b2 = bspec(succ={"S-b": 0, "S+b": 0}, kh={"b": dict(q2=0.0), "a": dict(tc=0.0)})   # R2、K1、P2；k_a 含 0
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual([rows(res, "B1")[g] for g in "RKP"], ["R2", "K1", "P1"])
        self.assertIn(R.SELECTIVE_BOTH, res["blocks"]["B1"]["decision"]["synthesis"]["选择性证据"])
        self.assertEqual([rows(res, "B2")[g] for g in "RKP"], ["R2", "K1", "P2"])
        sel = res["blocks"]["B2"]["decision"]["synthesis"]["选择性证据"]
        self.assertIn(R.SELECTIVE_SR, sel)
        self.assertNotIn(R.SELECTIVE_BOTH, sel)
        self.assertIn("两档都有", sent(res, "B1", "K"))
        self.assertNotIn("两档都有", sent(res, "B2", "K"))                          # k_a 含 0
        CASES_SEEN.update({"M12", "M16"})

    def test_M14_F1_dsrb_only_when_b_down__M19_h_not_in_R4(self):
        b1 = bspec(succ={"C4": CL(16), "S-b": CL(16), "S+b": CL(16), "S-a": 0, "S+a": 0})     # R1、D_SR(b)=0、F1
        b2 = bspec(succ={"C4": CL(16), "S-b": CL(16), "S+b": CL(16), "S-a": CL(16), "S+a": CL(16)})  # D_SR(h) 升高
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual((rows(res, "B1")["R"], rows(res, "B1")["F"]), ("R1", "F1"))
        self.assertNotIn(R.MOD["F1_DSRB"], sent(res, "B1", "F"))
        sr2 = res["blocks"]["B2"]["estimate"]["sr"]
        self.assertTrue(R.is_higher(sr2["D_SR_h"], 10))
        self.assertEqual(rows(res, "B2")["R"], "R3")
        CASES_SEEN.update({"M14", "M19"})

    def test_M07_F2_floor_needs_b_floor__M09_disp_line(self):
        b1 = bspec(succ={"S-a": 0, "S+a": 0, "SC-b": 0, "SC+b": 0})                # R1、F2；Rec(b) 提高，所以不是都见底
        b2 = bspec(pre_disp={"S+b": (5, 0.01), "S+a": (5, 0.01)})                  # 第 8 行物体恰漂 1 cm：不合格
        res = self.go({"B1": b1, "B2": b2})
        self.assertEqual((rows(res, "B1")["R"], rows(res, "B1")["F"]), ("R1", "F2"))
        self.assertEqual(res["blocks"]["B1"]["decision"]["levels"]["Rec_b"], "提高")
        self.assertNotIn("没有分辨力", sent(res, "B1", "F"))
        P = res["blocks"]["B2"]["estimate"]["P"]
        self.assertEqual((P["b1"]["n_qual"], P["b2"]["n_qual"], P["a2"]["n_qual"]), (NK, 0, 0))
        self.assertEqual(rows(res, "B2")["P"], "P2")
        CASES_SEEN.update({"M07", "M09"})

    def test_M10_M11_77_is_enough__M22_preclose_only(self):
        late = {"q_frac_inview": lambda i, q: 0.5 if q >= 4 else 1.0}            # 只在闭合之后的查询出画
        b1 = bspec(tc={"OB+b": first_n(19, -1, TC), "SC+b": first_n(19, -1, TC), "S+b": first_n(19, -1, TC)},
                   succ={"S-b": 0, "S+b": 0, "SC-b": 0, "SC+b": 0})
        b2 = bspec(q={"S+b": late, "S-b": late})
        res = self.go({"B1": b1, "B2": b2})
        e1 = res["blocks"]["B1"]["estimate"]
        self.assertEqual((e1["A"]["n_pairs"], e1["F"]["delta_off_n"], e1["K"]["b"]["n_pairs"]), (77, 77, 77))
        self.assertEqual(rows(res, "B1")["A"], "A1")
        self.assertIsNotNone(e1["F"]["delta_off"])
        self.assertEqual(rows(res, "B1")["F"], "F3")
        self.assertNotIn("算不出", sent(res, "B1", "F"))
        self.assertEqual(res["blocks"]["B1"]["decision"]["K_tier"], "b")
        g2 = res["blocks"]["B2"]["gate"]
        self.assertEqual(g2["cell_flags"]["S+b"], [])
        self.assertEqual(rows(res, "B2")["K"], "K1")
        self.assertEqual(res["blocks"]["B2"]["decision"]["K_tier"], "b")
        CASES_SEEN.update({"M10", "M11", "M22"})

    def test_M23_synth_flag_both_sides__sec18_sentences(self):
        b1 = bspec(q={"OB+b": {"q_ob_lost_offscreen": 0.5}})
        b2 = bspec(tc={"S+b": first_n(20, -1, TC)})                                # K 替补成 a 档
        res = self.go({"B1": b1, "B2": b2})
        fl = res["blocks"]["B1"]["gate"]["cell_flags"]
        self.assertIn(G.SYNTH_WEAK, fl["OB+b"])
        self.assertIn(G.SYNTH_WEAK, fl["OB-b"])
        self.assertEqual(rows(res, "B1")["A"], "A0")
        self.assertIn("无信息", sent(res, "B1", "A"))
        self.assertIn("篮子是叠加的照片", sent(res, "B1", "A"))
        self.assertNotIn("无信息", sent(res, "B2", "A"))
        self.assertEqual(res["blocks"]["B2"]["decision"]["K_tier"], "a")
        self.assertIn(R.SEC6_SUB_A, sent(res, "B2", "K"))
        self.assertNotIn(R.SEC6_SUB_A, sent(res, "B1", "K"))
        txt = open(os.path.join(self.tmp, "out_cue_closed", R.OUT_NAME + ".txt")).read()
        self.assertIn("-0.52 到 -0.61", txt)
        self.assertIn("-0.43 到 -0.51", txt)
        CASES_SEEN.update({"M23", "§18 两句", "K 替补写明 b 档预测"})


def explicit(vals, clus, B, seed, stat):
    """显式重采样：同一随机数序列，每次把抽到的簇里的键全部放进去，再取统计量。"""
    vals = np.asarray(vals, float)
    labs, inv = np.unique(np.asarray(clus), return_inverse=True)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(labs), size=(B, len(labs)))
    members = [np.nonzero(inv == j)[0] for j in range(len(labs))]
    stats = [stat(vals[np.concatenate([members[j] for j in row])]) for row in idx]
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(lo), float(hi)


class TestUnits(unittest.TestCase):
    def test_boot_explicit_unequal_clusters(self):
        rng = np.random.RandomState(3)
        clus = sum(([f"ep{j:02d}"] * (1 + j % 4) for j in range(24)), [])
        v = rng.randn(len(clus))
        bt = R.Boot(500, 0, [2.5, 97.5])
        m = bt.med(v, clus)
        self.assertEqual((m["lo"], m["hi"]), explicit(v, clus, 500, 0, np.median))
        mm = bt.mean(v, clus)
        lo, hi = explicit(v, clus, 500, 0, np.mean)
        self.assertAlmostEqual(mm["lo"], lo, 12)
        self.assertAlmostEqual(mm["hi"], hi, 12)

    def test_negative_boundary(self):
        self.assertEqual(R.sign_of({"point": -1.0, "lo": -2.0, "hi": 0.0}), R.ZERO)
        self.assertEqual(R.sign_of({"point": -1.0, "lo": -2.0, "hi": -1e-6}), R.NEG)
        self.assertEqual(R.sign_of({"point": 1.0, "lo": 0.0, "hi": 2.0}), R.ZERO)
        self.assertEqual(R.lv_down({"point": -10.0, "lo": -20.0, "hi": 0.0}, 10), "不确定")
        self.assertEqual(R.lv_down({"point": -10.0, "lo": -20.0, "hi": -0.01}, 10), "降低")

    def test_M01_up_and_higher_need_lo_gt_0(self):
        e = {"point": 10.0, "lo": 0.0, "hi": 20.0}
        self.assertEqual(R.lv_up(e, 10), "不确定")
        self.assertFalse(R.is_higher(e, 10))
        self.assertEqual(R.lv_up({"point": 10.0, "lo": 0.01, "hi": 20.0}, 10), "提高")
        self.assertFalse(R.is_lower({"point": -10.0, "lo": -20.0, "hi": 0.0}, 10))

    def test_table_has_27_rows(self):
        self.assertEqual(len(R.ROWS), 27)

    def test_zz_all_rows_seen(self):
        """在 TestRows 之后跑（unittest 按类名、方法名排序：TestRows < TestUnits）。"""
        if len(ROWS_SEEN) < 2:
            self.skipTest("单独跑本类时不检查")
        self.assertEqual(sorted(set(R.ROWS) - ROWS_SEEN), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
