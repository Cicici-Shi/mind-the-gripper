"""第二阶段 readout.py（连同它内部调用的 gate.run）的合成数据测试。由第一阶段 b2/test_readout.py 复制，原有用例全部保留，
只改文件名与合成冻结清单的字段；新增第二阶段门槛（上游哈希、策略配置、步数上限、VK_DRIVER_FILES）与标签 C 措辞的用例。
只用合成数据，不碰任何真实目录。运行：python test_readout.py"""
import copy
import hashlib
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import gate as G  # noqa: E402
import readout as R  # noqa: E402

CODE_SHA = {f: hashlib.sha256(f.encode()).hexdigest() for f in G.CODE_FILES_IN_RECORD}  # 合成的「代码哈希」
UPSTREAM = {"CogACT/vla/cogactvla.py": "1" * 64, "llama2_stub/tokenizer.model": "2" * 64}
POLICY = {"widowx_bridge": {"policy_setup": "widowx_bridge", "unnorm_key": "bridge_orig", "cfg_scale": 1.5,
                            "num_ddim_steps": 10, "action_ensemble_horizon": 7},
          "google_robot": {"policy_setup": "google_robot", "unnorm_key": "fractal20220817_data", "cfg_scale": 1.5,
                           "num_ddim_steps": 10, "action_ensemble_horizon": 2}}
SETUP = {"B1": "widowx_bridge", "F1": "google_robot", "F2": "google_robot"}
MAN = {"code_sha256": dict(CODE_SHA), "keylist_sha256": {}, "upstream_sha256": dict(UPSTREAM), "policy_cfg": POLICY,
       "blocks": {b: {"L": 8, "line_width": 8, "g_links": ["g1", "g2"], "protect": ["obj"], "n_keys": 120,
                      "blacken_weak": False, "lag_weak": False, "policy_setup": SETUP[b], "max_episode_steps": 4,
                      "control_freq": 5.0, "query_every": 1} for b in ("B1", "F1", "F2")}}


def make_keys(block, kind="bridge"):
    if kind == "bridge":
        return [dict(key=f"{block}__ep{e:02d}__t{t}", block=block, cluster=f"ep{e:02d}", reset={"episode_id": e})
                for t in range(5) for e in range(24)]
    if kind == "f1":
        return [dict(key=f"{block}__s{s}__t0", block=block, cluster=f"s{s}", reset={}) for s in range(120)]
    return [dict(key=f"{block}__ep{e:02d}_u{(e + 2 * t) % 4}__t{t}", block=block, cluster=f"ep{e:02d}",
                 reset={"episode_id": e, "urdf_version": f"u{(e + 2 * t) % 4}"}) for t in (0, 1) for e in range(60)]


A100, H100 = "NVIDIA A100-PCIE-40GB", "NVIDIA H100 NVL"
MAN["gpu_models"] = [A100, H100]
MAN["gpu_assignment"] = {b: {k["key"]: A100 for k in make_keys(b, kind)} for b, kind in (("B1", "bridge"), ("F1", "f1"), ("F2", "f2"))}


def qframe(cell, vanish=False):
    q = {"step": 0, "G_px": 100, "R_px": 120, "arm_px": 20, "changed_px": 100, "changed_out_E_px": 0}
    if cell in G.HIDE_CELLS:
        q.update(hide_changed_out_G_px=0, hide_changed_out_G_t30_px=0, G_vis_after_hide_px=0)
    if cell in G.OUTLINE_CELLS:
        q.update(Gl_px=100, O_px=50, O_drawn_px=45, O_protected_px=5, line_px_224=0 if vanish else 30)
    return q


def write_rec(path, block, cell, k, s, ever, lag=8, vanish=False, **over):
    """一条带齐门槛所需字段的合成记录。"""
    rec = dict(complete=np.bool_(True), key=k["key"], cell=cell, block=block, seeds=np.array([1, 2, 3], np.uint64),
               qstats=json.dumps([qframe(cell, vanish)] * 4), gpu_name="NVIDIA A100-PCIE-40GB",
               nvidia_smi="GPU 0: NVIDIA A100-PCIE-40GB (UUID: x)", restore_maxdiff=np.int32(0),
               init_robot_qpos=np.zeros(3), init_robot_pose=np.zeros(7), init_actor_poses=np.zeros((1, 7)),
               init_actor_names='["obj"]', init_art_names="[]", init_art_qpos="[]", init_urdf_version="",
               g_links='["g1", "g2"]', p_names='["obj"]', lag_steps=np.int32(lag), line_width=np.int32(8),
               n_queries=np.int32(4), code_sha=json.dumps(CODE_SHA), vk_icd_filenames=G.VK_ICD_NVIDIA,
               frame0_sha256="ab" * 32, host="synth0",
               actual_episode_id=np.int32(k["reset"].get("episode_id", -1)),
               actual_urdf_version=k["reset"].get("urdf_version", ""),
               upstream_sha=json.dumps(UPSTREAM, sort_keys=True), policy_cfg=json.dumps(POLICY[SETUP[block]], sort_keys=True),
               max_episode_steps=np.int32(4), control_freq=np.float64(5.0), query_every=np.int32(1), vk_driver_files="",
               success=np.bool_(s), success_ever=np.bool_(ever))
    rec.update(over)
    np.savez(path, **{kk: v for kk, v in rec.items() if v is not None})


def write_block(root, block, keys, rates=None, ever_bonus=0.0, seed=1, drop=None, succ=None, ever=None,
                vanish_cells=(), per_rec=None):
    """rates：格 -> 成功率（共用一个潜变量，五格强配对）；或直接给 succ / ever：格 -> 每键布尔数组。
    drop：不写记录的（格，键）；per_rec：（格，键）-> 覆盖字段。"""
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "SYNTHETIC_DATA"), "w").close()
    with open(os.path.join(root, "SYNTHETIC_MANIFEST.json"), "w") as f:
        json.dump(MAN, f)
    u = np.random.RandomState(seed).permutation(len(keys)) / len(keys)
    for c in R.CELLS:
        d = os.path.join(root, block, c)
        os.makedirs(d, exist_ok=True)
        for i, k in enumerate(keys):
            if drop and (c, k["key"]) in drop:
                continue
            if succ is not None:
                s, e = bool(succ[c][i]), bool(ever[c][i])
            else:
                s = bool(u[i] < rates[c])
                e = bool(s or u[i] < rates[c] + ever_bonus)
            write_rec(os.path.join(d, k["key"] + ".npz"), block, c, k, s, e, vanish=c in vanish_cells,
                      **((per_rec or {}).get((c, k["key"]), {})))


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="b2_synth_")
        self.root = os.path.join(self.tmp, "data")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def files(self, block, keys, flags=None, man_keys=None, drop_fields=(), assign=None):
        """写键清单，并把它的 sha256、键数和块级标记写进合成冻结清单（真实数据下这些都来自 FROZEN_bench2）。
        man_keys：清单里登记的那份键清单（默认就是 keys）；drop_fields：从清单 blocks.<块> 里删掉的字段。"""
        kp = os.path.join(self.tmp, "keys.json")
        with open(kp, "w") as fh:
            json.dump(keys, fh)
        reg = keys if man_keys is None else man_keys
        man = copy.deepcopy(MAN)
        man["keylist_sha256"][block] = hashlib.sha256(json.dumps(reg).encode()).hexdigest()
        man["blocks"][block]["n_keys"] = len(reg)
        man["blocks"][block].update(flags or {})
        man["gpu_assignment"][block] = {k["key"]: A100 for k in reg}
        man["gpu_assignment"][block].update(assign or {})
        for f in drop_fields:
            man["blocks"][block].pop(f)
        with open(os.path.join(self.root, "SYNTHETIC_MANIFEST.json"), "w") as fh:
            json.dump(man, fh)
        return kp

    def readout(self, block, keys, flags=None, **kw):
        kp = self.files(block, keys, flags, **kw)
        return R.main(["--root", self.root, "--block", block, "--keys", kp])

    def assert_no_output(self, block):
        for p in R.out_paths(self.root, block):
            self.assertFalse(os.path.exists(p), p)

    def run_case(self, rates, kind="bridge", flags=None, block="B1", **kw):
        keys = make_keys(block, kind)
        write_block(self.root, block, keys, rates, **kw)
        return self.readout(block, keys, flags)

    # ---- 四种标签与 A′
    def test_label_A(self):
        r = self.run_case({"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45})
        d = r["final"]["decision"]
        self.assertEqual(d["label"], "A")
        self.assertEqual(d["P1"], "提高")
        self.assertEqual(d["P2"], "提高")
        self.assertTrue(os.path.exists(os.path.join(self.root, "READOUT_bench2_cogact_B1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.root, "READOUT_bench2_cogact_B1.json")))

    def test_label_B(self):
        d = self.run_case({"C0": .90, "C1": .45, "C3": .40, "C4": .42, "C5": .41})["final"]["decision"]
        self.assertEqual(d["label"], "B")
        self.assertEqual(d["P1"], "未提高")

    def test_label_C(self):
        d = self.run_case({"C0": .90, "C1": .90, "C3": .89, "C4": .90, "C5": .89}, kind="f1", block="F1")["final"]["decision"]
        self.assertEqual(d["label"], "C")
        self.assertTrue(d["P1"].startswith("不判"))

    def test_label_D(self):
        # D3 = -8，区间下界 < -10（非 C），点估计 > -15（非 S1）：用独立潜变量把区间拉宽
        keys = make_keys("F1", "f1")
        rs = np.random.RandomState(3)
        succ = {"C0": rs.rand(120) < .80}
        for c in ("C1", "C3", "C4", "C5"):
            succ[c] = rs.rand(120) < .72
        est = R.estimate({c: succ[c].astype(float) for c in R.CELLS}, [k["cluster"] for k in keys])
        d = R.decide(est, {})
        self.assertGreater(est["D3"]["point"], -15)
        self.assertLess(est["D3"]["lo"], -10)
        self.assertEqual(d["label"], "D")

    def test_label_A_prime(self):
        d = self.run_case({"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45},
                          flags={"blacken_weak": True})["final"]["decision"]
        self.assertEqual(d["label"], "A′")
        self.assertIsNone(d["S2"])
        self.assertIsNone(d["S3"])

    def test_admission_fail(self):
        d = self.run_case({"C0": .55, "C1": .55, "C3": .20, "C4": .30, "C5": .20})["final"]["decision"]
        self.assertFalse(d["admitted"])
        self.assertEqual(d["label"], "基线不足")

    def test_admission_boundary_exact_60(self):
        r = self.run_case({"C0": .60, "C1": .60, "C3": .60, "C4": .60, "C5": .60})
        self.assertAlmostEqual(r["final"]["estimate"]["SR"]["C0"], 60.0)
        self.assertTrue(r["final"]["decision"]["admitted"])

    # ---- 边界值（直接喂估计量）
    def est(self, **kw):
        base = {"SR": {c: 90.0 for c in R.CELLS}}
        for n in R.CONTRASTS:
            base[n] = {"point": 0.0, "lo": -1.0, "hi": 1.0}
        for n, v in kw.items():
            base[n] = dict(zip(("point", "lo", "hi"), v))
        return base

    def test_boundaries(self):
        d = R.decide(self.est(D3=(-15.0, -30, -1e-6), D1=(0, -10.0, 5), **{"Δ7": (15.0, 1e-6, 30)}), {})
        self.assertTrue(d["S1"])
        self.assertTrue(d["S2"])
        self.assertTrue(d["S3"])
        self.assertEqual(d["label"], "A")
        self.assertFalse(R.decide(self.est(D3=(-14.999, -30, -1)), {})["S1"])
        self.assertFalse(R.decide(self.est(D3=(-20, -30, 0.0)), {})["S1"])  # 上界 = 0 不算 < 0
        self.assertFalse(R.decide(self.est(D3=(-20, -30, -1), D1=(0, -10.0001, 5)), {})["S2"])
        self.assertFalse(R.decide(self.est(D3=(-20, -30, -1), **{"Δ7": (15, 0.0, 30)}), {})["S3"])  # 下界 = 0 不算 > 0
        self.assertEqual(R.decide(self.est(D3=(-5, -10.0, 0)), {})["label"], "C")
        self.assertEqual(R.decide(self.est(D3=(-5, -10.0001, 0)), {})["label"], "D")  # 下界 < -10 不判 C（实现审查 2026-11-13）
        d = R.decide(self.est(D3=(-30, -40, -20), D1=(-15.0, -25, -0.1), **{"Δ7": (0, -5, 5)}), {})
        self.assertEqual(d["label"], "B")
        # P 三档
        self.assertEqual(R.p_level({"point": 10.0, "lo": 1e-6, "hi": 20}), "提高")
        self.assertEqual(R.p_level({"point": 10.0, "lo": 0.0, "hi": 20}), "不确定")
        self.assertEqual(R.p_level({"point": 2.0, "lo": -5, "hi": 9.9999}), "未提高")
        self.assertEqual(R.p_level({"point": 2.0, "lo": -5, "hi": 10.0}), "不确定")
        # 标记：轮廓 / 滞后操纵不足
        e = self.est(D3=(-30, -40, -20), **{"Δ1": (20, 10, 30), "Δ2": (20, 10, 30)})
        self.assertEqual(R.decide(e, {"lag_weak": True})["P2"], "不判（滞后操纵不足）")
        self.assertEqual(R.decide(e, {})["P1"], "提高")
        self.assertEqual(R.decide(e, {})["P2"], "提高")
        self.assertEqual(R.decide(e, {"lag_weak": True})["P1"], "提高")  # 滞后操纵不足只影响 P2
        # 协议 v1.1 §4 第 6 项：C4 被标，P1、P2 都不判
        self.assertEqual(R.decide(e, {"outline_weak_C4": True})["P1"], "不判（C4 轮廓操纵不足）")
        self.assertEqual(R.decide(e, {"outline_weak_C4": True})["P2"], "不判（轮廓操纵不足）")
        self.assertEqual(R.decide(e, {"outline_weak_C4": True, "outline_weak_C5": True})["P1"], "不判（C4 轮廓操纵不足）")

    def test_only_C5_outline_weak(self):
        # 只有 C5 被标：P1 照判，只有 P2 不判（P1 = SR(C4) - SR(C3)，不涉及 C5）
        e = self.est(D3=(-30, -40, -20), **{"Δ1": (20, 10, 30), "Δ2": (20, 10, 30)})
        d = R.decide(e, {"outline_weak_C5": True})
        self.assertEqual(d["P1"], "提高")
        self.assertEqual(d["P2"], "不判（轮廓操纵不足）")
        e = self.est(D3=(-30, -40, -20), **{"Δ1": (2, -5, 8), "Δ2": (20, 10, 30)})
        self.assertEqual(R.decide(e, {"outline_weak_C5": True})["P1"], "未提高")

    def test_blacken_weak_without_S1(self):
        d = R.decide(self.est(D3=(-5, -9, 0)), {"blacken_weak": True})
        self.assertEqual(d["label"], "C")
        self.assertIsNone(d["B_cond"])
        self.assertEqual(R.decide(self.est(D3=(-12, -25, 1)), {"blacken_weak": True})["label"], "D")

    # ---- 门槛标记到读数的字符串耦合（S6）：门槛真的标了，读数真的读到
    def test_gate_flag_reaches_readout_C5(self):
        rates = {"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45}
        r = self.run_case(rates, vanish_cells=("C5",))
        d = r["final"]["decision"]
        self.assertTrue(d["flags"]["outline_weak_C5"])
        self.assertFalse(d["flags"]["outline_weak_C4"])
        self.assertEqual(d["P1"], "提高")
        self.assertEqual(d["P2"], "不判（轮廓操纵不足）")

    def test_gate_flag_reaches_readout_C4(self):
        rates = {"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45}
        d = self.run_case(rates, vanish_cells=("C4",))["final"]["decision"]
        self.assertTrue(d["flags"]["outline_weak_C4"])
        self.assertEqual(d["P1"], "不判（C4 轮廓操纵不足）")
        self.assertEqual(d["P2"], "不判（轮廓操纵不足）")
        self.assertEqual(G.OUTLINE_WEAK, "轮廓操纵不足")

    # ---- 估计量数值核对（S6）：显式展开键下标的重采样独立重算；含剔键后簇大小不等
    @staticmethod
    def explicit_boot(succ, clusters, idx):
        labs = sorted(set(clusters))
        members = {l: [i for i, c in enumerate(clusters) if c == l] for l in labs}
        out = {}
        for name, (a, b) in R.CONTRASTS.items():
            d = np.asarray(succ[a], float) - np.asarray(succ[b], float)
            st = np.array([100.0 * d[[i for j in row for i in members[labs[j]]]].mean() for row in idx])
            out[name] = (100.0 * d.mean(), float(np.percentile(st, 2.5)), float(np.percentile(st, 97.5)))
        return out

    def check_estimate_against_explicit(self, keys, excluded_idx):
        rs = np.random.RandomState(7)
        cl_eff = {c: rs.normal(0, 1.2) for c in {k["cluster"] for k in keys}}
        base = {"C0": 1.6, "C1": 1.3, "C3": -0.4, "C4": 0.5, "C5": -0.1}
        succ = {c: [] for c in R.CELLS}
        for k in keys:
            ke = rs.normal(0, 0.6)
            for c in R.CELLS:
                succ[c].append(base[c] + cl_eff[k["cluster"]] + ke + rs.normal(0, 0.8) > 0)
        ever = {c: list(v) for c, v in succ.items()}
        # 剔键：让一个键在 C3 上 GPU 不合格（硬门槛 4），门槛现算时整键五格剔除
        per = {("C3", keys[i]["key"]): {"gpu_name": "Tesla T4"} for i in excluded_idx}
        write_block(self.root, keys[0]["block"], keys, succ=succ, ever=ever, per_rec=per)
        r = self.readout(keys[0]["block"], keys)
        keep = [i for i in range(len(keys)) if i not in set(excluded_idx)]
        cl = [keys[i]["cluster"] for i in keep]
        s = {c: np.array([succ[c][i] for i in keep], float) for c in R.CELLS}
        nC = len(set(cl))
        idx = np.random.default_rng(R.BOOT_SEED).integers(0, nC, size=(R.B_BOOT, nC))
        mine = self.explicit_boot(s, cl, idx)
        est = r["final"]["estimate"]
        self.assertEqual(est["n_keys"], len(keep))
        self.assertEqual(est["n_clusters"], nC)
        for c in R.CELLS:
            self.assertAlmostEqual(est["SR"][c], 100 * s[c].mean(), places=9)
        for n, (p, lo, hi) in mine.items():
            self.assertAlmostEqual(est[n]["point"], p, places=8)
            self.assertAlmostEqual(est[n]["lo"], lo, places=8)
            self.assertAlmostEqual(est[n]["hi"], hi, places=8)
        return r

    def test_estimate_explicit_bridge_equal_clusters(self):
        self.check_estimate_against_explicit(make_keys("B1", "bridge"), [])

    def test_estimate_explicit_unequal_clusters_after_exclusion(self):
        keys = make_keys("B1", "bridge")
        # 剔掉 ep00 整簇（5 键）再加 ep01..ep07 各 1 个：簇数 23，簇大小 4 和 5 混杂；统计量按键加权
        excl = [i for i, k in enumerate(keys) if k["cluster"] == "ep00"] + [i for i, k in enumerate(keys)
                                                                           if k["key"].endswith("__t0") and k["cluster"] in
                                                                           {f"ep{e:02d}" for e in range(1, 8)}]
        r = self.check_estimate_against_explicit(keys, excl)
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 23)
        self.assertEqual(len(r["excluded_keys"]), 12)

    def test_estimate_explicit_f2(self):
        r = self.check_estimate_against_explicit(make_keys("F2", "f2"), [3, 70])
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 60)

    # ---- F2 聚类与布局敏感性
    def test_f2_clusters_by_episode_and_layout_sensitivity(self):
        r = self.run_case({c: .8 for c in R.CELLS}, kind="f2", block="F2")
        self.assertEqual(r["final"]["estimate"]["n_keys"], 120)
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 60)
        lay = r["layout_cluster_sensitivity"]
        self.assertEqual(lay["estimate"]["n_clusters"], 10)
        self.assertIn("不进判定", lay["note"])
        self.assertNotIn("decision", lay)
        self.assertEqual(lay["estimate"]["SR"], r["final"]["estimate"]["SR"])  # 点估计相同，只有区间不同

    def test_fractal_clusters_per_key(self):
        r = self.run_case({c: .9 for c in R.CELLS}, kind="f1", block="F1")
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 120)
        self.assertNotIn("layout_cluster_sensitivity", r)

    def test_ever_parallel_and_clusters(self):
        r = self.run_case({c: .80 for c in R.CELLS}, ever_bonus=0.1)
        self.assertGreater(r["ever"]["estimate"]["SR"]["C0"], r["final"]["estimate"]["SR"]["C0"])
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 24)
        self.assertNotIn("decision", r["ever"])

    def test_bootstrap_deterministic(self):
        rs = np.random.RandomState(0)
        s = {c: (rs.rand(120) < .7).astype(float) for c in R.CELLS}
        cl = [f"ep{i % 24}" for i in range(120)]
        self.assertEqual(R.estimate(s, cl), R.estimate(s, cl))

    # ---- 门槛（读数内部现算）
    def test_manifest_mismatch_excluded(self):
        keys = make_keys("B1")
        per = {("C5", keys[4]["key"]): {"lag_steps": np.int32(16)},
               ("C4", keys[9]["key"]): {"code_sha": json.dumps(dict(CODE_SHA, **{"views.py": "0" * 64}))},
               ("C1", keys[11]["key"]): {"vk_icd_filenames": "/usr/share/vulkan/icd.d/lvp_icd.json"}}
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, per_rec=per)
        r = self.readout("B1", keys)
        self.assertEqual(sorted(r["excluded_keys"]), sorted([keys[4]["key"], keys[9]["key"], keys[11]["key"]]))
        self.assertEqual(r["final"]["estimate"]["n_keys"], 117)

    def test_phase2_manifest_fields_excluded(self):
        # 第二阶段新增的硬门槛 1 / 4 字段：上游文件哈希、策略配置、步数上限、控制频率、查询间隔、VK_DRIVER_FILES
        keys = make_keys("B1")
        per = {("C0", keys[1]["key"]): {"upstream_sha": json.dumps(dict(UPSTREAM, **{"llama2_stub/tokenizer.model": "3" * 64}))},
               ("C1", keys[2]["key"]): {"policy_cfg": json.dumps(dict(POLICY["widowx_bridge"], cfg_scale=3.0))},
               ("C3", keys[3]["key"]): {"max_episode_steps": np.int32(60)},
               ("C4", keys[5]["key"]): {"control_freq": np.float64(3.0)},
               ("C5", keys[6]["key"]): {"query_every": np.int32(4)},
               ("C0", keys[8]["key"]): {"vk_driver_files": "/usr/share/vulkan/icd.d/lvp_icd.json"},
               ("C1", keys[10]["key"]): {"policy_cfg": json.dumps(POLICY["google_robot"])},
               ("C3", keys[12]["key"]): {"upstream_sha": None}}
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, per_rec=per)
        r = self.readout("B1", keys)
        want = sorted(keys[i]["key"] for i in (1, 2, 3, 5, 6, 8, 10, 12))
        self.assertEqual(sorted(r["excluded_keys"]), want)
        self.assertEqual(r["final"]["estimate"]["n_keys"], 112)

    def test_gpu_model_rules(self):
        # 卡型（协调方 2026-11-13）：型号须在清单之内、等于分给该键的型号、同键五格相同；不过的键五格一起剔除
        keys = make_keys("B1")
        assign = {keys[i]["key"]: H100 for i in (0, 1, 2)}
        per = {(c, keys[0]["key"]): {"gpu_name": H100} for c in R.CELLS}            # 分给 H100、五格都在 H100：通过
        per.update({(c, keys[1]["key"]): {"gpu_name": H100} for c in ("C0", "C1", "C3", "C4")})  # C5 在 A100：五格不同型号
        per[("C3", keys[3]["key"])] = {"gpu_name": H100}                              # 分给 A100，C3 跑在 H100
        per.update({(c, keys[4]["key"]): {"gpu_name": "NVIDIA H200"} for c in R.CELLS})  # 不在允许型号之内
        # keys[2] 分给 H100，但五格都跑在 A100：型号与分组不符（表里没有的键只核型号在允许之列，读数先核键清单哈希，所以不会出现）
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, per_rec=per)
        kp = self.files("B1", keys, assign=assign)
        r = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(sorted(r["excluded_keys"]), sorted(keys[i]["key"] for i in (1, 2, 3, 4)))
        self.assertEqual(r["final"]["estimate"]["n_keys"], 116)

    def test_label_C_text_phase2(self):
        # 补充协议 §3 第 7 条：标签 C 的话不带「给了本体状态」的前提
        d = self.run_case({"C0": .90, "C1": .90, "C3": .89, "C4": .90, "C5": .89}, kind="f1", block="F1")["final"]["decision"]
        self.assertEqual(d["label"], "C")
        self.assertEqual(d["text"], "擦掉夹爪也几乎不掉分")
        self.assertNotIn("本体", d["text"])

    def test_retried_error_counted_not_excluded(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        ed = os.path.join(self.root, "B1", "C3", "_errors")
        os.makedirs(ed, exist_ok=True)
        open(os.path.join(ed, keys[2]["key"] + ".txt.resolved"), "w").close()
        g = G.run(self.root, "B1", keys, manifest=MAN)
        self.assertEqual(g["items"]["10_retried_after_error"]["C3"]["n"], 1)
        self.assertEqual(g["n_excluded"], 0)

    def test_frame0_inconsistency_reported_not_excluded(self):
        keys = make_keys("B1")
        per = {("C4", keys[0]["key"]): {"frame0_sha256": "cd" * 32, "host": "synth1"}}
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, per_rec=per)
        g = G.run(self.root, "B1", keys, manifest=MAN)
        f0 = g["items"]["11_frame0_by_host_combo"]
        self.assertEqual(f0["synth0+synth1"], {"keys": 1, "frame0_inconsistent": 1, "frame0_missing": 0})
        self.assertEqual(f0["synth0"]["frame0_inconsistent"], 0)
        self.assertEqual(g["n_excluded"], 0)

    def test_errored_key_excluded(self):
        # 报错后没有重试成功（只有 _errors/<键>.txt、没有记录）：硬门槛 1 剔除，读数照常进行
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, drop={("C3", keys[5]["key"])})
        ed = os.path.join(self.root, "B1", "C3", "_errors")
        os.makedirs(ed, exist_ok=True)
        open(os.path.join(ed, keys[5]["key"] + ".txt"), "w").close()
        r = self.readout("B1", keys)
        self.assertEqual(r["excluded_keys"], [keys[5]["key"]])
        self.assertEqual(r["final"]["estimate"]["n_keys"], 119)


    # ---- 门槛分支逐项（实现审查 2026-11-13：每个分支单独改坏都要有用例失败）
    def gate_excl(self, block, kind, per, manifest="default", **kw):
        keys = make_keys(block, kind)
        write_block(self.root, block, keys, {c: .9 for c in R.CELLS}, per_rec=per)
        man = copy.deepcopy(MAN) if manifest == "default" else manifest
        g = G.run(self.root, block, keys, manifest=man, **kw)
        return keys, sorted(g["excluded_keys"]), g

    def test_gate_branch_init_actor_poses(self):
        keys = make_keys("B1")
        keys, ex, g = self.gate_excl("B1", "bridge", {("C3", keys[3]["key"]): {"init_actor_poses": np.ones((1, 7))}})
        self.assertEqual(ex, [keys[3]["key"]])
        self.assertEqual(g["items"]["2_init_state_mismatch"], 1)

    def test_gate_branch_seeds(self):
        keys = make_keys("B1")
        keys, ex, g = self.gate_excl("B1", "bridge", {("C4", keys[4]["key"]): {"seeds": np.array([1, 2, 4], np.uint64)}})
        self.assertEqual(ex, [keys[4]["key"]])
        self.assertEqual(g["items"]["3_seed_mismatch"], 1)

    def test_gate_branch_mig(self):
        keys = make_keys("B1")
        per = {("C1", keys[5]["key"]): {"nvidia_smi": "GPU 0: NVIDIA A100-PCIE-40GB (UUID: x)\n  MIG 3g.20gb Device 0"}}
        keys, ex, g = self.gate_excl("B1", "bridge", per)
        self.assertEqual(ex, [keys[5]["key"]])
        self.assertTrue(any("MIG" in m for m in g["fail_reasons"][keys[5]["key"]]), g["fail_reasons"])

    def test_gate_branch_g_links(self):
        keys = make_keys("B1")
        keys, ex, _ = self.gate_excl("B1", "bridge", {("C5", keys[6]["key"]): {"g_links": '["g1"]'}})
        self.assertEqual(ex, [keys[6]["key"]])

    def test_gate_branch_p_names(self):
        keys = make_keys("B1")
        keys, ex, _ = self.gate_excl("B1", "bridge", {("C0", keys[7]["key"]): {"p_names": '["obj", "other"]'}})
        self.assertEqual(ex, [keys[7]["key"]])

    def test_gate_branch_line_width(self):
        keys = make_keys("B1")
        keys, ex, _ = self.gate_excl("B1", "bridge", {("C3", keys[8]["key"]): {"line_width": np.int32(6)}})
        self.assertEqual(ex, [keys[8]["key"]])

    def test_gate_branch_f2_urdf(self):
        keys = make_keys("F2", "f2")
        k = keys[11]
        wrong = "u" + str((int(k["reset"]["urdf_version"][1]) + 1) % 4)
        keys, ex, g = self.gate_excl("F2", "f2", {("C4", k["key"]): {"actual_urdf_version": wrong}})
        self.assertEqual(ex, [k["key"]])
        self.assertEqual(g["items"]["1_records"]["spec_mismatch"], {"C4": [k["key"]]})

    def test_gate_branch_incomplete(self):
        keys = make_keys("B1")
        keys, ex, _ = self.gate_excl("B1", "bridge", {("C1", keys[9]["key"]): {"complete": np.bool_(False)}})
        self.assertEqual(ex, [keys[9]["key"]])

    def test_gate_branch_model_within_key_without_assignment(self):
        # 不依赖分组表：没有清单时两种型号都在默认允许之列，只有「同键五格型号相同」这一条能剔它
        keys = make_keys("B1")
        per = {("C4", keys[10]["key"]): {"gpu_name": H100, "nvidia_smi": "GPU 0: NVIDIA H100 NVL (UUID: x)"}}
        keys, ex, g = self.gate_excl("B1", "bridge", per, manifest=None)
        self.assertEqual(ex, [keys[10]["key"]])
        self.assertEqual(g["items"]["4_gpu_model_mismatch_within_key"], 1)
        self.assertEqual(g["items"]["4_gpu_fail_records"], {})

    def test_gate_formal_key_missing_from_assignment(self):
        # 实现审查可选项：正式键不在分组表里要剔，不再静默放过
        keys = make_keys("B1")
        man = copy.deepcopy(MAN)
        del man["gpu_assignment"]["B1"][keys[13]["key"]]
        _, ex, g = self.gate_excl("B1", "bridge", {}, manifest=man)          # 自动：键清单与表有交集 -> 当正式键
        self.assertEqual(ex, [keys[13]["key"]])
        self.assertTrue(g["require_gpu_assignment"])
        _, ex, _ = self.gate_excl("B1", "bridge", {}, manifest=man, require_assignment=True)
        self.assertEqual(ex, [keys[13]["key"]])
        # 冒烟 / 校准键整份都不在表里：自动判为非正式，只核型号在允许之列，不剔
        man2 = copy.deepcopy(MAN)
        man2["gpu_assignment"]["B1"] = {}
        _, ex, g = self.gate_excl("B1", "bridge", {}, manifest=man2)
        self.assertEqual(ex, [])
        self.assertFalse(g["require_gpu_assignment"])

    def test_readout_formal_key_missing_from_assignment_excluded(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        kp = self.files("B1", keys)
        man = json.load(open(os.path.join(self.root, "SYNTHETIC_MANIFEST.json")))
        del man["gpu_assignment"]["B1"][keys[2]["key"]]
        with open(os.path.join(self.root, "SYNTHETIC_MANIFEST.json"), "w") as fh:
            json.dump(man, fh)
        r = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(r["excluded_keys"], [keys[2]["key"]])

    # ---- 按卡型并列报 SR(C0)（补充协议 §3 第 11 条；描述，不进判定）
    def test_by_gpu_model(self):
        keys = make_keys("B1")
        h = [keys[i]["key"] for i in range(0, 120, 4)]           # 30 键分给 H100
        n = len(keys)
        succ = {c: np.array([i % 4 != 0 or i % 8 == 0 for i in range(n)]) for c in R.CELLS}   # H100 键一半成功、A100 键全成功
        ever = {c: np.array([True] * n) for c in R.CELLS}
        per = {(c, k): {"gpu_name": H100, "nvidia_smi": "GPU 0: NVIDIA H100 NVL (UUID: x)"} for c in R.CELLS for k in h}
        write_block(self.root, "B1", keys, succ=succ, ever=ever, per_rec=per)
        kp = self.files("B1", keys, assign={k: H100 for k in h})
        r = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(r["excluded_keys"], [])
        bg = r["by_gpu_model"]
        self.assertIn("不进判定", bg["note"])
        self.assertEqual(bg["models"][H100], {"n_keys": 30, "SR_C0_final": 50.0, "SR_C0_ever": 100.0})
        self.assertEqual(bg["models"][A100], {"n_keys": 90, "SR_C0_final": 100.0, "SR_C0_ever": 100.0})
        self.assertAlmostEqual(r["final"]["estimate"]["SR"]["C0"], 100.0 * 105 / 120)   # 合并估计不受分组影响
        self.assertIn("按卡型的原图格成功率（描述，不进判定）", r["text"])
        txt = open(R.out_paths(self.root, "B1")[0]).read()
        self.assertIn("NVIDIA H100 NVL 键 30，SR(C0) 最后一步 50.0、任一步 100.0", txt)
        js = json.load(open(R.out_paths(self.root, "B1")[1]))
        self.assertEqual(js["by_gpu_model"]["models"][H100]["n_keys"], 30)

    def test_by_gpu_model_does_not_change_decision(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {"C0": .9, "C1": .9, "C3": .5, "C4": .7, "C5": .6})
        r1 = self.readout("B1", keys)
        d1 = r1["final"]["decision"]
        shutil.rmtree(self.root)
        write_block(self.root, "B1", keys, {"C0": .9, "C1": .9, "C3": .5, "C4": .7, "C5": .6},
                    per_rec={(c, keys[i]["key"]): {"gpu_name": H100} for c in R.CELLS for i in range(0, 120, 7)})
        kp = self.files("B1", keys, assign={keys[i]["key"]: H100 for i in range(0, 120, 7)})
        r2 = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(r2["final"], r1["final"])
        self.assertEqual(d1, r2["final"]["decision"])
        self.assertEqual(set(r2["by_gpu_model"]["models"]), {A100, H100})

    # ---- 四块都采完才读（实现审查可选项）
    def two_block_files(self, drop_f1=None):
        kb, kf = make_keys("B1"), make_keys("F1", "f1")
        write_block(self.root, "B1", kb, {c: .9 for c in R.CELLS})
        write_block(self.root, "F1", kf, {c: .9 for c in R.CELLS}, drop=drop_f1)
        man = copy.deepcopy(MAN)
        for b, ks in (("B1", kb), ("F1", kf)):
            raw = json.dumps(ks).encode()
            with open(os.path.join(self.tmp, f"formal_{b}.json"), "wb") as fh:
                fh.write(raw)
            man["keylist_sha256"][b] = hashlib.sha256(raw).hexdigest()
            man["blocks"][b]["n_keys"] = len(ks)
        with open(os.path.join(self.root, "SYNTHETIC_MANIFEST.json"), "w") as fh:
            json.dump(man, fh)
        return kb, kf, os.path.join(self.tmp, "formal_B1.json")

    def test_refuse_other_block_incomplete(self):
        kb, kf, kp = self.two_block_files(drop_f1={("C3", make_keys("F1", "f1")[7]["key"])})
        with self.assertRaises(R.Refuse):
            R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assert_no_output("B1")

    def test_refuse_other_block_keylist_changed(self):
        kb, kf, kp = self.two_block_files()
        with open(os.path.join(self.tmp, "formal_F1.json"), "w") as fh:
            json.dump(kf[:-1], fh)
        with self.assertRaises(R.Refuse):
            R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assert_no_output("B1")

    def test_all_blocks_collected_reads(self):
        kb, kf, kp = self.two_block_files()
        r = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(r["final"]["estimate"]["n_keys"], 120)
        # 别的块有报错文件（重试后仍失败）也算采完
        shutil.rmtree(self.root)
        kb, kf, kp = self.two_block_files(drop_f1={("C3", make_keys("F1", "f1")[7]["key"])})
        ed = os.path.join(self.root, "F1", "C3", "_errors")
        os.makedirs(ed, exist_ok=True)
        open(os.path.join(ed, kf[7]["key"] + ".txt"), "w").close()
        r = R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assertEqual(r["final"]["estimate"]["n_keys"], 120)

    # ---- 拒读
    def test_refuse_missing_record_without_error(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS}, drop={("C3", keys[5]["key"])})
        with self.assertRaises(R.Refuse):
            self.readout("B1", keys)

    def test_refuse_blind_record(self):
        # 盲采集的记录（没有 success 字段）不许被读成失败，必须拒读
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS},
                    per_rec={("C4", keys[7]["key"]): {"success": None, "success_ever": None}})
        with self.assertRaises(R.Refuse):
            self.readout("B1", keys)

    def test_refuse_second_readout(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        self.readout("B1", keys)
        with self.assertRaises(R.Refuse):
            self.readout("B1", keys)

    def test_refuse_real_data_without_frozen(self):
        if os.path.exists(R.FROZEN):
            self.skipTest("FROZEN 存在")
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        os.remove(os.path.join(self.root, "SYNTHETIC_DATA"))
        with self.assertRaises(R.Refuse):
            self.readout("B1", keys)

    # ---- 冻结清单：键清单哈希、键数、块级标记（复核 M1）
    def test_refuse_keys_subset(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        # 只给 60 键：清单里登记的是全部 120 键的哈希，拒读
        with self.assertRaises(R.Refuse):
            self.readout("B1", keys[:60], man_keys=keys)
        self.assert_no_output("B1")
        # 即使哈希恰好登记成这 60 键的那份，键数与 n_keys = 120 不符也拒读
        kp = self.files("B1", keys[:60], man_keys=keys[:60])
        mp = os.path.join(self.root, "SYNTHETIC_MANIFEST.json")
        man = json.load(open(mp))
        man["blocks"]["B1"]["n_keys"] = 120
        with open(mp, "w") as fh:
            json.dump(man, fh)
        with self.assertRaises(R.Refuse):
            R.main(["--root", self.root, "--block", "B1", "--keys", kp])
        self.assert_no_output("B1")

    def test_refuse_keys_modified(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        bad = copy.deepcopy(keys)
        bad[17]["cluster"] = "ep99"  # 改一个键的内容，键名不变
        with self.assertRaises(R.Refuse):
            self.readout("B1", bad, man_keys=keys)
        self.assert_no_output("B1")

    def test_refuse_manifest_missing_flag_fields(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        for f in ("blacken_weak", "lag_weak"):
            with self.assertRaises(R.Refuse):
                self.readout("B1", keys, drop_fields=(f,))
            self.assert_no_output("B1")

    def test_manifest_blacken_weak_true_gives_A_prime(self):
        d = self.run_case({"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45},
                          flags={"blacken_weak": True})["final"]["decision"]
        self.assertTrue(d["flags"]["blacken_weak"])
        self.assertEqual(d["label"], "A′")

    def test_manifest_lag_weak_true_blocks_P2(self):
        d = self.run_case({"C0": .90, "C1": .88, "C3": .40, "C4": .70, "C5": .45},
                          flags={"lag_weak": True})["final"]["decision"]
        self.assertEqual(d["label"], "A")
        self.assertEqual(d["P1"], "提高")
        self.assertEqual(d["P2"], "不判（滞后操纵不足）")

    def test_flags_option_removed(self):
        keys = make_keys("B1")
        write_block(self.root, "B1", keys, {c: .9 for c in R.CELLS})
        kp = self.files("B1", keys)
        with self.assertRaises(SystemExit):  # --flags 已去掉，传了就是未知参数
            R.main(["--root", self.root, "--block", "B1", "--keys", kp, "--flags", kp])
        self.assert_no_output("B1")


class TestGpuGroups(unittest.TestCase):
    """卡型分组（gpu_groups.py）在真实的四份冻结键清单上：只依赖序号、覆盖全部键、A100 键不集中在少数聚类上。
    只读 keysets/formal_*.json 的键名、聚类、任务，不碰任何记录。"""

    def test_pattern_spread_on_formal_keylists(self):
        import collections
        import gpu_groups as GG
        d = os.path.dirname(os.path.abspath(__file__))
        a100 = GG.GPU_MODELS["A"]
        for b in ("B1", "B2", "F1", "F2"):
            with open(os.path.join(d, "keysets", f"formal_{b}.json")) as f:
                ks = json.load(f)
            asg = GG.gpu_assignment(ks)
            self.assertEqual(set(asg), {k["key"] for k in ks}, b)
            self.assertEqual(set(asg.values()), set(GG.GPU_MODELS.values()), b)
            # 只依赖序号：打乱字段不改结果
            self.assertEqual(asg, {k["key"]: GG.gpu_model_of(i) for i, k in enumerate(ks)}, b)
            A = [k for k in ks if asg[k["key"]] == a100]
            per_cl = collections.Counter(k["cluster"] for k in A)
            n_cl = len({k["cluster"] for k in ks})
            # 不集中（周期 6 时 B 块只覆盖 4 个聚类）：覆盖的聚类数不少于 min(A100 键数, 聚类数) 的七成，单个聚类至多 2 个
            self.assertGreaterEqual(len(per_cl), 0.7 * min(len(A), n_cl), (b, len(per_cl), n_cl))
            self.assertLessEqual(max(per_cl.values()), 2, b)
            per_task = collections.Counter(k["task"] for k in A)
            self.assertEqual(len(per_task), len({k["task"] for k in ks}), b)


if __name__ == "__main__":
    unittest.main(verbosity=2)
