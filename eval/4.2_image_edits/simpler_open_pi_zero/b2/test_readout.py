"""readout.py（连同它内部调用的 gate.run）的合成数据测试。只用合成数据，不碰任何真实目录。运行：python test_readout.py"""
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
MAN = {"code_sha256": dict(CODE_SHA), "keylist_sha256": {},
       "blocks": {b: {"L": 8, "line_width": 8, "g_links": ["g1", "g2"], "protect": ["obj"], "n_keys": 120,
                      "blacken_weak": False, "lag_weak": False} for b in ("B1", "F1", "F2")}}


def make_keys(block, kind="bridge"):
    if kind == "bridge":
        return [dict(key=f"{block}__ep{e:02d}__t{t}", block=block, cluster=f"ep{e:02d}", reset={"episode_id": e})
                for t in range(5) for e in range(24)]
    if kind == "f1":
        return [dict(key=f"{block}__s{s}__t0", block=block, cluster=f"s{s}", reset={}) for s in range(120)]
    return [dict(key=f"{block}__ep{e:02d}_u{(e + 2 * t) % 4}__t{t}", block=block, cluster=f"ep{e:02d}",
                 reset={"episode_id": e, "urdf_version": f"u{(e + 2 * t) % 4}"}) for t in (0, 1) for e in range(60)]


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

    def files(self, block, keys, flags=None, man_keys=None, drop_fields=()):
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
        self.assertTrue(os.path.exists(os.path.join(self.root, "READOUT_bench2_B1.txt")))
        self.assertTrue(os.path.exists(os.path.join(self.root, "READOUT_bench2_B1.json")))

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


if __name__ == "__main__":
    unittest.main(verbosity=2)
