"""b2x readout.py（补充协议 v1，连同它内部调用的 gate.run）的合成数据测试。只用合成数据，不碰任何真实目录。
运行：python test_readout.py"""
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
import keys as K  # noqa: E402
import readout as R  # noqa: E402

CODE_SHA = {f: hashlib.sha256(f.encode()).hexdigest() for f in G.CODE_FILES_IN_RECORD}
FILLS = {"k": [0, 0, 0], "w": [255, 255, 255], "L": [190, 170, 120]}
GEOM = {"4": {"all": {"iou_median": 0.30}}, "12": {"all": {"iou_median": 0.07}}}


def mblock(b):
    d = {"L": 8 if b == "B2" else 4, "line_width": 8, "g_links": ["g1", "g2"], "protect": ["obj"], "n_keys": 120,
         "fill_rgb": FILLS, "fill_strength": {"k": 55.0 if b == "B2" else 148.6, "w": 160.0 if b == "B2" else 90.0, "L": 120.0 if b == "B2" else 30.0},
         "ring_contrast": {"k": 90.0, "w": 80.0, "L": 70.0}, "low_contrast": {"k": False, "w": False, "L": False},
         "cells": K.EXT_CELLS[b], "cell_lag": {"C5": 8 if b == "B2" else 4}}
    if b == "F1":
        d.update(cell_lag={"C5": 4, "C5L": 12}, geom_ref=GEOM, control_freq=3.0,
                 frozen_segment={"4": {"frozen_steps": 6}, "12": {"frozen_steps": 14}})
    return d


MAN = {"code_sha256": dict(CODE_SHA), "keylist_sha256": {}, "blocks": {b: mblock(b) for b in ("B2", "F1")}}
TASKS = ["google_robot_pick_horizontal_coke_can", "google_robot_pick_vertical_coke_can", "google_robot_pick_standing_coke_can"]


def make_keys(block):
    if block == "B2":
        return [dict(key=f"B2__ep{e:02d}__t{t}", block="B2", task="widowx_spoon_on_towel", cluster=f"ep{e:02d}",
                     reset={"episode_id": e}) for t in range(5) for e in range(24)]
    return [dict(key=f"F1__s{s}__t0", block="F1", task=TASKS[s // 40], cluster=f"F1_s{s}", reset={}) for s in range(120)]


def qframe(cell, vanish=False):
    q = {"step": 0, "G_px": 100, "R_px": 120, "arm_px": 20, "changed_px": 100, "changed_out_E_px": 0}
    if cell in G.HIDE_CELLS:
        q.update(hide_changed_out_G_px=0, hide_changed_out_G_t30_px=0, G_vis_after_hide_px=0)
    if cell in G.OUTLINE_CELLS:
        q.update(Gl_px=100, O_px=50, O_drawn_px=45, O_protected_px=5, line_px_224=0 if vanish else 30)
    return q


def write_rec(path, block, cell, k, s, vanish=False, **over):
    mb = MAN["blocks"][block]
    fill = FILLS[{"C1w": "w", "C1L": "L"}[cell]] if cell in ("C1w", "C1L") else [-1, -1, -1]
    steps = np.zeros(20, bool)
    if s:
        steps[7:] = True
    rec = dict(complete=np.bool_(True), key=k["key"], cell=cell, block=block, seeds=np.array([1, 2, 3], np.uint64),
               qstats=json.dumps([qframe(cell, vanish)] * 4), gpu_name="NVIDIA A100-PCIE-40GB",
               nvidia_smi="GPU 0: NVIDIA A100-PCIE-40GB (UUID: x)", restore_maxdiff=np.int32(0),
               init_robot_qpos=np.zeros(3), init_robot_pose=np.zeros(7), init_actor_poses=np.zeros((1, 7)),
               init_actor_names='["obj"]', init_art_names="[]", init_art_qpos="[]", init_urdf_version="",
               g_links='["g1", "g2"]', p_names='["obj"]', lag_steps=np.int32(mb["L"]), line_width=np.int32(8),
               n_queries=np.int32(4), code_sha=json.dumps(CODE_SHA), vk_icd_filenames=G.VK_ICD_NVIDIA,
               frame0_sha256="ab" * 32, host="synth0", fill_rgb=np.array(fill, np.int32),
               lag_long=np.int32(mb["cell_lag"].get("C5L", -1) if cell == "C5L" else -1),
               actual_episode_id=np.int32(k["reset"].get("episode_id", -1)), actual_urdf_version="",
               raw_actions=np.zeros((4, 4, 7), np.float32), env_actions=np.zeros((16, 7)),
               success=np.bool_(s), success_ever=np.bool_(s), success_steps=steps)
    rec.update(over)
    np.savez(path, **{kk: v for kk, v in rec.items() if v is not None})


def write_block(root, block, keys, rates=None, seed=1, drop=None, succ=None, vanish_cells=(), per_rec=None, ref=True,
                ref_over=None):
    """合成记录；ref=True 时另写 <root>/SYNTHETIC_REF（合成的第一阶段旧格参照，默认与新记录逐位相同）。"""
    os.makedirs(root, exist_ok=True)
    open(os.path.join(root, "SYNTHETIC_DATA"), "w").close()
    u = np.random.RandomState(seed).permutation(len(keys)) / len(keys)
    for c in K.EXT_CELLS[block]:
        for base, over_map in ((root, per_rec), (os.path.join(root, "SYNTHETIC_REF"), ref_over)):
            if base != root and (not ref or c not in K.EXT_OLD_CELLS):
                continue
            d = os.path.join(base, block, c)
            os.makedirs(d, exist_ok=True)
            for i, k in enumerate(keys):
                if base == root and drop and (c, k["key"]) in drop:
                    continue
                s = bool(succ[c][i]) if succ is not None else bool(u[i] < rates[c])
                write_rec(os.path.join(d, k["key"] + ".npz"), block, c, k, s, vanish=c in vanish_cells,
                          **((over_map or {}).get((c, k["key"]), {})))


def est(**kw):
    base = {"SR": {"C0": 90.0}}
    for n in R.CONTRASTS["F1"]:
        base[n] = {"point": 0.0, "lo": -1.0, "hi": 1.0}
    for n, v in kw.items():
        base[n] = dict(zip(("point", "lo", "hi"), v))
    return base


def info(block="F1", low_L=False, low_w=False, rerun_ok=True, geom=None, weak=None, strength_w=160.0):
    d = {"low_contrast": {"k": False, "w": low_w, "L": low_L}, "strength_here": {}, "ring_here": {},
         "strength_L_here": 120.0, "strength_w_here": strength_w, "strength_k_F1": 148.6, "rerun_ok": rerun_ok,
         "outline_weak": dict({"C4": False, "C5": False, "C5L": False}, **(weak or {}))}
    if block == "F1":
        d.update(geom_ref=geom or GEOM, lag=4, lag_long=12, lag_sec=4 / 3, lag_long_sec=4.0, frozen_segment={})
    return d


class T(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="b2x_synth_")
        self.root = os.path.join(self.tmp, "data")

    def tearDown(self):
        shutil.rmtree(self.tmp)

    def files(self, block, keys, man_keys=None, mb_over=None, drop_fields=()):
        kp = os.path.join(self.tmp, "keys.json")
        with open(kp, "w") as fh:
            json.dump(keys, fh)
        reg = keys if man_keys is None else man_keys
        man = copy.deepcopy(MAN)
        man["keylist_sha256"][block] = hashlib.sha256(json.dumps(reg).encode()).hexdigest()
        man["blocks"][block]["n_keys"] = len(reg)
        man["blocks"][block].update(mb_over or {})
        for f in drop_fields:
            man["blocks"][block].pop(f)
        with open(os.path.join(self.root, "SYNTHETIC_MANIFEST.json"), "w") as fh:
            json.dump(man, fh)
        return kp

    def readout(self, block, keys, **kw):
        kp = self.files(block, keys, **kw)
        return R.main(["--root", self.root, "--block", block, "--keys", kp])

    def run_case(self, block, rates, **kw):
        keys = make_keys(block)
        wkw = {k: kw.pop(k) for k in ("vanish_cells", "per_rec", "drop", "ref", "ref_over") if k in kw}
        write_block(self.root, block, keys, rates, **wkw)
        return self.readout(block, keys, **kw)

    def assert_no_output(self, block):
        for p in R.out_paths(self.root, block):
            self.assertFalse(os.path.exists(p), p)

    # ---------------------------------------------------------------- 判定线边界（§1）
    def test_drop_level_boundaries(self):
        self.assertEqual(R.drop_level({"point": -15.0, "lo": -30, "hi": -1e-6}), "掉分")
        self.assertEqual(R.drop_level({"point": -14.999, "lo": -30, "hi": -1}), "中间")
        self.assertEqual(R.drop_level({"point": -20, "lo": -30, "hi": 0.0}), "中间")  # 上界 = 0 不算 < 0
        self.assertEqual(R.drop_level({"point": -5, "lo": -10.0, "hi": 0}), "几乎不掉")
        self.assertEqual(R.drop_level({"point": -5, "lo": -10.0001, "hi": 0}), "中间")

    def test_p_level_and_sig_neg_boundaries(self):
        self.assertEqual(R.p_level({"point": 10.0, "lo": 1e-6, "hi": 20}), "提高")
        self.assertEqual(R.p_level({"point": 10.0, "lo": 0.0, "hi": 20}), "不确定")
        self.assertEqual(R.p_level({"point": 2.0, "lo": -5, "hi": 9.9999}), "未提高")
        self.assertEqual(R.p_level({"point": 2.0, "lo": -5, "hi": 10.0}), "不确定")
        self.assertTrue(R.sig_neg({"point": -5, "lo": -9, "hi": -1e-6}))
        self.assertFalse(R.sig_neg({"point": -5, "lo": -9, "hi": 0.0}))

    # ---------------------------------------------------------------- §3 抓可乐罐表：每一行
    def test_F1_table_rows(self):
        cases = [
            ({"D(L)": (-3, -8, 2), "E": (30, 15, 45)}, ["F1-1"]),
            ({"D(L)": (-12, -20, -3), "E": (25, 10, 40)}, ["F1-2"]),
            ({"D(L)": (-20, -30, -8), "E": (15, 3, 28)}, ["F1-3"]),
            ({"D(L)": (-20, -30, -8), "E": (5, -5, 15)}, ["F1-4"]),                 # E 不确定
            ({"D(L)": (-20, -30, -8), "E": (1, -5, 8)}, ["F1-4"]),                  # E 未提高（未显著为负）
            ({"D(L)": (-40, -50, -30), "E": (-12, -20, -3)}, ["F1-4", "F1-5"]),     # 掉分且 E 显著为负：两行同时成立
            ({"D(L)": (-3, -8, 2), "E": (-12, -20, -3)}, ["F1-5"]),                 # 任意 D(L)，E 显著为负
            ({"D(L)": (-3, -8, 2), "E": (5, -5, 15)}, ["F1-6"]),                    # 其余
            ({"D(L)": (-12, -20, -3), "E": (1, -5, 8)}, ["F1-6"]),
        ]
        for e, want in cases:
            self.assertEqual(R.decide("F1", est(**e), info())["fill_rows"], want, e)

    def test_F1_table_boundaries(self):
        # D(L) 恰在「几乎不掉」下界 -10、E 恰在「提高」线上
        self.assertEqual(R.decide("F1", est(**{"D(L)": (-5, -10.0, 0), "E": (10.0, 1e-6, 20)}), info())["fill_rows"], ["F1-1"])
        self.assertEqual(R.decide("F1", est(**{"D(L)": (-5, -10.0001, 0), "E": (10.0, 1e-6, 20)}), info())["fill_rows"], ["F1-2"])
        self.assertEqual(R.decide("F1", est(**{"D(L)": (-15.0, -25, -1e-6), "E": (10.0, 0.0, 20)}), info())["fill_rows"], ["F1-4"])
        self.assertEqual(R.decide("F1", est(**{"D(L)": (-3, -8, 2), "E": (-5, -12, 0.0)}), info())["fill_rows"], ["F1-6"])

    # ---------------------------------------------------------------- §3 勺子表（v1.1：用白色 D(w)）：每一行
    def test_B2_table_rows(self):
        for dw, want in (((-20, -30, -8), "B2-1"), ((-2, -8, 4), "B2-2"), ((-12, -20, -1), "B2-3")):
            d = R.decide("B2", est(**{"D(w)": dw, "D(L)": (-40, -50, -30)}), info("B2"))  # D(L) 只照报，不影响
            self.assertEqual(d["fill_rows"], [want])
            if want == "B2-2":
                self.assertTrue(any("操纵强度" in n for n in d["fill_notes"]))
        self.assertNotIn("lag_rows", R.decide("B2", est(), info("B2")))

    def test_B2_strength_comparison(self):
        e = est(**{"D(w)": (-2, -8, 4)})
        self.assertEqual(R.decide("B2", e, info("B2", strength_w=148.59))["fill_rows"], ["B2-2s"])  # 低于黑色 148.6
        self.assertEqual(R.decide("B2", e, info("B2", strength_w=148.6))["fill_rows"], ["B2-2"])    # 相等不算低于
        self.assertEqual(R.B2_TABLE["B2-2s"][1], "所测强度下不掉")

    def test_low_contrast_not_in_table(self):
        d = R.decide("F1", est(**{"D(L)": (-20, -30, -8), "E": (15, 3, 28)}), info(low_L=True))
        self.assertEqual(d["fill_rows"], [])
        self.assertTrue(any(R.LOW_CONTRAST_TEXT in n for n in d["fill_notes"]))
        # 勺子块：白色被标低反差，涉及白色的读法不判
        d = R.decide("B2", est(**{"D(w)": (-20, -30, -8)}), info("B2", low_w=True))
        self.assertEqual(d["fill_rows"], [])
        self.assertTrue(any(R.LOW_CONTRAST_TEXT in n for n in d["fill_notes"]))
        d = R.decide("B2", est(**{"D(w)": (-2, -8, 4)}), info("B2", low_w=True))
        self.assertEqual(d["fill_rows"], [])
        self.assertFalse(any(R.LOW_CONTRAST_TEXT in n for n in d["fill_notes"]))  # 不掉分时不写那句
        # 勺子块的 FL 低反差不影响勺子表（表用白色）；可乐罐块的白色低反差不影响可乐罐表
        self.assertEqual(R.decide("B2", est(**{"D(w)": (-20, -30, -8)}), info("B2", low_L=True))["fill_rows"], ["B2-1"])
        self.assertEqual(R.decide("F1", est(**{"D(L)": (-3, -8, 2), "E": (30, 15, 45)}), info(low_w=True))["fill_rows"], ["F1-1"])

    def test_rerun_not_identical_numbers_only(self):
        d = R.decide("F1", est(**{"E": (30, 15, 45)}, T=(20, 5, 30)), info(rerun_ok=False))
        self.assertTrue(d["numbers_only"])
        self.assertNotIn("fill_rows", d)
        self.assertNotIn("lag_rows", d)

    def test_admission(self):
        e = est(**{"E": (30, 15, 45)})
        e["SR"]["C0"] = 59.999
        self.assertTrue(R.decide("F1", e, info())["numbers_only"])
        e["SR"]["C0"] = 60.0
        self.assertFalse(R.decide("F1", e, info())["numbers_only"])

    # ---------------------------------------------------------------- §4 五条
    def test_lag_rows(self):
        cases = [
            (dict(T=(20, 5, 30), V=(15, 2, 28)), ["L-1"]),
            (dict(T=(20, 5, 30), V=(5, 0.0, 15)), ["L-2"]),         # V 下界 = 0 算 <= 0
            (dict(T=(20, 5, 30), V=(5, -3, 15)), ["L-2"]),
            (dict(T=(2, -6, 9)), ["L-3"]),
            (dict(T=(5, -5, 15)), ["L-4"]),
            (dict(T=(5, -5, 15), U=(12, 2, 22)), ["L-4", "L-5"]),
            (dict(T=(20, 5, 30), V=(15, 2, 28), U=(10.0, 1e-6, 20)), ["L-1", "L-5"]),
        ]
        for e, want in cases:
            self.assertEqual(R.decide("F1", est(**e), info())["lag_rows"], want, e)

    def test_lag_outline_weak(self):
        e = est(T=(20, 5, 30), U=(15, 5, 25), V=(15, 2, 28))
        d = R.decide("F1", e, info(weak={"C5L": True}))
        self.assertTrue(d["lag"]["T"].startswith("不判") and d["lag"]["U"].startswith("不判") and d["lag"]["V"].startswith("不判"))
        self.assertEqual(d["lag_rows"], [])
        d = R.decide("F1", e, info(weak={"C4": True}))
        self.assertTrue(d["lag"]["T"].startswith("不判"))
        self.assertEqual(d["lag_rows"], ["L-5"])  # U 不涉及 C4
        d = R.decide("F1", e, info(weak={"C5": True}))
        self.assertEqual(d["lag_rows"], ["L-2", "L-5"])  # V 不判时不能写成对比

    def test_lag_geometry_clause(self):
        g = {"4": {"all": {"iou_median": 0.3}}, "12": {"all": {"iou_median": 0.3}}}
        self.assertIn("只写「更陈旧」", R.decide("F1", est(T=(20, 5, 30)), info(geom=g))["lag"]["geom_clause"])
        self.assertNotIn("更陈旧", R.decide("F1", est(T=(20, 5, 30)), info())["lag"]["geom_clause"])

    # ---------------------------------------------------------------- 端到端（合成记录、门槛现算、合成参照）
    def test_end_to_end_F1(self):
        # U = C5L - C3 = +12、T = C4 - C5L = +23、V = C5 - C5L = +21（共用潜变量，区间很窄）
        r = self.run_case("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .9, "C3": .0, "C4": .35, "C5": .33, "C5L": .12})
        d = r["final"]["decision"]
        self.assertTrue(d["info"]["rerun_ok"])
        self.assertEqual(d["fill_rows"], ["F1-1"])
        self.assertEqual(d["lag_rows"], ["L-1", "L-5"])
        self.assertIn("C5L", r["lag_descriptive"])
        self.assertEqual(set(r["lag_descriptive"]["C5L"]["by_orientation"]), {"horizontal", "vertical", "standing"})
        for p in R.out_paths(self.root, "F1"):
            self.assertTrue(os.path.exists(p))
        txt = open(R.out_paths(self.root, "F1")[0]).read()
        self.assertIn("D(w) =", txt)
        self.assertRegex(txt, r"D\(w\) = [^\n]*（只照报）")  # 可乐罐块的 D(w) 只印数
        self.assertNotRegex(txt, r"D\(L\) = [^\n]*（只照报）")

    def test_end_to_end_B2(self):
        r = self.run_case("B2", {"C0": .9, "C1": .9, "C1w": .6, "C1L": .88})
        d = r["final"]["decision"]
        self.assertEqual(d["fill_rows"], ["B2-1"])
        txt = open(R.out_paths(self.root, "B2")[0]).read()
        self.assertRegex(txt, r"D\(L\) = [^\n]*（只照报）")  # 勺子块的 D(L)、E 只印数
        self.assertRegex(txt, r"\n  E = [^\n]*（只照报）")
        self.assertEqual(r["final"]["estimate"]["n_clusters"], 24)

    def test_rerun_mismatch_end_to_end(self):
        keys = make_keys("F1")
        r = self.run_case("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .9, "C3": .05, "C4": .35, "C5": .33, "C5L": .1},
                          ref_over={("C4", keys[3]["key"]): {"env_actions": np.ones((16, 7))}})
        d = r["final"]["decision"]
        self.assertTrue(d["numbers_only"])
        self.assertEqual(d["info"]["rerun"]["C4"]["differ"], [keys[3]["key"]])

    def test_missing_reference_numbers_only(self):
        r = self.run_case("B2", {"C0": .9, "C1": .9, "C1w": .6, "C1L": .5}, ref=False)
        self.assertTrue(r["final"]["decision"]["numbers_only"])

    def test_gate_C5L_vanish_reaches_readout(self):
        r = self.run_case("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .9, "C3": .05, "C4": .35, "C5": .33, "C5L": .1},
                          vanish_cells=("C5L",))
        d = r["final"]["decision"]
        self.assertTrue(d["info"]["outline_weak"]["C5L"])
        self.assertEqual(d["lag_rows"], [])

    def test_manifest_low_contrast_reaches_readout(self):
        mb = mblock("B2")
        mb["low_contrast"]["w"] = True
        r = self.run_case("B2", {"C0": .9, "C1": .9, "C1w": .6, "C1L": .5}, mb_over={"low_contrast": mb["low_contrast"]})
        self.assertEqual(r["final"]["decision"]["fill_rows"], [])

    def test_combine(self):
        for b, rates in (("B2", {"C0": .9, "C1": .9, "C1w": .6, "C1L": .88}),
                         ("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .93, "C3": .05, "C4": .35, "C5": .33, "C5L": .1})):
            write_block(self.root, b, make_keys(b), rates)
            self.readout(b, make_keys(b))
        lines = R.main(["--root", self.root, "--combine"])
        self.assertTrue(any("合-1" in x and R.COMBINED["合-1"][1] in x and R.COMBINED_SUFFIX in x for x in lines), lines)
        with self.assertRaises(R.Refuse):
            R.main(["--root", self.root, "--combine"])

    def test_combine_other_and_refuse_missing(self):
        with self.assertRaises(R.Refuse):
            os.makedirs(self.root)
            R.main(["--root", self.root, "--combine"])
        for b, rates in (("B2", {"C0": .9, "C1": .9, "C1w": .88, "C1L": .5}),
                         ("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .93, "C3": .05, "C4": .35, "C5": .33, "C5L": .1})):
            write_block(self.root, b, make_keys(b), rates)
            self.readout(b, make_keys(b))
        lines = R.main(["--root", self.root, "--combine"])
        self.assertTrue(any("不写合并句" in x for x in lines), lines)  # 可乐罐几乎不掉、勺子 D(w) 也几乎不掉：其余组合

    def test_combine_row2_and_white_low(self):
        for b, rates in (("B2", {"C0": .9, "C1": .9, "C1w": .88, "C1L": .5}),
                         ("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .6, "C3": .05, "C4": .35, "C5": .33, "C5L": .1})):
            write_block(self.root, b, make_keys(b), rates)
            self.readout(b, make_keys(b))
        lines = R.main(["--root", self.root, "--combine"])
        self.assertTrue(any("合-2" in x and R.COMBINED_SUFFIX in x for x in lines), lines)
        shutil.rmtree(self.root)
        mb = mblock("B2")
        mb["low_contrast"]["w"] = True
        for b, rates in (("B2", {"C0": .9, "C1": .9, "C1w": .6, "C1L": .5}),
                         ("F1", {"C0": .95, "C1": .55, "C1w": .6, "C1L": .93, "C3": .05, "C4": .35, "C5": .33, "C5L": .1})):
            write_block(self.root, b, make_keys(b), rates)
            self.readout(b, make_keys(b), mb_over={"low_contrast": mb["low_contrast"]} if b == "B2" else None)
        lines = R.main(["--root", self.root, "--combine"])
        self.assertTrue(any("低反差" in x and "不写合并句" in x for x in lines), lines)

    # ---------------------------------------------------------------- 估计量数值核对（显式展开键下标重算；含剔键后簇大小不等）
    def check_explicit(self, block, excl_idx):
        keys = make_keys(block)
        rs = np.random.RandomState(7)
        ce = {c: rs.normal(0, 1.2) for c in {k["cluster"] for k in keys}}
        base = {"C0": 1.6, "C1": .4, "C1w": .5, "C1L": 1.0, "C3": -0.4, "C4": .5, "C5": .3, "C5L": -0.1}
        succ = {c: [] for c in K.EXT_CELLS[block]}
        for k in keys:
            ke = rs.normal(0, .6)
            for c in succ:
                succ[c].append(base[c] + ce[k["cluster"]] + ke + rs.normal(0, .8) > 0)
        per = {("C1w", keys[i]["key"]): {"gpu_name": "Tesla T4"} for i in excl_idx}
        write_block(self.root, block, keys, succ=succ, per_rec=per)
        r = self.readout(block, keys)
        keep = [i for i in range(len(keys)) if i not in set(excl_idx)]
        cl = [keys[i]["cluster"] for i in keep]
        labs = sorted(set(cl))
        members = {l: [j for j, c in enumerate(cl) if c == l] for l in labs}
        idx = np.random.default_rng(R.BOOT_SEED).integers(0, len(labs), size=(R.B_BOOT, len(labs)))
        e = r["final"]["estimate"]
        self.assertEqual(e["n_keys"], len(keep))
        for name, (a, b) in R.CONTRASTS[block].items():
            d = np.array([succ[a][i] for i in keep], float) - np.array([succ[b][i] for i in keep], float)
            st = np.array([100 * d[[j for c in row for j in members[labs[c]]]].mean() for row in idx])
            self.assertAlmostEqual(e[name]["point"], 100 * d.mean(), places=8)
            self.assertAlmostEqual(e[name]["lo"], float(np.percentile(st, 2.5)), places=8)
            self.assertAlmostEqual(e[name]["hi"], float(np.percentile(st, 97.5)), places=8)
        return r

    def test_estimate_explicit_B2_unequal_clusters(self):
        keys = make_keys("B2")
        excl = [i for i, k in enumerate(keys) if k["cluster"] == "ep00"] + \
               [i for i, k in enumerate(keys) if k["key"].endswith("__t0") and k["cluster"] in {"ep01", "ep02", "ep03"}]
        self.assertEqual(self.check_explicit("B2", excl)["final"]["estimate"]["n_clusters"], 23)

    def test_estimate_explicit_F1(self):
        self.check_explicit("F1", [3, 70])

    # ---------------------------------------------------------------- 门槛新项
    def test_gate_fill_and_lag_manifest_mismatch_excluded(self):
        keys = make_keys("F1")
        per = {("C1L", keys[4]["key"]): {"fill_rgb": np.array([191, 170, 120], np.int32)},
               ("C1w", keys[9]["key"]): {"fill_rgb": np.array([254, 255, 255], np.int32)},
               ("C5L", keys[11]["key"]): {"lag_long": np.int32(8)},
               ("C5", keys[13]["key"]): {"lag_steps": np.int32(8)}}
        write_block(self.root, "F1", keys, {c: .8 for c in K.EXT_CELLS["F1"]}, per_rec=per)
        g = G.run(self.root, "F1", keys, manifest=MAN, ref_root=None)
        self.assertEqual(sorted(g["excluded_keys"]), sorted([keys[i]["key"] for i in (4, 9, 11, 13)]))
        self.assertEqual(g["n_retained"], 116)  # 剔键范围是该块全部八格

    def test_gate_fill_changed_outside_G_and_fill_report(self):
        keys = make_keys("B2")
        bad = json.dumps([dict(qframe("C1L"), changed_out_E_px=3)] * 4)
        stats = json.dumps([dict(qframe("C1w"), fill_strength_sum=800.0, ring_contrast_sum=300.0, ring_px=10)] * 4)
        write_block(self.root, "B2", keys, {c: .8 for c in K.EXT_CELLS["B2"]},
                    per_rec={("C1L", keys[0]["key"]): {"qstats": bad}, ("C1w", keys[1]["key"]): {"qstats": stats}})
        g = G.run(self.root, "B2", keys, manifest=MAN, ref_root=None)
        self.assertEqual(g["items"]["5_pixels_report"]["C1L"]["keys_with_any_violation"], 1)
        self.assertAlmostEqual(g["items"]["14_fill_strength_ring_contrast"]["C1w"]["all"]["fill_strength"], 8.0)
        self.assertAlmostEqual(g["items"]["14_fill_strength_ring_contrast"]["C1w"]["all"]["ring_contrast"], 30.0)
        self.assertEqual(g["n_excluded"], 0)

    def test_gate_geometry_phases(self):
        keys = make_keys("F1")
        q = [dict(qframe("C5L"), geom={"12": {"phase": ph, "iou": v, "centroid_shift_224": 1.0, "visible_ratio_224": 1.0}})
             for ph, v in (("pre", 0.9), ("pre", 0.8), ("full", 0.1), ("full", 0.2))]
        write_block(self.root, "F1", keys, {c: .8 for c in K.EXT_CELLS["F1"]},
                    per_rec={("C5L", k["key"]): {"qstats": json.dumps(q)} for k in keys})
        g = G.run(self.root, "F1", keys, manifest=MAN, ref_root=None)
        gm = g["items"]["12_lag_geometry"]["C5L"]["12"]
        self.assertAlmostEqual(gm["pre"]["iou_median"], 0.85)
        self.assertAlmostEqual(gm["full"]["iou_median"], 0.15)
        self.assertAlmostEqual(gm["all"]["iou_median"], 0.5)

    # ---------------------------------------------------------------- 拒读
    def test_refuse_keys_subset_and_modified(self):
        keys = make_keys("B2")
        write_block(self.root, "B2", keys, {c: .9 for c in K.EXT_CELLS["B2"]})
        with self.assertRaises(R.Refuse):
            self.readout("B2", keys[:60], man_keys=keys)
        bad = copy.deepcopy(keys)
        bad[17]["cluster"] = "ep99"
        with self.assertRaises(R.Refuse):
            self.readout("B2", bad, man_keys=keys)
        self.assert_no_output("B2")

    def test_refuse_manifest_missing_fields(self):
        keys = make_keys("F1")
        write_block(self.root, "F1", keys, {c: .9 for c in K.EXT_CELLS["F1"]})
        for f in ("low_contrast", "fill_strength", "geom_ref", "cell_lag", "frozen_segment"):
            with self.assertRaises(R.Refuse):
                self.readout("F1", keys, drop_fields=(f,))
            self.assert_no_output("F1")

    def test_refuse_missing_record_and_blind_record(self):
        keys = make_keys("F1")
        write_block(self.root, "F1", keys, {c: .9 for c in K.EXT_CELLS["F1"]}, drop={("C5L", keys[5]["key"])})
        with self.assertRaises(R.Refuse):
            self.readout("F1", keys)
        shutil.rmtree(self.root)
        write_block(self.root, "F1", keys, {c: .9 for c in K.EXT_CELLS["F1"]},
                    per_rec={("C1L", keys[7]["key"]): {"success": None, "success_ever": None}})
        with self.assertRaises(R.Refuse):
            self.readout("F1", keys)
        self.assert_no_output("F1")

    def test_errored_key_excluded(self):
        keys = make_keys("B2")
        write_block(self.root, "B2", keys, {c: .9 for c in K.EXT_CELLS["B2"]}, drop={("C1L", keys[5]["key"])})
        os.makedirs(os.path.join(self.root, "B2", "C1L", "_errors"), exist_ok=True)
        open(os.path.join(self.root, "B2", "C1L", "_errors", keys[5]["key"] + ".txt"), "w").close()
        r = self.readout("B2", keys)
        self.assertEqual(r["excluded_keys"], [keys[5]["key"]])

    def test_refuse_second_readout_and_real_without_frozen(self):
        keys = make_keys("B2")
        write_block(self.root, "B2", keys, {c: .9 for c in K.EXT_CELLS["B2"]})
        self.readout("B2", keys)
        with self.assertRaises(R.Refuse):
            self.readout("B2", keys)
        if not os.path.exists(R.FROZEN):
            shutil.rmtree(self.root)
            write_block(self.root, "B2", keys, {c: .9 for c in K.EXT_CELLS["B2"]})
            kp = self.files("B2", keys)
            os.remove(os.path.join(self.root, "SYNTHETIC_DATA"))
            with self.assertRaises(R.Refuse):
                R.main(["--root", self.root, "--block", "B2", "--keys", kp])


if __name__ == "__main__":
    unittest.main(verbosity=2)
