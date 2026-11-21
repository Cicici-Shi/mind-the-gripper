"""Synthetic-data tests for prereg_maskinfo_confirm.py (protocol_maskinfo_confirm.md v1.1; copy of the 59 frozen tests of
test_prereg_maskinfo.py adapted to the *_c30 blocks and ep 30-39 keys, plus the confirm additions: §3 R-A / R-B (i)-(iv)
/ G / R-0 / R-C, J row against the first round, Object F row, key-set check, three-way labels, --check-fields-only,
forbidden-word additions, fixture fields vs the fields the eval script writes).
Run: python3 -m unittest test_prereg_maskinfo_confirm -v   (from this directory)
The protocol file is read from ref/protocol_maskinfo_confirm.md (override with MI_PROTO); its sha is computed and must
equal the script constant PROTOCOL_SHA.
"""
import hashlib
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prereg_maskinfo_confirm.py")
PROTO = os.environ.get("MI_PROTO", os.path.join(HERE, "ref", "protocol_maskinfo_confirm.md"))
MASKREPL = os.path.join(HERE, "ref", "prereg_maskrepl.py")
with open(PROTO, "rb") as _f:
    PSHA = hashlib.sha256(_f.read()).hexdigest()
TMPROOT = os.path.join(HERE, "_testtmp")

sys.path.insert(0, HERE)
import prereg_maskinfo_confirm as PM  # noqa: E402

KEYS = PM.DESIGN_KEYS
CELLS = PM.CELLS
MARK = PM.SYN_MARKER
A100 = "NVIDIA H200 NVL|143771 MiB"  # default formal card (name kept from v2 tests; v3 allows h200 / h100-96 only)
FT_CYCLE = ["no_reach", "no_grasp", "drop", "other"]
DEFAULT_SR = {"C0": 95, "C1": 92, "C2": 75, "C3": 45, "C4": 80, "C5": 50, "C6": 78, "C7": 93}   # -> row A only
DEFAULT_TIMING = (20, 22, 30)   # t_near, t_close, t_lift


def jload(p):
    with open(p) as f:
        return json.load(f)


def jdump(o, p):
    with open(p, "w") as f:
        json.dump(o, f)


def rtext(p):
    with open(p) as f:
        return f.read()


def keyorder(seed=1):
    rng = np.random.default_rng(seed)
    return [KEYS[i] for i in rng.permutation(len(KEYS))]


ORDER = keyorder()


def nested(n):
    """Success on the first n keys of one fixed permutation -> success sets of different cells are nested."""
    s = set(ORDER[:n])
    return {k: (k in s) for k in KEYS}


def from_ranges(*ranges):
    s = set()
    for a, b in ranges:
        s |= set(ORDER[a:b])
    return {k: (k in s) for k in KEYS}


def ftype_of(k):
    return FT_CYCLE[(k[0] * 10 + k[1]) % 4]


PADS = json.dumps(["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"])


def episode(succ, k, timing):
    """Synthetic per-step trajectory (rows = states 0..T). timing = (t_near, t_close, t_lift); pads touch at
    t_close+1 (one side) and t_close+2 (both). success: carried 40 steps onto the plate / basket (object 1) and released
    there; drop: carried 20 steps then released off target; other: lifted but support contact kept (no carry);
    no_grasp: one pad only, no lift; no_reach: never near."""
    t_near, t_close, t_lift = timing
    ft = "success" if succ else ftype_of(k)
    T = 80 if succ else 100
    p0 = np.array([0.1, 0.2, 0.9])
    ref = np.array([0.3, 0.2, 0.88])
    obj = np.tile(p0, (T + 1, 1))
    grip = np.tile(p0 + [0.5, 0.0, 0.2], (T + 1, 1))
    lp = np.zeros(T + 1, bool)
    rp = np.zeros(T + 1, bool)
    sup = np.ones(T + 1, bool)
    if ft != "no_reach":
        grip[t_near:] = p0 + [0.02, 0.0, 0.0]
    if ft == "no_grasp":
        lp[t_close + 1:] = True
    if ft in ("success", "drop", "other"):
        lp[t_close + 1:] = True
        rp[t_close + 2:] = True
        rel = t_lift + (40 if ft == "success" else 20 if ft == "drop" else T)
        for j in range(t_lift, T + 1):
            if j < rel:
                obj[j] = p0 + [0.005 * (j - t_lift), 0.0, 0.05]
                grip[j] = obj[j] + [0.02, 0.0, 0.0]
            else:
                obj[j] = obj[rel - 1] * [1, 1, 0] + [0, 0, p0[2]]
                grip[j] = grip[rel - 1]
        if ft != "other":
            sup[t_lift:rel] = False
            lp[rel:] = False
            rp[rel:] = False
    act = np.zeros((T, 7), np.float32)
    act[:, -1] = -1
    act[t_close:, -1] = 1
    return dict(action=act, mi_obj_pos=np.stack([obj, np.tile(ref, (T + 1, 1))], 1).astype(np.float32),
                mi_grip_pos=grip.astype(np.float32), mi_contact_lpad=lp, mi_contact_rpad=rp,
                mi_contact_grip_target=lp | rp, mi_contact_target_support=sup,
                mi_contact_grip_interactive=lp | rp, mi_pad_geoms=np.array(PADS),
                mi_obj_names=np.array(["akita_black_bowl_1", "plate_1"]))


class Fixture:
    def __init__(self, blocks=None, gate_block=None, gate_cells=None, phys=None, drop=None, gpu=None, gate_sha=None,
                 marker=True, omit_field=None, extra_file=None, override=None):
        """blocks: {block: {"sr": {...}, "succ": {cell: map}, "timing": {cell: (r, c, l)}}}"""
        os.makedirs(TMPROOT, exist_ok=True)
        self.base = tempfile.mkdtemp(dir=TMPROOT)
        self.runs = os.path.join(self.base, "runs")
        self.out = os.path.join(self.base, "results_maskinfo_confirm.txt")
        self.vj = os.path.join(self.base, "verdict_maskinfo_confirm.json")
        self.gate = os.path.join(self.base, "gate_maskinfo_confirm.json")
        blocks = blocks if blocks is not None else {"oft_spatial_c30": {}}
        drop, gpu = drop or {}, gpu or {}
        self.succ, self.timing = {}, {}
        g = {"protocol_sha256": gate_sha or PSHA, "blocks": {}}
        for b, spec in blocks.items():
            sr = dict(DEFAULT_SR, **(spec.get("sr") or {}))
            self.succ[b], self.timing[b] = {}, {}
            for c in CELLS:
                s = (spec.get("succ") or {}).get(c) or nested(sr[c])
                tm = (spec.get("timing") or {}).get(c, DEFAULT_TIMING)
                self.succ[b][c], self.timing[b][c] = s, tm
                d = os.path.join(self.runs, b, c)
                os.makedirs(d, exist_ok=True)
                for k in KEYS:
                    if k in (drop.get(b, {}).get(c) or ()):
                        continue
                    t, e = k
                    fields = dict(success=np.array(bool(s[k])), task=np.array(t), ep=np.array(e), init=np.array(e),
                                  gpu=np.array(gpu.get(b, {}).get(c, A100)), host=np.array("node27"),
                                  mi_cell=np.array(c), mr_cell=np.array(c), mi_block=np.array(b),
                                  **episode(s[k], k, tm))
                    if omit_field and (b, c, k) == omit_field[:3]:
                        fields.pop(omit_field[3])
                    if override and (b, c, k) == override[:3]:
                        fields[override[3]] = np.array(override[4])
                    np.savez(os.path.join(d, f"t{t}_e{e}_i{e}.npz"), **fields)
            gb = {"card_type": "h200", "block_ok": True, "block_fail_reasons": [],
                  "cells": {c: {"status": "PASS", "bad_keys": [], "reasons": []} for c in CELLS},
                  "physics_breakdown": {c: [] for c in CELLS}}
            gb.update((gate_block or {}).get(b, {}))
            for c, v in ((gate_cells or {}).get(b) or {}).items():
                gb["cells"][c].update(v)
            for c, ks in ((phys or {}).get(b) or {}).items():
                gb["physics_breakdown"][c] = [list(k) for k in ks]
            g["blocks"][b] = gb
        if extra_file:
            b, c, name = extra_file
            shutil.copy(os.path.join(self.runs, b, c, "t0_e30_i30.npz"), os.path.join(self.runs, b, c, name))
        jdump(g, self.gate)
        if marker:
            for d in (self.base, self.runs):
                open(os.path.join(d, MARK), "w").close()

    def args(self, blocks, extra=(), synthetic=True, sha=None):
        a = ["--protocol", PROTO, "--expect-protocol-sha", sha or PSHA, "--runs", self.runs, "--gate", self.gate,
             "--blocks", blocks, "--out", self.out, "--verdict-json", self.vj, "--quiet"]
        if synthetic:
            a.append("--synthetic")
        return a + list(extra)

    def run(self, blocks="oft_spatial_c30", extra=(), synthetic=True, sha=None):
        return subprocess.run([sys.executable, "-I", SCRIPT] + self.args(blocks, extra, synthetic, sha),
                              capture_output=True, text=True)

    def lock(self, b):
        return os.path.join(self.base, f"prereg_maskinfo_{b}.lock")

    def cleanup(self):
        shutil.rmtree(self.base, ignore_errors=True)


class Base(unittest.TestCase):
    def make(self, **kw):
        fx = Fixture(**kw)
        self.addCleanup(fx.cleanup)
        return fx

    def ok_run(self, fx, blocks="oft_spatial_c30", extra=()):
        n0 = len(rtext(fx.out)) if os.path.exists(fx.out) else 0
        p = fx.run(blocks, extra)
        txt = rtext(fx.out) if os.path.exists(fx.out) else ""
        new = txt[n0:]
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        V = jload(fx.vj)
        self.assertEqual(PM.wording_violations(new), [])
        self.assertIn("PASS（无禁用词）", new)
        self.assertIn("合成数据测试运行", txt.splitlines()[0])
        return V, txt

    def block(self, fx, sr=None, blocks="oft_spatial_c30", **kw):
        name = blocks.split(",")[0]
        V, txt = self.ok_run(fx, blocks, **kw)
        return V["blocks"][name], V, txt

    def refused(self, fx, p, code, blocks=("oft_spatial_c30",)):
        self.assertEqual(p.returncode, code, p.stderr + p.stdout)
        self.assertFalse(os.path.exists(fx.out))
        self.assertFalse(os.path.exists(fx.vj))
        for b in blocks:
            self.assertFalse(os.path.exists(fx.lock(b)))


# ===================================================================================== §1.4 rows A-J, G, wording rule
class TestRows(Base):
    def test_row_A_partial_recovery(self):
        fx = self.make()
        B, V, txt = self.block(fx)
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["A"])
        self.assertTrue(B["is_A"])
        self.assertEqual((B["labels"]["P1"], B["labels"]["P2"]), (PM.L_REC, PM.L_REC))
        self.assertFalse(B["labels"]["G"])
        self.assertEqual(B["labels"]["P1_word"], PM.W_PART)          # lo95(C4 - C0) < 0
        self.assertAlmostEqual(B["estimates"]["P1"]["point"], 35.0)
        self.assertAlmostEqual(B["estimates"]["P2"]["point"], 30.0)
        self.assertIn(PM.ROW["A"].format(clause="带来 35.0 点的部分性能恢复", lag=16, d2="30.0"), txt)
        self.assertEqual(B["labels"]["D6"], PM.L_HARMLESS)
        for s in PM.LIMITS:
            self.assertIn(s, txt)
        self.assertIn("[J] 不触发：" + PM.ROW["J_same"], txt)
        self.assertIn("PASS（无禁用词）", txt)
        self.assertNotIn("[G]", txt)

    def test_row_A_enhancement(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C0": 80, "C3": 40, "C4": 95, "C5": 50}}})
        B, _, txt = self.block(fx)
        self.assertGreater(B["estimates"]["EN"]["ci95"][0], 0)
        self.assertEqual(B["labels"]["P1_word"], PM.W_ENH)
        self.assertIn(PM.ROW["A"].format(clause="带来 55.0 点的增强", lag=16, d2="45.0"), txt)

    def test_row_A_lag_24(self):
        fx = self.make()
        B, V, txt = self.block(fx, extra=["--lag-steps", "24"])
        self.assertIn("且优于滞后 24 步的轮廓（30.0 点）", txt)

    def test_row_G_does_not_block_and_enhancement(self):
        # F = 0 -> G; P1 = 25, P2 = 35 still judged; C4 above C0 -> 增强
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50,
                                                      "C6": 52, "C7": 60}}})
        B, _, txt = self.block(fx)
        self.assertTrue(B["labels"]["G"])
        self.assertLessEqual(B["estimates"]["F"]["ci95"][0], 0)
        self.assertEqual((B["labels"]["P1"], B["labels"]["P2"]), (PM.L_REC, PM.L_REC))
        self.assertEqual(B["labels"]["P1_word"], PM.W_ENH)
        self.assertEqual(B["row_letters"][:2], ["G", "A"])
        self.assertIn(PM.ROW["G"], txt)
        self.assertIn(PM.NOTE_G, txt)
        self.assertIn("带来 25.0 点的增强", txt)

    def test_row_G_no_enhancement_no_recovery_word(self):
        # F = +8 (b 18, c 10; lo95 <= 0) -> G; P1 = 12 提高; C4 - C0 = +4 (b 10, c 6; lo95 <= 0) -> neither word
        c0 = from_ranges((0, 50), (60, 78))
        c3 = from_ranges((0, 60))
        c4 = from_ranges((0, 72))
        fx = self.make(blocks={"oft_spatial_c30": {"succ": {"C0": c0, "C3": c3, "C4": c4}, "sr": {"C5": 30}}})
        B, _, txt = self.block(fx)
        self.assertAlmostEqual(B["estimates"]["F"]["point"], 8.0)
        self.assertLessEqual(B["estimates"]["F"]["ci95"][0], 0)
        self.assertTrue(B["labels"]["G"])
        self.assertEqual(B["labels"]["P1"], PM.L_REC)
        self.assertLessEqual(B["estimates"]["EN"]["ci95"][0], 0)
        self.assertEqual(B["labels"]["P1_word"], PM.W_G)
        self.assertIn("G", B["row_letters"])
        self.assertIn("A", B["row_letters"])
        self.assertIn(PM.ROW["A"].format(clause="后，C4 高于 C3 12.0 点", lag=16, d2="42.0"), txt)
        self.assertNotIn("部分性能恢复", txt)
        self.assertNotIn("点的增强", txt)

    def test_G_not_triggered_when_lo95_positive(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C3": 85}}})      # F = 10, nested -> lo95 > 0
        B, _, _ = self.block(fx)
        self.assertGreater(B["estimates"]["F"]["ci95"][0], 0)
        self.assertFalse(B["labels"]["G"])
        self.assertNotIn("G", B["row_letters"])

    def test_row_B(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C5": 75}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["P1"], PM.L_REC)
        self.assertIn(B["labels"]["P2"], (PM.L_NOREC, PM.L_UNC))
        self.assertEqual(B["labels"]["D3"], PM.L_REC)
        self.assertEqual(B["row_letters"], ["B"])
        self.assertIn(PM.ROW["B"], txt)
        self.assertIn("§2.4 第 7 项", txt)
        self.assertFalse(B["is_A"])

    def test_row_C(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C4": 45, "C5": 45}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["P1"], PM.L_NOREC)
        self.assertEqual(B["row_letters"], ["C"])
        self.assertIn(PM.ROW["C"].format(d4="47.0"), txt)

    def test_row_D(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C5": 30}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["D3"], PM.L_HARM)
        self.assertEqual(B["row_letters"], ["A", "D"])
        self.assertIn(PM.ROW["D"], txt)

    def test_row_E(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C6": 90}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["row_letters"], ["A", "E"])
        self.assertIn(PM.ROW["E"].format(d5="15.0"), txt)

    def test_rows_E_and_F_no_additive_reading(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C6": 90, "C7": 80}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["D6"], PM.L_HARM)
        self.assertEqual(B["row_letters"], ["A", "E", "F"])
        self.assertIn("[E] " + PM.ROW["E"].format(d5="15.0"), txt)
        self.assertIn("[F] " + PM.ROW["F"].format(d6="15.0") + "\n", txt)  # Spatial: F sentence alone
        self.assertNotIn(PM.F_OBJ, txt)
        self.assertNotIn("须核", txt)
        self.assertNotIn("occ_P(C7)", txt)

    def test_row_J_triggered_and_not(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C1": 70}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["J"]["labels"]["C1"], PM.L_MID)
        self.assertEqual(B["J"]["differ"], ["C1"])
        self.assertEqual(B["J"]["first_round"], PM.FIRST_J["oft_spatial_c30"])
        self.assertIn("J", B["row_letters"])
        self.assertIn(PM.ROW["J"].format(f1="几乎不掉", f2="中间", f3="大幅掉分", c1="中间", c2="中间", c3="大幅掉分"), txt)
        self.assertNotIn(PM.ROW["J_same"], txt)
        self.assertNotIn("模式", txt)
        # both rounds side by side: first-round D(C1) of oft_spatial = +1.0 [-2.0, +5.0]
        self.assertIn("D(C1) 首轮 +1.0 [-2.0, +5.0] 几乎不掉；确认轮 -25.0", txt)
        fx2 = self.make()
        B2, _, txt2 = self.block(fx2)
        self.assertEqual(B2["J"]["labels"], PM.FIRST_J["oft_spatial_c30"])
        self.assertNotIn("J", B2["row_letters"])
        self.assertIn("[J] 不触发：三档标签与 ep 20-29 相同；这不说明两轮效应的大小关系", txt2)

    def test_residual_combination_no_row(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C4": 60, "C5": 52}}})   # P1 15 提高, P2 8, D3 7
        B, _, txt = self.block(fx)
        self.assertEqual(B["labels"]["P1"], PM.L_REC)
        self.assertNotEqual(B["labels"]["P2"], PM.L_REC)
        self.assertNotEqual(B["labels"]["D3"], PM.L_REC)
        self.assertEqual(B["row_letters"], [])
        self.assertIn("不属于 A / B / C 行", txt)

    def test_row_I_two_spatial_A(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "ofti_spatial_c30": {}})
        V, txt = self.ok_run(fx, "oft_spatial_c30,ofti_spatial_c30")
        self.assertEqual(V["summary_I"]["spatial"], PM.ROW["I_both"])
        self.assertEqual(V["summary_I"]["oft_object_c30"], "未读")
        self.assertIn("ofti_object_c30: 未读", txt)

    def test_row_I_one_A_and_object_cases(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "ofti_spatial_c30": {"sr": {"C4": 45, "C5": 45}},
                               "oft_object_c30": {}, "ofti_object_c30": {"sr": {"C4": 45, "C5": 45}}})
        V, txt = self.ok_run(fx, "oft_spatial_c30,ofti_spatial_c30,oft_object_c30,ofti_object_c30")
        I = V["summary_I"]
        self.assertTrue(I["spatial"].startswith(PM.ROW["I_one"].format(x="合训 OFT")))
        self.assertIn("ofti_spatial_c30", I["spatial"])
        self.assertEqual(I["oft_object_c30"], PM.ROW["I_obj_ok"])
        self.assertTrue(I["ofti_object_c30"].startswith(PM.ROW["I_obj_ii"]))     # (ii): F > 0, P1 未提高

    def test_row_I_object_case_i(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "oft_object_c30": {"sr": {"C3": 95, "C4": 95, "C5": 95}}})
        V, _ = self.ok_run(fx, "oft_spatial_c30,oft_object_c30")
        self.assertTrue(V["blocks"]["oft_object_c30"]["labels"]["G"])
        self.assertTrue(V["summary_I"]["oft_object_c30"].startswith(PM.ROW["I_obj_i"]))

    def test_row_I_object_without_spatial(self):
        fx = self.make(blocks={"oft_object_c30": {}})
        V, _ = self.ok_run(fx, "oft_object_c30")
        self.assertIn("oft_spatial_c30：未读", V["summary_I"]["oft_object_c30"])
        self.assertEqual(V["summary_I"]["spatial"], "两个 Spatial 块均未读")


# ===================================================================================== H rows and gates
class TestGates(Base):
    def test_cell_fail_marks_comparisons_H(self):
        fx = self.make(gate_cells={"oft_spatial_c30": {"C4": {"status": "FAIL", "reasons": ["首帧不一致"]}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["H"])
        self.assertIsNone(B["labels"]["P1"])
        self.assertIsNone(B["labels"]["P2"])
        self.assertIsNone(B["labels"]["D4"])
        self.assertEqual(B["labels_by_line"]["P1"], PM.L_REC)
        self.assertIn("格门槛未过（H 行）", txt)
        self.assertIn("按线本应：提高", txt)
        self.assertEqual(B["mcnemar_p_for_holm"]["D4"], 1.0)

    def test_cell_unk_C3(self):
        fx = self.make(gate_cells={"oft_spatial_c30": {"C3": {"status": "UNK"}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["H"])
        self.assertIsNone(B["labels"]["G"])
        self.assertIsNone(B["labels"]["P1"])
        self.assertIsNone(B["labels"]["D3"])
        self.assertEqual(B["labels"]["P2"], PM.L_REC)          # C4 vs C5 unaffected
        self.assertFalse(B["J"]["judged"])

    def test_cell_fail_C0_word_not_judged(self):
        fx = self.make(gate_cells={"oft_spatial_c30": {"C0": {"status": "FAIL"}}})
        B, _, txt = self.block(fx)
        self.assertIsNone(B["labels"]["G"])
        self.assertIsNone(B["labels"]["P1_word"])
        self.assertEqual(B["row_letters"], ["H", "A"])
        self.assertIn("C4 高于 C3 35.0 点（用词不判：C0 门槛未过）", txt)

    def test_block_ok_false(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"block_ok": False, "block_fail_reasons": ["IK 门槛"]}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertEqual(B["row_letters"], ["H"])
        self.assertTrue(all(v is None for v in B["labels"].values()))
        self.assertIn("整块门槛未过", txt)
        self.assertIn("IK 门槛", txt)
        self.assertFalse(B["is_A"])

    def test_card_type_mixed(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"card_type": "mixed"}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertIn("card_type=mixed", txt)

    def test_card_type_null(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"card_type": None}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["status"], "H")

    def test_dump_gpu_disagrees_with_gate(self):
        fx = self.make(gpu={"oft_spatial_c30": {"C6": "NVIDIA H100 NVL|95830 MiB"}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertEqual(B["gate"]["dump_cards"], ["h100-96", "h200"])
        self.assertEqual(B["gpu"]["C6"], {"NVIDIA H100 NVL|95830 MiB": 100})

    def test_a100_block_is_H(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"card_type": "a100-80"}},
                       gpu={"oft_spatial_c30": {c: "NVIDIA A100 80GB PCIe|81920 MiB" for c in CELLS}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["status"], "H")

    def test_h100_96_block_allowed(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"card_type": "h100-96"}},
                       gpu={"oft_spatial_c30": {c: "NVIDIA H100 NVL|95830 MiB" for c in CELLS}})
        B, _, _ = self.block(fx)
        self.assertNotEqual(B["status"], "H")

    def test_gate_protocol_sha_mismatch(self):
        fx = self.make(gate_sha="0" * 64)
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertIn("不符", txt)

    def test_bad_keys_union_removed_from_all_cells(self):
        fx = self.make(gate_cells={"oft_spatial_c30": {"C2": {"bad_keys": [[0, 30], [1, 31]]}, "C7": {"bad_keys": [[2, 32]]}}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["n_universe"], 97)
        for k in ("F", "P1", "D5", "D6"):
            self.assertEqual(B["estimates"][k]["n"], 97)


# ===================================================================================== sensitivity
class TestSensitivity(Base):
    def test_downgrade_when_label_changes(self):
        phys = {"oft_spatial_c30": {"C4": ORDER[45:55]}}
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C4": 57}}}, phys=phys)
        B, _, txt = self.block(fx)
        self.assertEqual(B["sensitivity"]["n"], 90)
        self.assertIn("P1", B["sensitivity"]["changed"])
        self.assertTrue(B["sensitivity"]["downgraded"])
        self.assertEqual(B["status"], "不确定")
        self.assertNotIn("A", B["row_letters"])
        self.assertFalse(B["is_A"])
        self.assertIn("该块降为「不确定」", txt)
        self.assertIn("P1 Δ1", txt)
        # main read-out is not trimmed
        self.assertEqual(B["n_universe"], 100)
        self.assertAlmostEqual(B["estimates"]["P1"]["point"], 12.0)

    def test_no_downgrade_when_labels_stable(self):
        phys = {"oft_spatial_c30": {"C2": [(0, 30)], "C5": [(3, 37)]}}
        fx = self.make(phys=phys)
        B, _, txt = self.block(fx)
        self.assertEqual(B["sensitivity"]["n"], 98)
        self.assertEqual(B["sensitivity"]["changed"], [])
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["A"])

    def test_no_physics_keys(self):
        fx = self.make()
        B, _, txt = self.block(fx)
        self.assertIn("无失效键", txt)
        self.assertFalse(B["sensitivity"]["downgraded"])


# ===================================================================================== guards
class TestGuards(Base):
    def test_incomplete_block_refused(self):
        fx = self.make(drop={"oft_spatial_c30": {"C5": [(3, 37)]}})
        p = fx.run()
        self.refused(fx, p, 8)
        self.assertIn("C5 片 ep35-39 缺 1 键", p.stderr)

    def test_incomplete_one_of_two_blocks_refuses_all(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "ofti_spatial_c30": {}}, drop={"ofti_spatial_c30": {"C0": [(0, 30)]}})
        p = fx.run("oft_spatial_c30,ofti_spatial_c30")
        self.refused(fx, p, 8, ("oft_spatial_c30", "ofti_spatial_c30"))

    def test_missing_field_refused(self):
        fx = self.make(omit_field=("oft_spatial_c30", "C2", (4, 34), "mi_obj_pos"))
        p = fx.run()
        self.refused(fx, p, 9)
        self.assertIn("mi_obj_pos", p.stderr)

    def test_duplicate_key_refused(self):
        fx = self.make(extra_file=("oft_spatial_c30", "C1", "t0_e30_i99.npz"))
        p = fx.run()
        self.refused(fx, p, 9)

    def test_files_outside_key_set_fail_the_cell(self):
        # confirm §4 item 9 / K1: an ep 20 file (first-round key) in C1 -> C1 fails, comparisons with C1 are H
        fx = self.make(extra_file=("oft_spatial_c30", "C1", "t0_e20_i20.npz"))
        B, _, txt = self.block(fx)
        self.assertIn("之外的文件 1 个", txt)
        self.assertIn("C1", B["gate"]["cell_h"])
        self.assertIn("键集核 fail", B["gate"]["cell_h"]["C1"])
        self.assertIsNone(B["labels"]["D4"])
        self.assertFalse(B["J"]["judged"])
        self.assertEqual(B["labels"]["P1"], PM.L_REC)             # C1 not involved
        fx2 = self.make(extra_file=("oft_spatial_c30", "C4", "t0_e40_i40.npz"))
        B2, _, _ = self.block(fx2)
        self.assertIsNone(B2["labels"]["P1"])
        self.assertEqual(B2["R"]["case"], "R-H")

    def test_lock_and_refuse_rerun(self):
        fx = self.make()
        self.ok_run(fx)
        self.assertTrue(os.path.exists(fx.lock("oft_spatial_c30")))
        before = (rtext(fx.out), rtext(fx.vj))
        p = fx.run()
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertEqual((rtext(fx.out), rtext(fx.vj)), before)

    def test_existing_lock_refused(self):
        fx = self.make()
        open(fx.lock("oft_spatial_c30"), "w").close()
        p = fx.run()
        self.assertEqual(p.returncode, 5)
        self.assertFalse(os.path.exists(fx.out))

    def test_append_second_block_keeps_first(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "ofti_spatial_c30": {"sr": {"C3": 95}}})
        V1, t1 = self.ok_run(fx, "oft_spatial_c30")
        b1 = json.dumps(V1["blocks"]["oft_spatial_c30"], sort_keys=True)
        V2, t2 = self.ok_run(fx, "ofti_spatial_c30")
        self.assertTrue(t2.startswith(t1))
        self.assertEqual(json.dumps(V2["blocks"]["oft_spatial_c30"], sort_keys=True), b1)
        self.assertTrue(V2["blocks"]["ofti_spatial_c30"]["labels"]["G"])
        self.assertIn("G", V2["blocks"]["ofti_spatial_c30"]["row_letters"])
        self.assertEqual(len(V2["runs"]), 2)
        self.assertEqual(t2.count(PM.block_header("oft_spatial_c30") + "（"), 1)
        self.assertIn("在 合训 OFT checkpoint 上", V2["summary_I"]["spatial"])
        p = fx.run("ofti_spatial_c30")
        self.assertEqual(p.returncode, 3)

    def test_out_without_verdict_refused(self):
        fx = self.make()
        open(fx.out, "w").close()
        p = fx.run()
        self.assertEqual(p.returncode, 3)
        self.assertFalse(os.path.exists(fx.lock("oft_spatial_c30")))

    def test_deviations_sha_printed(self):
        fx = self.make()
        dev = os.path.join(fx.base, "protocol_maskinfo_deviations.md")
        with open(dev, "w") as f:
            f.write("偏离 1：测试\n")
        with open(dev, "rb") as f:
            dsha = hashlib.sha256(f.read()).hexdigest()
        V, txt = self.ok_run(fx, extra=["--deviations", dev])
        self.assertIn(f"偏离文件 sha256 {dsha}", txt)
        self.assertEqual(V["runs"][-1]["deviations"]["sha256"], dsha)
        self.assertEqual(V["blocks"]["oft_spatial_c30"]["row_letters"], ["A"])

    def test_deviations_missing_file_noted(self):
        fx = self.make()
        V, txt = self.ok_run(fx, extra=["--deviations", os.path.join(fx.base, "nope.md")])
        self.assertIn("偏离文件 sha256 不存在", txt)

    def test_sha_mismatch(self):
        fx = self.make()
        p = fx.run(sha="f" * 64)
        self.refused(fx, p, 2)

    def test_not_before(self):
        fx = self.make()
        p = fx.run(extra=["--now", "2026-11-12T00:00:00+08:00", "--not-before", "2026-11-13T12:00:00+08:00"])
        self.refused(fx, p, 4)
        V, _ = self.ok_run(fx, extra=["--now", "2026-11-13T12:00:00+08:00", "--not-before", "2026-11-13T12:00:00+08:00"])
        self.assertEqual(V["blocks"]["oft_spatial_c30"]["read_time"], "2026-11-13T12:00:00+08:00")

    def test_not_before_real_clock(self):
        fx = self.make()
        p = fx.run(extra=["--not-before", "2099-01-01T00:00:00+00:00"])
        self.refused(fx, p, 4)

    def test_now_requires_synthetic(self):
        fx = self.make()
        p = fx.run(extra=["--now", "2026-11-13T12:00:00+08:00"], synthetic=False)
        self.refused(fx, p, 4)

    def test_synthetic_requires_marker(self):
        fx = self.make(marker=False)
        p = fx.run()
        self.refused(fx, p, 6)

    def test_unknown_block_name(self):
        fx = self.make()
        p = fx.run("oft_spatial_c30,foo")
        self.refused(fx, p, 9)

    def test_wording_hit_exit_7(self):
        fx = self.make(gate_block={"oft_spatial_c30": {"block_ok": False, "block_fail_reasons": ["疑似视觉伺服"]}})
        p = fx.run()
        self.assertEqual(p.returncode, 7, p.stderr)
        txt = rtext(fx.out)
        self.assertIn("FAIL 命中 视觉伺服", txt)
        V = jload(fx.vj)
        self.assertEqual(V["runs"][-1]["wording_check"]["status"], "FAIL")
        self.assertEqual(V["blocks"]["oft_spatial_c30"]["status"], "H")


# ===================================================================================== numerics and timing
class TestNumerics(Base):
    def test_mcnemar_and_boot_match_maskrepl(self):
        spec = importlib.util.spec_from_file_location("prereg_maskrepl_ref", MASKREPL)
        mr = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mr)
        for b, c in [(5, 25), (0, 0), (16, 5), (12, 0)]:
            self.assertEqual(PM.mcnemar_exact(b, c), mr.mcnemar_exact(b, c))
        rng = np.random.default_rng(5)
        Y = {"a": rng.integers(0, 2, 100).astype(float), "b": rng.integers(0, 2, 100).astype(float)}
        Y["a"][[3, 7]] = np.nan
        r1, r2 = PM.Boot(100, 2000).diff(Y, "a", "b"), mr.Boot(100, 2000).diff(Y, "a", "b")
        self.assertEqual(r1, r2)
        try:
            from scipy.stats import binomtest
            self.assertAlmostEqual(PM.mcnemar_exact(5, 25), binomtest(5, 30, 0.5).pvalue, places=10)
        except ImportError:
            pass

    def test_holm(self):
        h = PM.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
        self.assertAlmostEqual(h["a"], 0.04)
        self.assertAlmostEqual(h["c"], 0.09)
        self.assertAlmostEqual(h["b"], 0.09)
        self.assertAlmostEqual(h["d"], 0.5)

    def test_label_rules(self):
        r = lambda pt, lo, hi: {"point": pt, "ci95": (lo, hi)}
        self.assertEqual(PM.ci_label("rec", r(10, 0.1, 20)), PM.L_REC)
        self.assertEqual(PM.ci_label("rec", r(10, 0.0, 20)), PM.L_UNC)
        self.assertEqual(PM.ci_label("rec", r(9.9, 1, 20)), PM.L_UNC)
        self.assertEqual(PM.ci_label("rec", r(3, -2, 9.99)), PM.L_NOREC)
        self.assertEqual(PM.ci_label("rec_harm", r(-10, -20, -0.1)), PM.L_HARM)
        self.assertEqual(PM.ci_label("rec_harm", r(-10, -20, 0.0)), PM.L_NOREC)
        self.assertEqual(PM.ci_label("harm_harmless", r(-2, -9.9, 5)), PM.L_HARMLESS)
        self.assertEqual(PM.ci_label("harm_harmless", r(-5, -10, 5)), PM.L_UNC)
        self.assertEqual(PM.ci_label("harm_harmless", r(-10, -18, -1)), PM.L_HARM)
        self.assertTrue(PM.g_active(r(5, 0.0, 10)))
        self.assertFalse(PM.g_active(r(5, 0.01, 10)))
        self.assertEqual(PM.p1_word(PM.L_REC, r(5, 0.01, 9), False), PM.W_ENH)
        self.assertEqual(PM.p1_word(PM.L_REC, r(5, 0.01, 9), True), PM.W_ENH)
        self.assertEqual(PM.p1_word(PM.L_REC, r(5, 0.0, 9), False), PM.W_PART)
        self.assertEqual(PM.p1_word(PM.L_REC, r(5, 0.0, 9), True), PM.W_G)
        self.assertEqual(PM.W_G, "提高")
        self.assertIsNone(PM.p1_word(PM.L_UNC, r(5, 1, 9), False))
        self.assertEqual(PM.j_label(r(-4, -10, 2)), PM.L_NO)
        self.assertEqual(PM.j_label(r(-50, -60, -30)), PM.L_BIG)
        self.assertEqual(PM.j_label(r(-20, -28, -12)), PM.L_MID)

    def test_holm_gates_secondary_label(self):
        # Δ5: C6 only on 16 keys, C2 only on 5 -> point 11, CI above 0, raw p 0.027, Holm-adjusted > 0.05
        c6 = from_ranges((0, 70))
        c2 = from_ranges((0, 54), (70, 75))
        fx = self.make(blocks={"oft_spatial_c30": {"succ": {"C6": c6, "C2": c2}}})
        B, _, txt = self.block(fx)
        e = B["estimates"]["D5"]
        self.assertAlmostEqual(e["point"], 11.0)
        self.assertEqual((e["b"], e["c"]), (16, 5))
        self.assertGreater(e["ci95"][0], 0)
        self.assertEqual(B["labels_ci_only"]["D5"], PM.L_REC)
        self.assertGreaterEqual(B["holm_p"]["D5"], 0.05)
        self.assertEqual(B["labels"]["D5"], PM.L_UNC_HOLM)
        self.assertIn("不确定（Holm 未过）", txt)
        self.assertNotIn("E", B["row_letters"])

    def test_wording_checker(self):
        self.assertEqual(PM.wording_violations("该策略做视觉伺服"), ["视觉伺服"])
        self.assertIn("模型利用轮廓", PM.wording_violations("说明模型利用轮廓"))
        self.assertIn("必要", PM.wording_violations("轮廓是必要的"))
        self.assertIn("充分", PM.wording_violations("对控制是充分的"))
        self.assertIn("面积效应", PM.wording_violations("在面积效应之外，"))
        self.assertIn("过早合拢", PM.wording_violations("C3 过早合拢"))
        self.assertIn("抓稳", PM.wording_violations("抓稳比例"))
        self.assertIn("轮廓机制未复现", PM.wording_violations("Object 上轮廓机制未复现"))
        self.assertIn("混杂", PM.wording_violations("本体感受混杂"))
        for k, s in PM.ROW.items():
            self.assertEqual(PM.wording_violations(s), [], k)
        for s in PM.LIMITS + [PM.NOTE_B, PM.NOTE_G, PM.L_UNC_HOLM, PM.F_OBJ] + list(PM.A_CLAUSE.values()) + list(PM.CNAME.values()):
            self.assertEqual(PM.wording_violations(s), [], s)

    def test_episode_events(self):
        cases = {"no_reach": (0, 20), "no_grasp": (0, 21), "drop": (0, 22), "other": (0, 23)}
        for ft, kk in cases.items():
            self.assertEqual(ftype_of(kk), ft)
        for suite in ("spatial", "object"):
            ev = PM.episode_events(dict(episode(True, (0, 20), (20, 22, 30)), success=np.array(True)), suite)
            self.assertEqual({k: ev[k] for k in PM.EVENTS},
                             {"t_near": 20, "t_close": 22, "t_pad1": 23, "t_pad2": 24, "t_lift": 30, "t_carry": 30,
                              "t_drop": None})
            self.assertEqual(ev["ftype"], "success")
            exp = {"no_reach": dict(t_near=None, t_pad1=None, t_pad2=None, t_lift=None, t_carry=None, t_drop=None),
                   "no_grasp": dict(t_near=20, t_pad1=23, t_pad2=None, t_lift=None, t_carry=None, t_drop=None),
                   "drop": dict(t_near=20, t_pad1=23, t_pad2=24, t_lift=30, t_carry=30, t_drop=50),
                   "other": dict(t_near=20, t_pad1=23, t_pad2=24, t_lift=30, t_carry=None, t_drop=None)}
            for ft, kk in cases.items():
                ev = PM.episode_events(dict(episode(False, kk, (20, 22, 30)), success=np.array(False)), suite)
                self.assertEqual(ev["ftype"], ft)
                self.assertEqual(ev["t_close"], 22)
                for k, v in exp[ft].items():
                    self.assertEqual(ev[k], v, (suite, ft, k))

    def test_carry_needs_16_steps_and_terminal_release_is_not_drop(self):
        z = dict(episode(False, (0, 22), (20, 22, 30)), success=np.array(False))
        lp = z["mi_contact_lpad"].copy()
        lp[30 + 15] = False                      # break the window at step 45 -> first window starts at 46
        z2 = dict(z, mi_contact_lpad=lp)
        ev = PM.episode_events(z2, "spatial")
        self.assertIsNone(ev["t_carry"])         # 46 .. 49 < 16 steps before release at 50
        # release over the target (success trajectory) is not a drop even if not flagged success
        zs = dict(episode(True, (0, 20), (20, 22, 30)), success=np.array(False))
        ev = PM.episode_events(zs, "spatial")
        self.assertEqual(ev["t_carry"], 30)
        self.assertIsNone(ev["t_drop"])
        self.assertEqual(ev["ftype"], "other")
        # the same release far from the reference object is a drop
        zf = dict(zs, mi_obj_pos=zs["mi_obj_pos"] + np.array([[0, 0, 0], [1.0, 0, 0]], np.float32))
        self.assertEqual(PM.episode_events(zf, "object")["t_drop"], 70)

    def test_drop_needs_4_step_bilateral_loss(self):
        zs = dict(episode(True, (0, 20), (20, 22, 30)), success=np.array(True))   # carry 30..69, release on target at 70
        for gap, exp in [(3, None), (4, 50)]:
            lp = zs["mi_contact_lpad"].copy()
            lp[50:50 + gap] = False              # bilateral loss mid-carry, object off target
            ev = PM.episode_events(dict(zs, mi_contact_lpad=lp), "spatial")
            self.assertEqual(ev["t_carry"], 30)
            self.assertEqual(ev["t_drop"], exp, gap)
        # a 3-step loss is skipped, the next 5-step loss is the drop
        lp = zs["mi_contact_lpad"].copy()
        lp[48:51] = False
        lp[55:60] = False
        self.assertEqual(PM.episode_events(dict(zs, mi_contact_lpad=lp), "spatial")["t_drop"], 55)
        # loss run that is terminal from its start (release on the plate) is skipped
        self.assertIsNone(PM.episode_events(zs, "spatial")["t_drop"])
        # a 3-step run truncated at the episode end does not count
        zd = dict(episode(False, (0, 22), (20, 22, 30)), success=np.array(False))
        T = len(zd["action"])
        lp = np.ones(T + 1, bool)
        rp = np.ones(T + 1, bool)
        rp[T - 2:] = False
        sup = zd["mi_contact_target_support"].copy()
        sup[:] = True
        sup[30:] = False
        obj = zd["mi_obj_pos"].copy()
        grip = zd["mi_grip_pos"].copy()
        for j in range(30, T + 1):
            obj[j, 0] = [0.1 + 0.001 * (j - 30), 0.2, 0.95]
            grip[j] = obj[j, 0] + [0.02, 0, 0]
        ev = PM.episode_events(dict(zd, mi_contact_lpad=lp, mi_contact_rpad=rp, mi_contact_target_support=sup,
                                    mi_obj_pos=obj, mi_grip_pos=grip), "spatial")
        self.assertEqual(ev["t_carry"], 30)
        self.assertIsNone(ev["t_drop"])

    def test_timing_section(self):
        fx = self.make(blocks={"oft_spatial_c30": {"timing": {"C3": (20, 14, 30), "C4": (20, 22, 26)}}})
        B, _, txt = self.block(fx)
        T = B["timing"]
        succ0 = fx.succ["oft_spatial_c30"]["C0"]
        fts0 = {f: 0 for f in PM.FTYPES}
        for k in KEYS:
            fts0["success" if succ0[k] else ftype_of(k)] += 1
        self.assertEqual(T["C0"]["ftypes"], fts0)
        self.assertEqual(T["C0"]["intervals"]["close_minus_near"]["median"], 2.0)
        self.assertEqual(T["C0"]["intervals"]["pad2_minus_close"]["median"], 2.0)
        self.assertEqual(T["C0"]["intervals"]["carry_minus_pad2"]["median"], 6.0)
        self.assertAlmostEqual(T["C0"]["outside_near_close_pct"], 100 * fts0["no_reach"] / 100)
        self.assertAlmostEqual(T["C0"]["events"]["t_carry"]["pct"], fts0["success"] + fts0["drop"])
        self.assertEqual(T["C0"]["events"]["t_drop"]["n"], fts0["drop"])
        self.assertEqual(T["C3"]["intervals"]["close_minus_near"]["median"], -6.0)
        self.assertAlmostEqual(T["C3"]["outside_near_close_pct"], 100.0)
        cm = T["C3"]["vs_C0"]["close_minus_near"]
        self.assertAlmostEqual(cm["point"], -8.0)
        self.assertEqual(cm["ci95"], [-8.0, -8.0])
        n_both = sum(1 for k in KEYS if (succ0[k] or ftype_of(k) != "no_reach")
                     and (fx.succ["oft_spatial_c30"]["C3"][k] or ftype_of(k) != "no_reach"))
        self.assertEqual(cm["n"], n_both)
        self.assertAlmostEqual(T["C4"]["vs_C0"]["t_lift"]["point"], -4.0)
        self.assertAlmostEqual(T["C4"]["vs_C0"]["carry_minus_pad2"]["point"], -4.0)
        self.assertEqual(T["C0"]["events"]["t_lift"]["median"], 30.0)
        self.assertIn("§5.4", txt)
        self.assertNotIn("过早", txt)

    def test_gpu_table_and_header(self):
        fx = self.make()
        B, V, txt = self.block(fx)
        self.assertEqual(B["gpu"]["C0"], {A100: 100})
        self.assertIn(f"协议 sha256 {PSHA}", txt)
        self.assertIn("脚本 sha256", txt)
        self.assertIn("门槛 json sha256", txt)
        self.assertIn("读数真实时刻", txt)
        self.assertIn("本次读的块 ['oft_spatial_c30']", txt)


# ===================================================================================== confirm additions (§3, §4, §7)
import ast  # noqa: E402
import types  # noqa: E402

ROOT = os.path.dirname(HERE)
EVAL = os.path.join(ROOT, "oft", "run_libero_eval_mi_c.py")
GATE = os.path.join(ROOT, "tools", "gate_maskinfo_confirm.py")
FIRST = PM.load_first_round()[0]


def r_(pt, lo, hi):
    return {"point": float(pt), "ci95": (float(lo), float(hi))}


def written_fields():
    """Field names the confirm eval script writes into a dump (static parse; deviations-file lesson 10 (b)):
    keywords of np.savez_compressed in _dump_episode, of mi_extra = dict(...), of MaskHook.dump_fields (dict literal
    string keys) and MaskInfoHook.dump_fields (out.update keywords)."""
    names = set()
    with open(EVAL) as g:
        t = ast.parse(g.read())
    for n in ast.walk(t):
        if isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "savez_compressed":
            names |= {k.arg for k in n.keywords if k.arg}
        if isinstance(n, ast.Assign) and any(getattr(x, "id", None) == "mi_extra" for x in n.targets) \
                and isinstance(n.value, ast.Call):
            names |= {k.arg for k in n.value.keywords if k.arg}
    for f, kind in ((os.path.join(ROOT, "embody", "maskrepl.py"), "dict"), (os.path.join(ROOT, "embody", "maskinfo.py"), "update")):
        with open(f) as g:
            t = ast.parse(g.read())
        for fn in ast.walk(t):
            if isinstance(fn, ast.FunctionDef) and fn.name == "dump_fields":
                for n in ast.walk(fn):
                    if kind == "dict" and isinstance(n, ast.Dict):
                        names |= {k.value for k in n.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}
                    if kind == "update" and isinstance(n, ast.Call) and getattr(n.func, "attr", None) == "update":
                        names |= {k.arg for k in n.keywords if k.arg}
    return names


class TestConfirmFields(Base):
    def test_required_fields_subset_of_written(self):
        w = written_fields()
        self.assertIn("mi_cell", w)          # confirm §2 item 3: the eval copy writes the label explicitly
        self.assertIn("mr_cell", w)
        self.assertIn("mi_block", w)
        self.assertTrue(set(PM.REQUIRED_FIELDS) <= w, set(PM.REQUIRED_FIELDS) - w)

    def test_fixture_fields_subset_of_written(self):
        fx = self.make()
        f = os.path.join(fx.runs, "oft_spatial_c30", "C0", "t0_e30_i30.npz")
        with np.load(f) as z:
            fields = set(z.files)
        self.assertTrue(fields <= written_fields(), fields - written_fields())

    def test_check_fields_only_ok_and_missing(self):
        fx = self.make(omit_field=("oft_spatial_c30", "C2", (4, 34), "mi_cell"))
        ok = [os.path.join(fx.runs, "oft_spatial_c30", c, "t0_e30_i30.npz") for c in CELLS]
        p = subprocess.run([sys.executable, "-I", SCRIPT, "--check-fields-only"] + ok, capture_output=True, text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("CHECK_FIELDS_ONLY OK 8", p.stdout)
        self.assertNotIn("success", p.stdout + p.stderr)
        bad = os.path.join(fx.runs, "oft_spatial_c30", "C2", "t4_e34_i34.npz")
        p = subprocess.run([sys.executable, "-I", SCRIPT, "--check-fields-only"] + ok + [bad], capture_output=True, text=True)
        self.assertEqual(p.returncode, 9)
        self.assertIn("缺字段 ['mi_cell']", p.stderr)
        self.assertFalse(os.path.exists(fx.out))
        self.assertFalse(os.path.exists(fx.lock("oft_spatial_c30")))

    def test_check_fields_only_smoke_keys_and_bad_path(self):
        # smoke uses already-read keys (ep 20 / ep 0): fields-only does not apply the key-set rule
        fx = self.make(extra_file=("oft_spatial_c30", "C1", "t0_e20_i20.npz"))
        f = os.path.join(fx.runs, "oft_spatial_c30", "C1", "t0_e20_i20.npz")
        p = subprocess.run([sys.executable, "-I", SCRIPT, "--check-fields-only", f], capture_output=True, text=True)
        self.assertEqual(p.returncode, 9)       # the copied npz says ep 30 / init 30: file name mismatch is caught
        self.assertIn("task / ep / init 与文件名不符", p.stderr)
        d = os.path.join(fx.base, "oft_spatial", "C0")   # first-round block name is not a confirm block
        os.makedirs(d)
        shutil.copy(os.path.join(fx.runs, "oft_spatial_c30", "C0", "t0_e30_i30.npz"), d)
        p = subprocess.run([sys.executable, "-I", SCRIPT, "--check-fields-only", os.path.join(d, "t0_e30_i30.npz")],
                           capture_output=True, text=True)
        self.assertEqual(p.returncode, 9)

    def test_three_way_label_mismatch_refused(self):
        fx = self.make(override=("oft_spatial_c30", "C3", (2, 33), "mr_cell", "C4"))
        p = fx.run()
        self.refused(fx, p, 9)
        self.assertIn("mi_cell / mr_cell / mi_block 与路径不符", p.stderr)
        fx2 = self.make(omit_field=("oft_spatial_c30", "C3", (2, 33), "mr_cell"))
        self.refused(fx2, fx2.run(), 9)
        fx3 = self.make(override=("oft_spatial_c30", "C3", (2, 33), "mi_block", "oft_spatial"))
        self.refused(fx3, fx3.run(), 9)

    def test_protocol_sha_constant(self):
        self.assertEqual(PSHA, PM.PROTOCOL_SHA)
        fx = self.make()
        other = os.path.join(fx.base, "other_protocol.md")
        with open(other, "w") as f:
            f.write("not the frozen protocol\n")
        with open(other, "rb") as g:
            osha = hashlib.sha256(g.read()).hexdigest()
        a = fx.args("oft_spatial_c30", sha=osha)
        a[a.index("--protocol") + 1] = other
        p = subprocess.run([sys.executable, "-I", SCRIPT] + a, capture_output=True, text=True)
        self.refused(fx, p, 2)
        self.assertIn("PROTOCOL_SHA", p.stderr)

    def test_first_round_file_sha_checked(self):
        d = tempfile.mkdtemp(dir=TMPROOT if os.path.isdir(TMPROOT) else None)
        self.addCleanup(shutil.rmtree, d, True)
        f = os.path.join(d, "first_round_maskinfo.json")
        with open(PM.FIRST_ROUND_FILE) as g:
            txt = g.read()
        with open(f, "w") as g:
            g.write(txt.replace('"P1_word": "部分恢复"', '"P1_word": "增强"', 1))
        with self.assertRaises(PM.Refuse) as cm:
            PM.load_first_round(f)
        self.assertEqual(cm.exception.code, 9)
        fr, sha = PM.load_first_round()
        self.assertEqual(sha, PM.FIRST_ROUND_SHA)
        self.assertEqual(fr["blocks"]["oft_object"]["estimates"]["P2"]["ci95"], [5.0, 17.0])
        self.assertEqual(fr["blocks"]["ofti_object"]["successes"], {"C2": 0, "C3": 0, "C6": 0})
        for b, b0 in PM.FIRST_OF.items():
            self.assertTrue(fr["blocks"][b0]["is_A"])

    def test_forbidden_additions_and_templates(self):
        for w in ["假阳性", "挑了好的一组键", "效应量复现", "效应量相同", "效应量相等", "机制得到确认", "真实成功概率为零",
                  "两组键的差异来自"]:
            self.assertIn(w, PM.wording_violations(f"前文{w}后文"), w)
        tmpl = list(PM.R_TXT.values()) + list(PM.ROW.values()) + [PM.F_OBJ]
        for t in tmpl:
            self.assertNotIn("复现失败", t)
            self.assertNotIn("未复现", t)
            self.assertNotIn("模式不一致", t)
        for k, t in PM.R_TXT.items():
            self.assertEqual(PM.wording_violations(t), [], k)
        self.assertEqual(PM.wording_violations(PM.ROW["J_same"]), [])   # protocol v1.1
        with open(SCRIPT) as g:
            code = g.read()
        self.assertNotIn("须核", code.split("# ============================================================================ output buffer")[1])


class TestConfirmSection3(Base):
    """§3 R-A / R-B (i)-(iv) / G / R-H / R-0 / R-C."""

    def rsec(self, labels, est, b="oft_spatial_c30", status="judged", block_h=(), cell_h=None, n=100, Y=None):
        a = types.SimpleNamespace(first_round=FIRST, lag_steps=16)
        PM.P.lines.clear()
        PM.P.quiet = True
        lab = dict(G=False, P1_word=PM.W_PART, D4=PM.L_UNC)
        lab.update(labels)
        e = dict(F=r_(30, 20, 40), D4=r_(5, -2, 12))
        e.update(est)
        Y = Y or {c: np.ones(n) for c in CELLS}
        rec = PM.section_R(a, b, status, list(block_h), cell_h or {}, {"labels": lab, "est": e}, list(range(n)), Y)
        txt = PM.P.text()
        PM.P.lines.clear()
        self.assertNotIn("复现失败", txt)
        self.assertNotIn("未复现", txt)
        self.assertNotIn("假阳性", txt)
        return rec, txt

    def test_RA_full_run(self):
        fx = self.make()
        B, V, txt = self.block(fx)
        self.assertEqual(B["R"]["case"], "R-A")
        self.assertEqual(B["R"]["category"], PM.RC_REACH)
        self.assertIn("[R-A] 在该 checkpoint 与套件上，两组各 100 个初始化均达到预设合取标准：在 alpha0 图像上添加保护对象后"
                      "剩余的当前可见轮廓，提高成功率，并优于滞后 16 步轮廓。两轮效应量及区间分别报告：Δ1 首轮 +38.0 [+27.0, +49.0]"
                      "、确认轮 +35.0 [", txt)
        self.assertIn("Δ2 首轮 +42.0 [+32.0, +52.0]、确认轮 +30.0 [", txt)
        self.assertIn("后一组在协议冻结后前瞻性采集。", txt)
        self.assertIn(PM.R_TXT["A_may"], txt)
        self.assertIn("P1 用词（§1.3）：部分恢复", txt)
        self.assertIn("P1 Δ1 = SR(C4) − SR(C3)：首轮 +38.0 [+27.0, +49.0] 提高；确认轮 +35.0", txt)
        self.assertIn("lo95(F)：首轮 +44.0；确认轮", txt)
        self.assertIn("[R-C] 首轮 4/4 达到 A（oft_spatial、ofti_spatial、oft_object、ofti_object）；确认轮 1/4 达到 A（未读 3 块", txt)
        self.assertNotIn("/8", txt)
        self.assertEqual(V["summary_RC"]["confirm_A"], 1)

    def test_RA_with_G(self):
        # full run: F = 0 -> G; A still holds; P1 word 增强 (C4 above C0) as in the original G test
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50,
                                                          "C6": 52, "C7": 60}}})
        B, _, txt = self.block(fx)
        self.assertTrue(B["labels"]["G"])
        self.assertEqual(B["R"]["case"], "R-A")
        self.assertIn("[G] " + PM.R_TXT["G"], txt)
        self.assertIn("oft_spatial_c30: 达到预设复现标准（R-A；G）", txt)    # R-C reason carries G
        # unit: G and P1 word 提高 (the G-only word)
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_REC, "D3": PM.L_NOREC, "G": True, "P1_word": PM.W_G},
                           {"P1": r_(12, 2, 22), "P2": r_(20, 10, 30), "D3": r_(-8, -18, 2), "F": r_(5, -3, 13)})
        self.assertEqual(rec["case"], "R-A")
        self.assertIn("P1 用词（§1.3）：提高", t)
        self.assertIn(PM.R_TXT["G"], t)
        self.assertNotIn("部分恢复", t)

    def test_RA_groups_when_not_100(self):
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_REC, "D3": PM.L_NOREC},
                           {"P1": r_(30, 20, 40), "P2": r_(30, 20, 40), "D3": r_(0, -5, 5)}, n=97)
        self.assertIn("两组各 100 个初始化均达到预设合取标准", t)            # K5: verbatim
        self.assertIn("（分析用键全集：首轮 n=100，确认轮 n=97）", t)

    def test_RA_iff_A_row(self):
        for sr in ({}, {"C5": 75}, {"C4": 45, "C5": 45}, {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50,
                                                               "C6": 52, "C7": 60}):
            fx = self.make(blocks={"oft_spatial_c30": {"sr": sr}})
            B, _, _ = self.block(fx)
            self.assertEqual(B["R"]["case"] == "R-A", B["is_A"], sr)

    def test_RB_i_full_run(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C5": 75}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["R"]["case"], "R-B (i)")
        self.assertIn("[R-B (i)] " + PM.R_TXT["Bi"] + "；P2 ", txt)
        self.assertIn("（Δ2 = +5.0 [", txt)
        self.assertIn(B["R"]["category"], (PM.RC_MAG, PM.RC_UNC))

    def test_RB_ii_with_D3_harm(self):
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_UNC, "D3": PM.L_HARM},
                           {"P1": r_(15, 5, 25), "P2": r_(25, -1, 51), "D3": r_(-12, -22, -2)})
        self.assertEqual(rec["case"], "R-B (ii)")
        self.assertIn("[R-B (ii)] P2 不确定（Δ2 = +25.0 [-1.0, +51.0]）；Δ3 有害（-12.0 [-22.0, -2.0]）", t)
        self.assertIn("[D] " + PM.ROW["D"], t)
        self.assertNotIn("均带来帮助", t)
        self.assertEqual(rec["category"], PM.RC_UNC)
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_NOREC, "D3": PM.L_UNC_HOLM},
                           {"P1": r_(15, 5, 25), "P2": r_(2, -6, 9), "D3": r_(13, 2, 24)})
        self.assertEqual(rec["case"], "R-B (ii)")
        self.assertIn("Δ3 不确定（Holm 未过）（+13.0 [+2.0, +24.0]）", t)
        self.assertNotIn("[D]", t)
        self.assertEqual(rec["category"], PM.RC_MAG)

    def test_RB_iii_full_run(self):
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C4": 45, "C5": 45}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["R"]["case"], "R-B (iii)")
        self.assertIn("[R-B (iii)] ep 30-39 上当前轮廓叠加未达到预设的 10 点提高幅度（Δ1 = +0.0 [", txt)
        self.assertIn("[C] " + PM.ROW["C"].format(d4="47.0"), txt)
        self.assertEqual(B["R"]["category"], PM.RC_MAG)

    def test_RB_iii_with_D4_cell_failed(self):
        # blind review r2 must-fix 1: C1 fails the gate (D4 not judged) while P1 is 未提高 -> no C sentence, no crash
        fx = self.make(blocks={"oft_spatial_c30": {"sr": {"C4": 45, "C5": 45}}},
                       gate_cells={"oft_spatial_c30": {"C1": {"status": "FAIL"}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["R"]["case"], "R-B (iii)")
        self.assertIsNone(B["labels"]["D4"])
        self.assertIn("C 句不判（Δ4 涉及的格门槛未过）", txt)
        self.assertNotIn("[C] ", txt.split("==== §3")[1].split("==== §5.4")[0])
        rec, t = self.rsec({"P1": PM.L_NOREC, "P2": PM.L_UNC, "D3": PM.L_UNC, "D4": None},
                           {"P1": r_(1, -6, 8), "P2": r_(4, -5, 13), "D3": r_(2, -7, 11), "D4": None})
        self.assertIn("C 句不判", t)

    def test_R0_cell_gate_failed_not_judged(self):
        fx = self.make(blocks={"ofti_object_c30": {"sr": {"C2": 0, "C3": 0, "C6": 0}}},
                       gate_cells={"ofti_object_c30": {"C6": {"status": "FAIL"}}})
        _, _, txt = self.block(fx, blocks="ofti_object_c30")
        self.assertIn("C6 门槛未过，照印不判", txt)
        self.assertIn("确认样本中再次观察到零成功（C2 / C3）", txt)

    def test_RB_iv(self):
        rec, t = self.rsec({"P1": PM.L_UNC, "P2": PM.L_UNC, "D3": PM.L_UNC},
                           {"P1": r_(6, -3, 15), "P2": r_(4, -5, 13), "D3": r_(2, -7, 11)})
        self.assertEqual(rec["case"], "R-B (iv)")
        self.assertIn("[R-B (iv)] ep 30-39 上证据不足以判定当前轮廓叠加是否达到预设标准（Δ1 = +6.0 [-3.0, +15.0]）", t)
        self.assertNotIn("未带来", t)
        self.assertNotIn("未达到", t)
        self.assertEqual(rec["category"], PM.RC_UNC)
        # 不确定 with hi95 < 10 cannot come from label_rec, but the branch still routes it to (iii)
        rec, t = self.rsec({"P1": PM.L_UNC, "P2": PM.L_UNC, "D3": PM.L_UNC},
                           {"P1": r_(2, -3, 8), "P2": r_(4, -5, 13), "D3": r_(2, -7, 11)})
        self.assertEqual(rec["case"], "R-B (iii)")

    def test_RB_G_added(self):
        rec, t = self.rsec({"P1": PM.L_NOREC, "P2": PM.L_UNC, "D3": PM.L_UNC, "G": True, "P1_word": None},
                           {"P1": r_(1, -6, 8), "P2": r_(4, -5, 13), "D3": r_(2, -7, 11), "F": r_(3, -4, 10)})
        self.assertEqual(rec["case"], "R-B (iii)")
        self.assertIn("[G] " + PM.R_TXT["G"], t)

    def test_RH_and_downgraded(self):
        rec, t = self.rsec({"P1": None, "P2": None, "D3": None}, {"P1": r_(30, 20, 40), "P2": r_(30, 20, 40),
                           "D3": r_(0, -5, 5)}, status="H", block_h=["block_ok=False"])
        self.assertEqual((rec["case"], rec["category"]), ("R-H", PM.RC_GATE))
        self.assertIn(PM.R_TXT["H"], t)
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_REC, "D3": PM.L_NOREC},
                           {"P1": r_(30, 20, 40), "P2": r_(30, 20, 40), "D3": r_(0, -5, 5)}, status="不确定")
        self.assertEqual((rec["case"], rec["category"]), ("downgraded", PM.RC_UNC))
        self.assertNotIn("[R-A]", t)
        # full run: cell gate failure on C4 -> R-H
        fx = self.make(gate_cells={"oft_spatial_c30": {"C4": {"status": "FAIL"}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["R"]["case"], "R-H")

    def test_oft_object_P2_borderline_note(self):
        rec, t = self.rsec({"P1": PM.L_REC, "P2": PM.L_UNC, "D3": PM.L_REC},
                           {"P1": r_(40, 30, 50), "P2": r_(9, 2, 16), "D3": r_(31, 20, 42)}, b="oft_object_c30")
        self.assertIn("[oft_object 专条] 首轮 Δ2 = +11.0 [+5.0, +17.0]", t)
        self.assertNotIn("消失", t)

    def test_R0_zero_and_some(self):
        sr0 = {"C2": 0, "C3": 0, "C6": 0}
        fx = self.make(blocks={"ofti_object_c30": {"sr": sr0}})
        B, _, txt = self.block(fx, blocks="ofti_object_c30")
        self.assertIn("[R-0] ep 20-29 上 box / alpha0 / box 保护物体为 0 / 0 / 0（每格 n = 100）；ep 30-39 上为 0 / 0 / 0\n", txt)
        self.assertIn("[R-0] 确认样本中再次观察到零成功（C2 / C3 / C6）", txt)
        self.assertNotIn("真实成功概率为零", txt)
        fx2 = self.make(blocks={"ofti_object_c30": {"sr": {"C2": 3, "C3": 0, "C6": 2}}})
        B2, _, txt2 = self.block(fx2, blocks="ofti_object_c30")
        self.assertIn("ep 30-39 上为 3 / 0 / 2", txt2)
        self.assertIn("确认样本中再次观察到零成功（C3）", txt2)
        self.assertEqual(B2["R"]["R0"]["confirm"], {"C2": 3, "C3": 0, "C6": 2})
        # R-0 only for ofti_object
        fx3 = self.make(blocks={"oft_object_c30": {}})
        _, _, txt3 = self.block(fx3, blocks="oft_object_c30")
        self.assertNotIn("[R-0]", txt3)

    def test_object_F_row_sentence(self):
        fx = self.make(blocks={"oft_object_c30": {"sr": {"C7": 80}}})
        B, _, txt = self.block(fx, blocks="oft_object_c30")
        self.assertIn("F", B["row_letters"])
        self.assertIn("[F] " + PM.ROW["F"].format(d6="15.0") + PM.F_OBJ, txt)
        self.assertIn("该干预在既有缓存校验中额外遮住腕部视角 5.2%、第三视角 0.0% 的可交互物体像素；这些不是本确认集的轨迹统计。", txt)
        self.assertNotIn("须核", txt)

    def test_RC_four_blocks(self):
        fx = self.make(blocks={"oft_spatial_c30": {}, "ofti_spatial_c30": {"sr": {"C4": 45, "C5": 45}},
                               "oft_object_c30": {}, "ofti_object_c30": {}},
                       gate_block={"ofti_object_c30": {"block_ok": False, "block_fail_reasons": ["label fields"]}})
        V, txt = self.ok_run(fx, "oft_spatial_c30,ofti_spatial_c30,oft_object_c30,ofti_object_c30")
        RC = V["summary_RC"]
        self.assertEqual(RC["confirm_A"], 2)
        self.assertEqual(RC["categories"]["ofti_spatial_c30"], PM.RC_MAG)
        self.assertEqual(RC["categories"]["ofti_object_c30"], PM.RC_GATE)
        self.assertIn("[R-C] 首轮 4/4 达到 A（oft_spatial、ofti_spatial、oft_object、ofti_object）；确认轮 2/4 达到 A", txt)
        self.assertIn("确认轮 2 个块达到预设复现标准，其余分别为未达到幅度标准、不确定或门槛不合格（达到：oft_spatial_c30、"
                      "oft_object_c30）；未达到幅度标准：ofti_spatial_c30；不确定：无；门槛不合格：ofti_object_c30。原结果：首轮 4/4 达到 A。", txt)
        self.assertNotIn("/8", txt)
        self.assertNotIn("未读 ", txt.split("==== §3 R-C")[1])

    def test_header_line(self):
        fx = self.make()
        _, _, txt = self.block(fx)
        self.assertIn("确认集 ep 30-39，前瞻性复现", txt.splitlines()[1])
        self.assertIn(f"首轮常量 sha256 {PM.FIRST_ROUND_SHA}", txt)


class TestConfirmGate(Base):
    """gate_maskinfo_confirm.py scan_block: strict labels (§4 item 10) and key-set check (§4 item 9)."""

    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location("gate_c", GATE)
        cls.G = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.G)

    def write(self, root, block, c, t, e, init=None, **over):
        d = os.path.join(root, block, c)
        os.makedirs(d, exist_ok=True)
        init = self.G.expected_init(t, e) if init is None else init
        names = [f"robot0_joint{i}" for i in range(1, 8)] + ["obj_joint0"]
        f = dict(success=np.array(True), task=np.array(t), ep=np.array(e), init=np.array(init),
                 gpu=np.array("NVIDIA H200 NVL|143771 MiB"), align_json=np.array(json.dumps({"copy_grip_cmd": 1})),
                 mr_checks=np.array(json.dumps([])), mi_rows=np.array(json.dumps([])), joint_names=np.array(names),
                 qpos=np.zeros((3, 8)), ref_body_pos=np.zeros((2, 3)), mi_cell=np.array(c), mr_cell=np.array(c),
                 mi_block=np.array(block))
        for k, v in over.items():
            if v is None:
                f.pop(k)
            else:
                f[k] = np.array(v)
        np.savez(os.path.join(d, f"t{t}_e{e}_i{init}.npz"), **f)

    def full(self, root, block):
        for c in CELLS:
            for t in range(10):
                for e in range(30, 40):
                    self.write(root, block, c, t, e)

    def det(self):
        return types.SimpleNamespace(detect=lambda *a, **k: (False, False, None))

    def test_clean_block_pass(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.full(d, "oft_spatial_c30")
        r, _ = self.G.scan_block(d, "oft_spatial_c30", None, self.det())
        self.assertEqual(r["label_fail"], [])
        self.assertTrue(all(x["status"] == "PASS" and x["n"] == 100 for x in r["cells"].values()))

    def test_missing_mi_cell_fails_block_label(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.full(d, "oft_spatial_c30")
        self.write(d, "oft_spatial_c30", "C2", 3, 33, mi_cell=None)
        self.write(d, "oft_spatial_c30", "C5", 4, 34, mr_cell="C4")
        r, _ = self.G.scan_block(d, "oft_spatial_c30", None, self.det())
        self.assertEqual(len(r["label_fail"]), 2)
        self.assertIn("missing", r["label_fail"][0])

    def test_keyset_fails_cell(self):
        d = tempfile.mkdtemp()
        self.addCleanup(shutil.rmtree, d, True)
        self.full(d, "oft_object_c30")
        self.write(d, "oft_object_c30", "C1", 0, 20)                         # first-round key
        bad_init = (self.G.expected_init(5, 36) + 1) % 50
        os.remove(os.path.join(d, "oft_object_c30", "C6", f"t5_e36_i{self.G.expected_init(5, 36)}.npz"))
        self.write(d, "oft_object_c30", "C6", 5, 36, init=bad_init)           # init not permutation(50)[ep]
        r, _ = self.G.scan_block(d, "oft_object_c30", None, self.det())
        self.assertEqual(r["cells"]["C1"]["status"], "FAIL")
        self.assertEqual(r["cells"]["C6"]["status"], "FAIL")
        self.assertEqual(r["cells"]["C0"]["status"], "PASS")
        self.assertIn("outside", r["keyset_fail"]["C1"][0])
        self.assertIn("permutation", r["keyset_fail"]["C6"][0])

    def test_expected_init_matches_eval_rule(self):
        for t in range(10):
            perm = np.random.RandomState(5478 + t).permutation(50)
            for e in range(30, 40):
                self.assertEqual(self.G.expected_init(t, e), int(perm[e]))
        self.assertEqual(self.G.BLOCKS, PM.BLOCKS)
        self.assertEqual(self.G.EPS, list(range(30, 40)))


if __name__ == "__main__":
    unittest.main()
