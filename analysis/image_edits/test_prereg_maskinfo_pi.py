"""Synthetic-data tests for prereg_maskinfo_pi.py (protocol_maskinfo_pi.md v2 on top of protocol_maskinfo.md v3: §1.4
rows A-J (A overridden), K / K' / L / M, G trigger, 部分恢复 / 增强 wording, C8 outside Holm, nine-cell completeness, gates
(h200 only), sensitivity, guards, §5.4 events, --check-fields-only). Adapted from the frozen test_prereg_maskinfo.py.
Run: python3 -m unittest test_prereg_maskinfo_pi -v   (from this directory)
The protocol file is read from ref/protocol_maskinfo_pi.md (override with MI_PROTO); its sha is computed, not hard-coded.

Field set of the synthetic npz (deviation 10 (b) lesson): CLIENT_KEYS is parsed (AST, static) from the real client
sources -- _mi_dump in ../pi/zoo_client_mi.py and every dump_fields in the hook chain ../embody/maskinfo_pi.py,
maskinfo.py, maskrepl.py (override dir with MI_PI_ROOT). REQUIRED_FIELDS must be a subset of it, and the fixture refuses
to write any field name that is not in it. No field list is re-typed here.
"""
import ast
import hashlib
import importlib.util
import io
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
import zipfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
SCRIPT = os.path.join(HERE, "prereg_maskinfo_pi.py")
PROTO = os.environ.get("MI_PROTO", os.path.join(HERE, "ref", "protocol_maskinfo_pi.md"))
MASKREPL = os.path.join(HERE, "ref", "prereg_maskrepl.py")
PI_ROOT = os.environ.get("MI_PI_ROOT", os.path.dirname(HERE))
CLIENT_SRC = os.path.join(PI_ROOT, "pi", "zoo_client_mi.py")
HOOK_SRCS = [os.path.join(PI_ROOT, "embody", x) for x in ("maskinfo_pi.py", "maskinfo.py", "maskrepl.py")]
with open(PROTO, "rb") as _f:
    PSHA = hashlib.sha256(_f.read()).hexdigest()
TMPROOT = os.path.join(HERE, "_testtmp")

sys.path.insert(0, HERE)
import prereg_maskinfo_pi as PM  # noqa: E402


# ===================================================================================== client key names (static, AST)
def _str_keys_of_dict(node):
    return {k.value for k in node.keys if isinstance(k, ast.Constant) and isinstance(k.value, str)}


def client_keys():
    """Field names the v2 client writes: keywords of np.savez* / dict(...) calls and str keys of dict literals inside
    _mi_dump; in every dump_fields: dict-literal keys, out.update(...) keywords, out["k"] / out[f"..{v}"] assignments
    (f-strings expanded over the enclosing `for v in (...)` tuple)."""
    keys = set()
    with open(CLIENT_SRC) as f:
        tree = ast.parse(f.read())
    fn = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_mi_dump"]
    assert len(fn) == 1, "client has no unique _mi_dump"
    for m in ast.walk(fn[0]):
        if isinstance(m, ast.Call):
            f_ = m.func
            name = f_.attr if isinstance(f_, ast.Attribute) else f_.id if isinstance(f_, ast.Name) else None
            if name in ("savez_compressed", "savez", "dict"):
                keys |= {k.arg for k in m.keywords if k.arg}
        elif isinstance(m, ast.Dict):
            keys |= _str_keys_of_dict(m)
    n_df = 0
    for src in HOOK_SRCS:
        with open(src) as f:
            tree = ast.parse(f.read())
        for df in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "dump_fields"]:
            n_df += 1
            loopvals = {}
            for m in ast.walk(df):
                if (isinstance(m, ast.For) and isinstance(m.target, ast.Name) and isinstance(m.iter, (ast.Tuple, ast.List))
                        and all(isinstance(e, ast.Constant) for e in m.iter.elts)):
                    loopvals[m.target.id] = [e.value for e in m.iter.elts]
            for m in ast.walk(df):
                if isinstance(m, ast.Dict):
                    keys |= _str_keys_of_dict(m)
                elif (isinstance(m, ast.Call) and isinstance(m.func, ast.Attribute) and m.func.attr == "update"):
                    keys |= {k.arg for k in m.keywords if k.arg}
                elif isinstance(m, ast.Subscript) and isinstance(m.ctx, ast.Store):
                    sl = m.slice
                    if isinstance(sl, ast.Constant) and isinstance(sl.value, str):
                        keys.add(sl.value)
                    elif isinstance(sl, ast.JoinedStr):
                        parts, names = [], []
                        for v in sl.values:
                            if isinstance(v, ast.Constant):
                                parts.append([v.value])
                            elif isinstance(v.value, ast.Name) and v.value.id in loopvals:
                                parts.append([str(x) for x in loopvals[v.value.id]])
                            else:
                                parts = None
                                break
                        if parts:
                            keys |= {"".join(p) for p in itertools.product(*parts)}
    assert n_df >= 3, f"expected dump_fields in the three hook files, found {n_df}"
    return keys


CLIENT_KEYS = client_keys()

KEYS = PM.DESIGN_KEYS
CELLS = PM.CELLS
MARK = PM.SYN_MARKER
H200 = "NVIDIA H200 NVL|143771 MiB"
FT_CYCLE = ["no_reach", "no_grasp", "drop", "other"]
DEFAULT_SR = {"C0": 95, "C1": 92, "C2": 75, "C3": 45, "C4": 80, "C5": 50, "C6": 78, "C7": 93, "C8": 70}   # -> rows A, M
DEFAULT_TIMING = (20, 22, 30)   # t_near, t_close, t_lift
BLK = "pi05_spatial"


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


def nested(n, keys=None):
    """Success on the first n keys of one fixed permutation -> success sets of different cells are nested."""
    s = set(ORDER[:n])
    return {k: (k in s) for k in (keys or KEYS)}


def from_ranges(*ranges):
    s = set()
    for a, b in ranges:
        s |= set(ORDER[a:b])
    return {k: (k in s) for k in KEYS}


def ftype_of(k):
    return FT_CYCLE[(k[0] * 10 + k[1]) % 4]


PADS = json.dumps(["gripper0_finger1_pad_collision", "gripper0_finger2_pad_collision"])


def episode(succ, k, timing):
    """Synthetic per-step trajectory (rows = states 0..T); verbatim from the frozen test."""
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


def query_fields(T):
    """Per-query fields of the v2 client (values arbitrary: the read-out only checks presence / dtype / shape)."""
    nq = max(1, (T - 10) // 5 + 1)
    return dict(noise_seed=np.arange(nq, dtype=np.int64) + 123, mi_lag_step_actual=np.full(nq, -1, np.int32),
                mi_lag_step_target=np.full(nq, -1, np.int32))


class Fixture:
    def __init__(self, blocks=None, gate_block=None, gate_cells=None, phys=None, drop=None, gpu=None, gate_sha=None,
                 marker=True, omit_field=None, extra_file=None, keys=None, mutate=None):
        """blocks: {block: {"sr": {...}, "succ": {cell: map}, "timing": {cell: (r, c, l)}}}
        mutate: (block, cell, key, field, value) -> overwrite one field of one file."""
        os.makedirs(TMPROOT, exist_ok=True)
        self.base = tempfile.mkdtemp(dir=TMPROOT)
        self.runs = os.path.join(self.base, "runs")
        self.out = os.path.join(self.base, "results_maskinfo_pi.txt")
        self.vj = os.path.join(self.base, "verdict_maskinfo_pi.json")
        self.gate = os.path.join(self.base, "gate_maskinfo_pi.json")
        blocks = blocks if blocks is not None else {BLK: {}}
        keys = keys or KEYS
        drop, gpu = drop or {}, gpu or {}
        self.succ, self.timing = {}, {}
        g = {"protocol_sha256": gate_sha or PSHA, "blocks": {}}
        for b, spec in blocks.items():
            sr = dict(DEFAULT_SR, **(spec.get("sr") or {}))
            self.succ[b], self.timing[b] = {}, {}
            for c in CELLS:
                s = (spec.get("succ") or {}).get(c) or nested(sr[c], keys)
                tm = (spec.get("timing") or {}).get(c, DEFAULT_TIMING)
                self.succ[b][c], self.timing[b][c] = s, tm
                d = os.path.join(self.runs, b, c)
                os.makedirs(d, exist_ok=True)
                for k in keys:
                    if k in (drop.get(b, {}).get(c) or ()):
                        continue
                    t, e = k
                    ep = episode(s[k], k, tm)
                    fields = dict(success=np.array(bool(s[k])), task=np.array(t), ep=np.array(e), init=np.array(e),
                                  gpu=np.array(gpu.get(b, {}).get(c, H200)), host=np.array("node41"),
                                  mi_cell=np.array(c), mr_cell=np.array(c), mi_block=np.array(b),
                                  **ep, **query_fields(len(ep["action"])))
                    if omit_field and (b, c, k) == omit_field[:3]:
                        fields.pop(omit_field[3])
                    if mutate and (b, c, k) == mutate[:3]:
                        fields[mutate[3]] = mutate[4]
                    alien = sorted(set(fields) - CLIENT_KEYS)
                    assert not alien, f"fixture field not written by the client: {alien}"
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
            shutil.copy(os.path.join(self.runs, b, c, "t0_e20_i20.npz"), os.path.join(self.runs, b, c, name))
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

    def run(self, blocks=BLK, extra=(), synthetic=True, sha=None):
        return subprocess.run([sys.executable, "-I", SCRIPT] + self.args(blocks, extra, synthetic, sha),
                              capture_output=True, text=True)

    def check_fields(self, blocks=BLK, extra=()):
        return subprocess.run([sys.executable, "-I", SCRIPT, "--check-fields-only", "--runs", self.runs,
                               "--blocks", blocks] + list(extra), capture_output=True, text=True)

    def lock(self, b):
        return os.path.join(self.base, f"prereg_maskinfo_{b}.lock")

    def cleanup(self):
        shutil.rmtree(self.base, ignore_errors=True)


class Base(unittest.TestCase):
    def make(self, **kw):
        fx = Fixture(**kw)
        self.addCleanup(fx.cleanup)
        return fx

    def ok_run(self, fx, blocks=BLK, extra=()):
        p = fx.run(blocks, extra)
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        V = jload(fx.vj)
        txt = rtext(fx.out)
        self.assertEqual(PM.wording_violations(txt), [])
        self.assertIn("合成数据测试运行", txt.splitlines()[0])
        return V, txt

    def block(self, fx, sr=None, blocks=BLK, **kw):
        name = blocks.split(",")[0]
        V, txt = self.ok_run(fx, blocks, **kw)
        return V["blocks"][name], V, txt

    def refused(self, fx, p, code, blocks=(BLK,)):
        self.assertEqual(p.returncode, code, p.stderr + p.stdout)
        self.assertFalse(os.path.exists(fx.out))
        self.assertFalse(os.path.exists(fx.vj))
        for b in blocks:
            self.assertFalse(os.path.exists(fx.lock(b)))


# ===================================================================================== fixture field set vs client
class TestFieldSource(unittest.TestCase):
    def test_required_fields_subset_of_client_keys(self):
        lack = sorted(set(PM.REQUIRED_FIELDS) - CLIENT_KEYS)
        self.assertEqual(lack, [], f"REQUIRED_FIELDS not written by the client: {lack}")

    def test_client_keys_parsed_from_both_sources(self):
        # sanity of the parser itself: one key from _mi_dump, one from each hook level
        for k in ("noise_seed", "mi_block", "success"):           # _mi_dump (savez keyword / mi_extra dict)
            self.assertIn(k, CLIENT_KEYS)
        for k in ("mi_lag_step_actual", "mi_cell", "mi_mask_md5_agent", "mi_mask_md5_wrist"):   # maskinfo_pi
            self.assertIn(k, CLIENT_KEYS)
        self.assertIn("mi_spec", CLIENT_KEYS)                                 # maskinfo
        for k in ("mr_cell", "mr_img_agent", "mr_model_md5_wrist"):           # maskrepl
            self.assertIn(k, CLIENT_KEYS)

    def test_v2_field_checks_present(self):
        for k in ("noise_seed", "mi_lag_step_actual", "mi_lag_step_target", "mi_cell", "mr_cell", "mi_block"):
            self.assertIn(k, PM.REQUIRED_FIELDS)


# ===================================================================================== §1.4 rows A-J, G, wording rule
class TestRows(Base):
    def test_row_A_partial_recovery(self):
        fx = self.make()
        B, V, txt = self.block(fx)
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["A", "M"])
        self.assertTrue(B["is_A"])
        self.assertEqual((B["labels"]["P1"], B["labels"]["P2"]), (PM.L_REC, PM.L_REC))
        self.assertFalse(B["labels"]["G"])
        self.assertEqual(B["labels"]["P1_word"], PM.W_PART)
        self.assertAlmostEqual(B["estimates"]["P1"]["point"], 35.0)
        self.assertAlmostEqual(B["estimates"]["P2"]["point"], 30.0)
        self.assertIn(PM.ROW["A"].format(clause="带来 35.0 点的部分性能恢复", lag=15, d2="30.0"), txt)
        self.assertEqual(B["labels"]["D6"], PM.L_HARMLESS)
        for s in PM.LIMITS:
            self.assertIn(s, txt)
        self.assertIn("PASS（无禁用词）", txt)
        self.assertNotIn("[G]", txt)

    def test_row_A_override_clause(self):
        # maskinfo-π §1.4: the frozen clause is replaced, the rest of row A is verbatim
        self.assertNotIn("因此精确分割遮罩保留了", PM.ROW["A"])
        self.assertIn("因此在 alpha0 擦除图上添加的当前轮廓携带与控制相关的、当前遮挡条件下的身体相关二维线索；"
                      "seg-black 图里的同一边界是否在原图或 seg-black 条件下被该策略使用，本实验不回答", PM.ROW["A"])
        self.assertTrue(PM.ROW["A"].endswith("『精确遮罩后不掉分』不能单独作为『不依赖身体视觉』的证据。"))
        self.assertTrue(PM.ROW["A"].startswith("在该 checkpoint 与套件上，其他输入保留时"))

    def test_row_A_enhancement(self):
        fx = self.make(blocks={BLK: {"sr": {"C0": 80, "C3": 40, "C4": 95, "C5": 50}}})
        B, _, txt = self.block(fx)
        self.assertGreater(B["estimates"]["EN"]["ci95"][0], 0)
        self.assertEqual(B["labels"]["P1_word"], PM.W_ENH)
        self.assertIn(PM.ROW["A"].format(clause="带来 55.0 点的增强", lag=15, d2="45.0"), txt)

    def test_row_A_lag_20_and_choices(self):
        fx = self.make()
        B, V, txt = self.block(fx, extra=["--lag-steps", "20"])
        self.assertIn("且优于滞后 20 步的轮廓（30.0 点）", txt)
        self.assertIn("π 家族 20 步，约 1 秒", txt)
        for bad in ("16", "24"):
            fx2 = self.make()
            p = fx2.run(extra=["--lag-steps", bad])
            self.refused(fx2, p, 2)

    def test_row_G_does_not_block_and_enhancement(self):
        fx = self.make(blocks={BLK: {"sr": {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50,
                                            "C6": 52, "C7": 60, "C8": 55}}})
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
        c0 = from_ranges((0, 50), (60, 78))
        c3 = from_ranges((0, 60))
        c4 = from_ranges((0, 72))
        fx = self.make(blocks={BLK: {"succ": {"C0": c0, "C3": c3, "C4": c4}, "sr": {"C5": 30}}})
        B, _, txt = self.block(fx)
        self.assertAlmostEqual(B["estimates"]["F"]["point"], 8.0)
        self.assertTrue(B["labels"]["G"])
        self.assertEqual(B["labels"]["P1"], PM.L_REC)
        self.assertEqual(B["labels"]["P1_word"], PM.W_G)
        self.assertIn("G", B["row_letters"])
        self.assertIn("A", B["row_letters"])
        self.assertIn(PM.ROW["A"].format(clause="后，C4 高于 C3 12.0 点", lag=15, d2="42.0"), txt)
        self.assertNotIn("部分性能恢复", txt)
        self.assertNotIn("点的增强", txt)

    def test_G_not_triggered_when_lo95_positive(self):
        fx = self.make(blocks={BLK: {"sr": {"C3": 85}}})
        B, _, _ = self.block(fx)
        self.assertGreater(B["estimates"]["F"]["ci95"][0], 0)
        self.assertFalse(B["labels"]["G"])
        self.assertNotIn("G", B["row_letters"])

    def test_row_B(self):
        fx = self.make(blocks={BLK: {"sr": {"C5": 75}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["P1"], PM.L_REC)
        self.assertIn(B["labels"]["P2"], (PM.L_NOREC, PM.L_UNC))
        self.assertEqual(B["labels"]["D3"], PM.L_REC)
        self.assertEqual(B["row_letters"], ["B", "M"])
        self.assertIn(PM.ROW["B"], txt)
        self.assertFalse(B["is_A"])

    def test_row_C(self):
        fx = self.make(blocks={BLK: {"sr": {"C4": 45, "C5": 45}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["P1"], PM.L_NOREC)
        self.assertEqual(B["row_letters"], ["C", "M"])
        self.assertIn(PM.ROW["C"].format(d4="47.0"), txt)

    def test_row_D(self):
        fx = self.make(blocks={BLK: {"sr": {"C5": 30}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["D3"], PM.L_HARM)
        self.assertEqual(B["row_letters"], ["A", "D", "M"])

    def test_rows_E_and_F(self):
        fx = self.make(blocks={BLK: {"sr": {"C6": 90, "C7": 80}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["labels"]["D6"], PM.L_HARM)
        self.assertEqual(B["row_letters"], ["A", "E", "F", "M"])
        self.assertIn("[E] " + PM.ROW["E"].format(d5="15.0"), txt)
        self.assertIn(PM.ROW["F"].format(d6="15.0"), txt)

    def test_row_F_object_occlusion_note_only_on_object(self):  # protocol v2.4 2.6 item 2 (deviation 1)
        sr = {"C6": 90, "C7": 80}
        fx = self.make(blocks={"pi05_spatial": {"sr": sr}, "pi05_object": {"sr": sr}})
        V, txt = self.ok_run(fx, "pi05_spatial,pi05_object")
        plain = PM.ROW["F"].format(d6="15.0")
        noted = plain.replace("异位遮挡也损伤", "异位遮挡" + PM.F_OCC_OBJECT + "也损伤")
        self.assertEqual(PM.F_OCC_OBJECT, "（另遮住腕部视角 5.1%、第三视角 0% 可交互物体像素）")
        self.assertNotEqual(plain, noted)
        self.assertEqual(txt.count(noted), 1)   # Object block
        self.assertEqual(txt.count(plain), 1)   # Spatial block
        for b in ("pi05_spatial", "pi05_object"):
            self.assertEqual(V["blocks"][b]["labels"]["D6"], PM.L_HARM)   # the label itself is untouched

    def test_row_J_triggered_and_not(self):
        fx = self.make(blocks={BLK: {"sr": {"C1": 70}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["J"]["differ"], ["C1"])
        self.assertIn("J", B["row_letters"])
        fx2 = self.make()
        B2, _, txt2 = self.block(fx2)
        self.assertEqual(B2["J"]["labels"], PM.MASKREPL_J)
        self.assertIn("[J] 不触发", txt2)

    def test_residual_combination_no_row(self):
        fx = self.make(blocks={BLK: {"sr": {"C4": 60, "C5": 52}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["row_letters"], ["M"])
        self.assertIn("不属于 A / B / C 行", txt)

    def test_row_I_same_model_object_vs_spatial(self):
        fx = self.make(blocks={"pi05_spatial": {}, "pi05_object": {}, "pi0_spatial": {},
                               "pi0_object": {"sr": {"C4": 45, "C5": 45}}})
        V, txt = self.ok_run(fx, "pi05_spatial,pi05_object,pi0_spatial,pi0_object")
        I = V["summary_I"]
        self.assertEqual(I["pi05_object"], PM.ROW["I_obj_ok"])
        self.assertTrue(I["pi0_object"].startswith(PM.ROW["I_obj_ii"]))
        self.assertNotIn("spatial", {k for k in I if k not in ("blocks", "K", "L")})   # no cross-model Spatial I row
        self.assertNotIn(PM.ROW["I_both"], txt)

    def test_row_I_object_case_i_and_unread_spatial(self):
        fx = self.make(blocks={"pi0_object": {"sr": {"C3": 95, "C4": 95, "C5": 95}}})
        V, _ = self.ok_run(fx, "pi0_object")
        self.assertTrue(V["summary_I"]["pi0_object"].startswith(PM.ROW["I_obj_i"]))
        self.assertIn("pi0_spatial：未读", V["summary_I"]["pi0_object"])
        self.assertEqual(V["summary_I"]["pi05_object"], "未读")


# ===================================================================================== K / K' / L rows
class TestKL(Base):
    def K(self, V):
        return "\n".join(V["summary_I"]["K"]["text"])

    def test_K_pi05_spatial_A(self):
        fx = self.make()
        V, txt = self.ok_run(fx)
        self.assertEqual(V["summary_I"]["K"]["row"], "K")
        k = self.K(V)
        self.assertIn(PM.KROW["K"].format(systems=PM.KROW["sys2"], d1s="π0.5 × Spatial Δ1 = 35.0 点，块内解释：部分恢复",
                                          lag=15, sec="约 0.8 秒"), k)
        self.assertIn(PM.KROW["K_note"], k)
        self.assertIn("合训 OFT × Spatial", k)
        self.assertNotIn("三个独立架构", txt)

    def test_K_with_pi0_A_lists_each_model(self):
        fx = self.make(blocks={"pi05_spatial": {}, "pi0_spatial": {}})
        V, txt = self.ok_run(fx, "pi05_spatial,pi0_spatial")
        k = self.K(V)
        self.assertIn(f"所测的策略系统（{PM.KROW['sys_list']}）", k)
        self.assertIn("π0 × Spatial Δ1 = 35.0 点", k)
        self.assertIn("π0.5 × Spatial、π0 × Spatial", k)

    def test_K_not_blocked_by_F(self):
        fx = self.make(blocks={BLK: {"sr": {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50,
                                            "C6": 52, "C7": 60, "C8": 55}}})
        V, _ = self.ok_run(fx)
        self.assertTrue(V["blocks"][BLK]["labels"]["G"])
        self.assertEqual(V["summary_I"]["K"]["row"], "K")
        self.assertIn("块内解释：增强", self.K(V))

    def test_Kprime_P1_reached_P2_not_needs_D3(self):
        fx = self.make(blocks={BLK: {"sr": {"C5": 75}}})          # P1 提高, P2 not, Δ3 提高
        V, txt = self.ok_run(fx)
        self.assertEqual(V["summary_I"]["K"]["row"], "K′")
        k = self.K(V)
        self.assertIn(PM.KROW["P1_" + PM.T_REC].format(d="35.0"), k)
        self.assertIn(PM.KROW["both"], k)
        self.assertTrue(PM.KROW["P2_" + PM.T_UNC].split("（")[0] in k or "未优于滞后轮廓 ≥ 10 点" in k)
        self.assertIn("C4 是否不及 C1 只看 Δ4", k)
        self.assertIn(PM.KROW["others"].format(lst="、".join(PM.OFT_A_BLOCKS)), k)

    def test_Kprime_both_not_written_without_D3(self):
        # P1 提高 (C4 60 - C3 45 = 15), P2 8 not reached, Δ3 = 7 not 提高 -> no "均带来帮助"
        fx = self.make(blocks={BLK: {"sr": {"C4": 60, "C5": 52}}})
        V, _ = self.ok_run(fx)
        k = self.K(V)
        self.assertIn(PM.KROW["P1_" + PM.T_REC].format(d="15.0"), k)
        self.assertNotIn(PM.KROW["both"], k)

    def test_Kprime_P1_excluded(self):
        fx = self.make(blocks={BLK: {"sr": {"C4": 45, "C5": 45}}})
        V, txt = self.ok_run(fx)
        k = self.K(V)
        self.assertIn("当前轮廓叠加（保护对象后剩余的部分）未带来 ≥ 10 点的提高（Δ1 = 0.0 [0.0, 0.0]）", k)
        self.assertIn("未优于滞后轮廓 ≥ 10 点（Δ2 = 0.0 [0.0, 0.0]）", k)
        self.assertIn("Δ4 = 47.0", k)
        self.assertIn("标签 提高", k)
        self.assertNotIn("跨架构未复现", txt)

    def test_Kprime_P1_uncertain_never_says_no_improvement(self):
        fx = self.make(blocks={BLK: {"sr": {"C4": 52}}})          # Δ1 = 7, hi95 >= 10 -> 不确定
        V, txt = self.ok_run(fx)
        B = V["blocks"][BLK]
        self.assertEqual(B["labels"]["P1"], PM.L_UNC)
        k = self.K(V)
        self.assertIn("P1 不确定（Δ1 = 7.0 [", k)
        self.assertNotIn("未带来提高", txt)
        self.assertNotIn("未带来 ≥ 10 点", k)
        self.assertNotIn(PM.KROW["both"], k)

    def test_Kprime_other_blocks_by_model_and_suite(self):
        fx = self.make(blocks={"pi05_spatial": {"sr": {"C4": 45, "C5": 45}}, "pi0_spatial": {}})
        V, txt = self.ok_run(fx, "pi05_spatial,pi0_spatial")
        k = self.K(V)
        self.assertIn("、".join(PM.OFT_A_BLOCKS) + "、π0 × Spatial 上为 A", k)
        self.assertNotIn("OFT 特有", txt)

    def test_Kprime_G_note(self):
        fx = self.make(blocks={BLK: {"sr": {"C3": 95, "C4": 95, "C5": 95}}})
        V, _ = self.ok_run(fx)
        self.assertIn(PM.KROW["noF"].format(blk="π0.5 × Spatial"), self.K(V))

    def test_K_not_judged_when_pi05_spatial_unread_or_H(self):
        fx = self.make(blocks={"pi0_spatial": {}})
        V, _ = self.ok_run(fx, "pi0_spatial")
        self.assertIsNone(V["summary_I"]["K"]["row"])
        self.assertIn("pi05_spatial 未读：K、K′ 都不判", self.K(V))
        fx2 = self.make(gate_block={BLK: {"block_ok": False}})
        V2, _ = self.ok_run(fx2)
        self.assertIn("pi05_spatial 为 H：K、K′ 都不判", self.K(V2))

    def test_L_pi0_excluded_and_candidate_explanation(self):
        fx = self.make(blocks={"pi05_spatial": {}, "pi0_spatial": {"sr": {"C4": 45, "C5": 45}}})
        V, txt = self.ok_run(fx, "pi05_spatial,pi0_spatial")
        L = "\n".join(V["summary_I"]["L"]["pi0_spatial"])
        self.assertIn("在 π0 的该设置下当前轮廓未带来 ≥ 10 点的提高（Δ1 = 0.0 [0.0, 0.0]）", L)
        self.assertIn(PM.KROW["L_cand"].format(suite="Spatial"), L)
        self.assertIn(PM.KROW["L_limit"], L)
        self.assertNotIn("未检出轮廓收益", txt)
        self.assertNotIn("阴性不可读", txt)
        self.assertEqual(V["summary_I"]["L"]["pi0_object"], ["pi0_object 未读：L 不判"])

    def test_L_same_pattern_no_candidate_and_F_not_blocking(self):
        sr = {"C0": 60, "C3": 60, "C4": 85, "C5": 50, "C1": 60, "C2": 50, "C6": 52, "C7": 60, "C8": 55}
        fx = self.make(blocks={"pi05_spatial": {}, "pi0_spatial": {"sr": sr}})
        V, _ = self.ok_run(fx, "pi05_spatial,pi0_spatial")
        L = "\n".join(V["summary_I"]["L"]["pi0_spatial"])
        self.assertIn(PM.KROW["P1_" + PM.T_REC].format(d="25.0"), L)        # judged although lo95(F) <= 0
        self.assertIn(PM.KROW["noF"].format(blk="π0 × Spatial"), L)
        self.assertNotIn(PM.KROW["L_cand"].format(suite="Spatial"), L)

    def test_L_P2_not_reached_is_not_no_contour_gain(self):
        fx = self.make(blocks={"pi0_spatial": {"sr": {"C5": 75}}})   # not A only because P2 is not reached
        V, txt = self.ok_run(fx, "pi0_spatial")
        L = "\n".join(V["summary_I"]["L"]["pi0_spatial"])
        self.assertFalse(V["blocks"]["pi0_spatial"]["is_A"])
        self.assertIn(PM.KROW["P1_" + PM.T_REC].format(d="35.0"), L)
        self.assertNotIn("未检出", txt)
        self.assertNotIn("未带来", L)


# ===================================================================================== M row, C8 and Holm
class TestM(Base):
    def M(self, B):
        return [r["text"] for r in B["rows"] if r["row"] == "M"]

    def test_M_bg_worse_effect_reached_pi05(self):
        fx = self.make()                                           # Δ8 = 92 - 70 = 22
        B, _, txt = self.block(fx)
        self.assertEqual(B["M"], {"judged": True, "direction": "BG 更差", "vs_trace": "相同", "effect_standard": "达到"})
        m = self.M(B)[0]
        self.assertIn("π0.5 × Spatial 上 seg-black 对原图 -3.0 点、Trace 风格 BG 对原图 -25.0 点，两者差 Δ8 = 22.0 点 [", m)
        self.assertIn("黑色填充掉 1.0、BG 填充掉 31.8", m)
        self.assertIn("效应量 22.0 点达到预定的 10 点效应量标准", m)
        self.assertNotIn("一致", m)
        self.assertAlmostEqual(B["estimates"]["D8"]["point"], 22.0)
        self.assertAlmostEqual(B["D_c"]["C8"]["point"], -25.0)

    def test_M_small_positive_is_same_direction_not_inconsistent(self):
        fx = self.make(blocks={BLK: {"sr": {"C8": 87}}})          # Δ8 = +5, lo95 > 0
        B, _, _ = self.block(fx)
        self.assertGreater(B["estimates"]["D8"]["ci95"][0], 0)
        self.assertEqual((B["M"]["direction"], B["M"]["vs_trace"], B["M"]["effect_standard"]), ("BG 更差", "相同", "未达到"))
        self.assertNotIn("一致", self.M(B)[0])

    def test_M_black_worse_and_interval_with_zero(self):
        fx = self.make(blocks={"pi05_object": {"sr": {"C8": 100}}})
        B, _, _ = self.block(fx, blocks="pi05_object")
        self.assertEqual((B["M"]["direction"], B["M"]["vs_trace"]), ("黑色更差", "相反"))
        self.assertIn("黑色填充掉 −1.8、BG 填充掉 92.4", self.M(B)[0])
        fx2 = self.make(blocks={BLK: {"sr": {"C8": 92}}})
        B2, _, _ = self.block(fx2)
        self.assertEqual((B2["M"]["direction"], B2["M"]["vs_trace"], B2["M"]["effect_standard"]),
                         ("区间含 0", "区间含 0", "未达到"))

    def test_M_pi0_no_trace_anchor(self):
        fx = self.make(blocks={"pi0_object": {}})
        B, _, txt = self.block(fx, blocks="pi0_object")
        m = self.M(B)[0]
        self.assertIn("π0 × Object 上", m)
        self.assertIn(PM.KROW["M_pi0"], m)
        self.assertNotIn("VLA-Trace", m)
        self.assertNotIn("vs_trace", B["M"])
        self.assertNotIn("π 家族共有", txt)

    def test_M_not_judged_when_C8_gate_fails_and_P1_P2_unaffected(self):
        fx = self.make(gate_cells={BLK: {"C8": {"status": "FAIL"}}})
        B, _, txt = self.block(fx)
        self.assertFalse(B["M"]["judged"])
        self.assertEqual(B["row_letters"], ["A"])
        self.assertEqual((B["labels"]["P1"], B["labels"]["P2"]), (PM.L_REC, PM.L_REC))
        self.assertIn("[M] 不判", txt)

    def test_C8_not_in_holm_family(self):
        self.assertEqual(PM.SECONDARY, ["D3", "D4", "D5", "D6"])
        self.assertEqual(PM.HOLM_M, 4)
        self.assertNotIn("D8", PM.JUDGED)
        self.assertNotIn("D8", PM.SENS_KEYS)
        res = []
        for c8 in (70, 100, 0):
            fx = self.make(blocks={BLK: {"sr": {"C8": c8, "C6": 88}}})   # Δ5 close to the Holm line
            B, _, _ = self.block(fx)
            self.assertEqual(set(B["holm_p"]), set(PM.SECONDARY))
            self.assertEqual(set(B["mcnemar_p_for_holm"]), set(PM.SECONDARY))
            res.append((B["holm_p"], B["labels"]))
        self.assertTrue(all(r == res[0] for r in res))

    def test_C8_physics_keys_do_not_change_judged_labels(self):
        fx = self.make(phys={BLK: {"C8": ORDER[0:10]}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["sensitivity"]["changed"], [])
        self.assertEqual(B["status"], "judged")


# ===================================================================================== H rows and gates
class TestGates(Base):
    def test_cell_fail_marks_comparisons_H(self):
        fx = self.make(gate_cells={BLK: {"C4": {"status": "FAIL", "reasons": ["首帧不一致"]}}})
        B, V, txt = self.block(fx)
        self.assertEqual(B["status"], "judged")
        self.assertEqual(B["row_letters"], ["H", "M"])
        self.assertIsNone(B["labels"]["P1"])
        self.assertIsNone(B["labels"]["P2"])
        self.assertEqual(B["labels_by_line"]["P1"], PM.L_REC)
        self.assertIn("按线本应：提高", txt)
        self.assertIn("P1 不判（门槛未过，H）", "\n".join(V["summary_I"]["K"]["text"]))

    def test_cell_fail_C0_word_not_judged(self):
        fx = self.make(gate_cells={BLK: {"C0": {"status": "FAIL"}}})
        B, V, txt = self.block(fx)
        self.assertIsNone(B["labels"]["P1_word"])
        self.assertEqual(B["row_letters"], ["H", "A"])
        self.assertFalse(B["M"]["judged"])
        self.assertIn("C4 高于 C3 35.0 点（用词不判：C0 门槛未过）", txt)
        self.assertIn("块内解释：不判", "\n".join(V["summary_I"]["K"]["text"]))

    def test_block_ok_false(self):
        fx = self.make(gate_block={BLK: {"block_ok": False, "block_fail_reasons": ["IK 门槛"]}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertEqual(B["row_letters"], ["H"])
        self.assertTrue(all(v is None for v in B["labels"].values()))

    def test_only_h200(self):
        self.assertEqual(PM.ALLOWED_CARDS, ("h200",))
        fx = self.make(gate_block={BLK: {"card_type": "h100-96"}},
                       gpu={BLK: {c: "NVIDIA H100 NVL|95830 MiB" for c in CELLS}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertIn("只允许 h200", txt)
        fx2 = self.make(gate_block={BLK: {"card_type": "a100-80"}},
                        gpu={BLK: {c: "NVIDIA A100 80GB PCIe|81920 MiB" for c in CELLS}})
        self.assertEqual(self.block(fx2)[0]["status"], "H")

    def test_dump_gpu_disagrees_with_gate(self):
        fx = self.make(gpu={BLK: {"C8": "NVIDIA H100 NVL|95830 MiB"}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")
        self.assertEqual(B["gate"]["dump_cards"], ["h100-96", "h200"])

    def test_gate_protocol_sha_mismatch(self):
        fx = self.make(gate_sha="0" * 64)
        B, _, txt = self.block(fx)
        self.assertEqual(B["status"], "H")

    def test_bad_keys_union_removed_from_all_nine_cells(self):
        fx = self.make(gate_cells={BLK: {"C2": {"bad_keys": [[0, 20], [1, 21]]}, "C8": {"bad_keys": [[2, 22]]}}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["n_universe"], 97)
        for k in ("F", "P1", "D5", "D6", "D8"):
            self.assertEqual(B["estimates"][k]["n"], 97)


# ===================================================================================== sensitivity
class TestSensitivity(Base):
    def test_downgrade_when_label_changes(self):
        phys = {BLK: {"C4": ORDER[45:55]}}
        fx = self.make(blocks={BLK: {"sr": {"C4": 57}}}, phys=phys)
        B, V, txt = self.block(fx)
        self.assertIn("P1", B["sensitivity"]["changed"])
        self.assertEqual(B["status"], "不确定")
        self.assertNotIn("A", B["row_letters"])
        self.assertIn("pi05_spatial 为 不确定：K、K′ 都不判", "\n".join(V["summary_I"]["K"]["text"]))

    def test_no_downgrade_when_labels_stable(self):
        fx = self.make(phys={BLK: {"C2": [(0, 20)], "C5": [(3, 27)]}})
        B, _, _ = self.block(fx)
        self.assertEqual(B["sensitivity"]["n"], 98)
        self.assertEqual(B["row_letters"], ["A", "M"])


# ===================================================================================== guards
class TestGuards(Base):
    def test_nine_cells_required(self):
        fx = self.make(drop={BLK: {"C8": [(3, 27)]}})
        p = fx.run()
        self.refused(fx, p, 8)
        self.assertIn("C8 片 ep25-29 缺 1 键", p.stderr)
        fx2 = self.make()
        shutil.rmtree(os.path.join(fx2.runs, BLK, "C8"))
        p2 = fx2.run()
        self.refused(fx2, p2, 8)
        self.assertIn("C8 片 ep20-24 缺 50 键", p2.stderr)

    def test_incomplete_one_of_two_blocks_refuses_all(self):
        fx = self.make(blocks={"pi05_spatial": {}, "pi0_spatial": {}}, drop={"pi0_spatial": {"C0": [(0, 20)]}})
        p = fx.run("pi05_spatial,pi0_spatial")
        self.refused(fx, p, 8, ("pi05_spatial", "pi0_spatial"))

    def test_missing_field_refused(self):
        for fld in ("mi_obj_pos", "mi_cell", "mr_cell", "noise_seed", "mi_lag_step_actual"):
            fx = self.make(omit_field=(BLK, "C2", (4, 24), fld))
            p = fx.run()
            self.refused(fx, p, 9)
            self.assertIn(f"缺字段 ['{fld}']", p.stderr)

    def test_label_mismatch_refused(self):
        fx = self.make(mutate=(BLK, "C3", (0, 20), "mr_cell", np.array("C4")))
        p = fx.run()
        self.refused(fx, p, 9)
        self.assertIn("mi_cell / mr_cell / mi_block 与路径不符", p.stderr)

    def test_query_field_shape_refused(self):
        fx = self.make(mutate=(BLK, "C5", (0, 20), "mi_lag_step_target", np.zeros(3, np.int32)))
        p = fx.run()
        self.refused(fx, p, 9)
        fx2 = self.make(mutate=(BLK, "C5", (0, 20), "noise_seed", np.array(1.5)))
        self.refused(fx2, fx2.run(), 9)

    def test_old_block_names_refused(self):
        fx = self.make()
        p = fx.run("oft_spatial")
        self.refused(fx, p, 9, ())

    def test_duplicate_key_refused(self):
        fx = self.make(extra_file=(BLK, "C1", "t0_e20_i99.npz"))
        self.refused(fx, fx.run(), 9)

    def test_files_outside_design_ignored(self):
        fx = self.make(extra_file=(BLK, "C1", "t0_e40_i40.npz"))
        B, _, txt = self.block(fx)
        self.assertIn("之外的文件 1 个", txt)

    def test_lock_and_refuse_rerun(self):
        fx = self.make()
        self.ok_run(fx)
        self.assertTrue(os.path.exists(fx.lock(BLK)))
        before = (rtext(fx.out), rtext(fx.vj))
        p = fx.run()
        self.assertEqual(p.returncode, 3, p.stderr)
        self.assertEqual((rtext(fx.out), rtext(fx.vj)), before)

    def test_existing_lock_refused(self):
        fx = self.make()
        open(fx.lock(BLK), "w").close()
        p = fx.run()
        self.assertEqual(p.returncode, 5)
        self.assertFalse(os.path.exists(fx.out))

    def test_append_second_block_keeps_first_and_regenerates_K(self):
        fx = self.make(blocks={"pi05_spatial": {}, "pi0_spatial": {}})
        V1, t1 = self.ok_run(fx, "pi0_spatial")
        self.assertIsNone(V1["summary_I"]["K"]["row"])
        b1 = json.dumps(V1["blocks"]["pi0_spatial"], sort_keys=True)
        V2, t2 = self.ok_run(fx, "pi05_spatial")
        self.assertTrue(t2.startswith(t1))
        self.assertEqual(json.dumps(V2["blocks"]["pi0_spatial"], sort_keys=True), b1)
        self.assertEqual(V2["summary_I"]["K"]["row"], "K")
        self.assertEqual(fx.run("pi05_spatial").returncode, 3)

    def test_sha_mismatch(self):
        fx = self.make()
        self.refused(fx, fx.run(sha="f" * 64), 2)

    def test_not_before(self):
        fx = self.make()
        p = fx.run(extra=["--now", "2026-11-12T00:00:00+08:00", "--not-before", "2026-11-13T12:00:00+08:00"])
        self.refused(fx, p, 4)

    def test_now_requires_synthetic(self):
        fx = self.make()
        self.refused(fx, fx.run(extra=["--now", "2026-11-13T12:00:00+08:00"], synthetic=False), 4)

    def test_synthetic_requires_marker(self):
        fx = self.make(marker=False)
        self.refused(fx, fx.run(), 6)

    def test_readout_still_requires_its_arguments(self):
        fx = self.make()
        a = [x for x in fx.args(BLK) if x not in ("--gate", fx.gate)]
        p = subprocess.run([sys.executable, "-I", SCRIPT] + a, capture_output=True, text=True)
        self.refused(fx, p, 2)
        self.assertIn("--gate", p.stderr)

    def test_wording_hit_exit_7(self):
        fx = self.make(gate_block={BLK: {"block_ok": False, "block_fail_reasons": ["疑似纯架构效应"]}})
        p = fx.run()
        self.assertEqual(p.returncode, 7, p.stderr)
        self.assertIn("FAIL 命中 纯架构效应", rtext(fx.out))


# ===================================================================================== --check-fields-only
class TestCheckFieldsOnly(Base):
    SMOKE = [(0, 40)]

    def smoke(self, **kw):
        return self.make(blocks={b: {} for b in ("pi05_spatial", "pi05_object", "pi0_spatial", "pi0_object")},
                         keys=self.SMOKE, marker=False, **kw)

    def snapshot(self, fx):
        return sorted(os.path.relpath(os.path.join(d, f), fx.base) for d, _, fs in os.walk(fx.base) for f in fs)

    def test_smoke_keys_pass_and_nothing_written(self):
        fx = self.smoke()
        before = self.snapshot(fx)
        p = fx.check_fields("pi05_spatial,pi05_object,pi0_spatial,pi0_object", ["--keys", "0:40"])
        self.assertEqual(p.returncode, 0, p.stderr + p.stdout)
        self.assertIn("FIELDS_ALL_OK", p.stdout)
        self.assertEqual(p.stdout.count("FIELDS_OK"), 4)
        self.assertIn("格 9  文件 9", p.stdout)
        self.assertEqual(self.snapshot(fx), before)
        self.assertNotIn("True", p.stdout + p.stderr)
        self.assertNotIn("False", p.stdout + p.stderr)
        p2 = fx.check_fields("pi05_spatial")                     # default --keys all
        self.assertEqual(p2.returncode, 0, p2.stderr)

    def test_formal_fixture_passes(self):
        fx = self.make()
        p = fx.check_fields(BLK)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("文件 900", p.stdout)

    def test_missing_field_exit_9(self):
        for fld in ("mi_cell", "noise_seed", "mi_lag_step_target", "mr_cell", "mi_block"):
            fx = self.smoke(omit_field=("pi0_spatial", "C8", (0, 40), fld))
            p = fx.check_fields("pi05_spatial,pi0_spatial", ["--keys", "0:40"])
            self.assertEqual(p.returncode, 9, p.stdout)
            self.assertIn(f"pi0_spatial/C8/t0_e40_i40.npz: 缺字段 ['{fld}']", p.stderr)
            self.assertIn("FIELDS_REFUSE code=9", p.stdout)
            self.assertNotIn("FIELDS_ALL_OK", p.stdout)

    def test_label_mismatch_exit_9(self):
        fx = self.smoke(mutate=("pi05_object", "C1", (0, 40), "mi_cell", np.array("C2")))
        p = fx.check_fields("pi05_object", ["--keys", "0:40"])
        self.assertEqual(p.returncode, 9)

    def test_missing_key_exit_8_and_bad_args_9(self):
        fx = self.smoke()
        self.assertEqual(fx.check_fields("pi05_spatial", ["--keys", "0:41"]).returncode, 8)
        self.assertEqual(fx.check_fields("pi05_spatial", ["--keys", "0-40"]).returncode, 9)
        self.assertEqual(fx.check_fields("pi05_spatial", ["--cells", "C9"]).returncode, 9)
        self.assertEqual(fx.check_fields("oft_spatial").returncode, 9)
        p = fx.check_fields("pi05_spatial", ["--cells", "C0,C8", "--keys", "0:40"])
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn("格 2  文件 2", p.stdout)

    def test_success_value_is_never_read(self):
        # truncate the data byte of success.npy (header intact): loading the value would fail, the check must pass
        fx = self.smoke()
        f = os.path.join(fx.runs, "pi05_spatial", "C3", "t0_e40_i40.npz")
        with zipfile.ZipFile(f) as zf:
            members = {n: zf.read(n) for n in zf.namelist()}
        hdr_len = len(members["success.npy"]) - 1
        members["success.npy"] = members["success.npy"][:hdr_len]
        with zipfile.ZipFile(f, "w") as zf:
            for n, data in members.items():
                zf.writestr(n, data)
        with np.load(f) as z:
            with self.assertRaises(Exception):
                np.asarray(z["success"])
        p = fx.check_fields("pi05_spatial", ["--keys", "0:40"])
        self.assertEqual(p.returncode, 0, p.stderr)
        # a wrong success dtype is still caught from the header
        fx2 = self.smoke(mutate=("pi05_spatial", "C3", (0, 40), "success", np.array(0.5)))
        self.assertEqual(fx2.check_fields("pi05_spatial", ["--keys", "0:40"]).returncode, 9)


# ===================================================================================== numerics and wording
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
        self.assertEqual(PM.Boot(100, 2000).diff(Y, "a", "b"), mr.Boot(100, 2000).diff(Y, "a", "b"))
        self.assertEqual((PM.B_DEFAULT, PM.SEED), (10000, 0))

    def test_holm(self):
        h = PM.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
        self.assertAlmostEqual(h["a"], 0.04)
        self.assertAlmostEqual(h["c"], 0.09)
        self.assertAlmostEqual(h["b"], 0.09)
        self.assertAlmostEqual(h["d"], 0.5)

    def test_label_rules_and_tiers(self):
        r = lambda pt, lo, hi: {"point": pt, "ci95": (lo, hi)}
        self.assertEqual(PM.ci_label("rec", r(10, 0.1, 20)), PM.L_REC)
        self.assertEqual(PM.ci_label("rec", r(10, 0.0, 20)), PM.L_UNC)
        self.assertEqual(PM.ci_label("rec", r(3, -2, 9.99)), PM.L_NOREC)
        self.assertEqual(PM.TIER, {PM.L_REC: "达到标准", PM.L_NOREC: "排除 ≥ 10 点提高", PM.L_UNC: "不确定"})

    def test_holm_gates_secondary_label(self):
        c6 = from_ranges((0, 70))
        c2 = from_ranges((0, 54), (70, 75))
        fx = self.make(blocks={BLK: {"succ": {"C6": c6, "C2": c2}}})
        B, _, txt = self.block(fx)
        self.assertEqual(B["labels_ci_only"]["D5"], PM.L_REC)
        self.assertEqual(B["labels"]["D5"], PM.L_UNC_HOLM)

    def test_wording_checker(self):
        for w in ("与架构无关", "普遍", "OFT 特有", "三个独立架构", "π0 用本体感受补偿", "本体感受的作用得到验证",
                  "π0 不依赖视觉身体线索", "阴性不可读", "复现 Trace", "跨架构未复现", "两模型像素完全相同",
                  "找到了 seg-black 不掉分的原因", "原图策略依赖同一线索", "首次发现黑色遮罩保留剪影", "填充值决定结论",
                  "未检出轮廓收益", "与 Trace 一致", "与 Trace 不一致", "π 家族共有", "π0.5 特有", "纯架构效应",
                  "由 maskmech 回答", "由 maskprobe 回答", "视觉伺服", "充分"):
            self.assertIn(w, PM.wording_violations(f"前文{w}后文"), w)
        self.assertIn("π0 用本体感受补偿", PM.wording_violations("π0 用本体感受补偿了视觉"))
        for k, s in list(PM.ROW.items()) + list(PM.KROW.items()):
            self.assertEqual(PM.wording_violations(s), [], k)
        for s in PM.LIMITS + [PM.NOTE_B, PM.NOTE_F, PM.NOTE_G, PM.L_UNC_HOLM] + list(PM.A_CLAUSE.values()) + list(PM.CNAME.values()):
            self.assertEqual(PM.wording_violations(s), [], s)

    def test_wording_exemption_is_sentence_exact(self):
        g = [s for s in PM.LIMITS if s.startswith("(g)")][0]
        self.assertIn("原图策略依赖同一线索", g)
        self.assertEqual(PM.wording_violations(g), [])
        self.assertEqual(PM.wording_violations(g + "因此原图策略依赖同一线索"), ["原图策略依赖同一线索"])
        self.assertEqual(PM.wording_violations(g[:-3]), ["原图策略依赖同一线索"])

    def test_episode_events(self):
        ev = PM.episode_events(dict(episode(True, (0, 20), (20, 22, 30)), success=np.array(True)), "spatial")
        self.assertEqual({k: ev[k] for k in PM.EVENTS},
                         {"t_near": 20, "t_close": 22, "t_pad1": 23, "t_pad2": 24, "t_lift": 30, "t_carry": 30,
                          "t_drop": None})

    def test_header_and_timing_present(self):
        fx = self.make()
        B, V, txt = self.block(fx)
        self.assertEqual(B["gpu"]["C8"], {H200: 100})
        self.assertIn(f"协议 sha256 {PSHA}", txt)
        self.assertIn("protocol_maskinfo_pi.md v2", txt)
        self.assertIn("§5.4", txt)
        self.assertIn("C8", B["timing"])
        self.assertIn("Δ8 = SR(C1) − SR(C8)", txt)


if __name__ == "__main__":
    unittest.main()
