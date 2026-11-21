"""Synthetic-data tests for prereg_physfix.py, protocol_physfix.md REVISION 3 (12 cells, 48 slices, decision O only).
No pf_ or models1 data read; numpy only. Rewritten 2026-11-08 from the revision-1 tests (backup .bak_20261108_prerev3).

Run:  PYTHONDONTWRITEBYTECODE=1 TMPDIR=$HOME/tmp python test_prereg_physfix.py      (unittest; a few minutes on 4 CPUs)
Templates: one published OFT npz per suite ($PROJ_ROOT/runs/sham_cg/t0_e0_i42.npz, $PROJ_ROOT/runs/obj_s/t0_e0_i42.npz), read only;
success and all physfix fields rewritten, time series truncated to 3 rows (override root with PHYSFIX_PUB_RUNS).
Outcome coupling: per (checkpoint, suite) one fixed permutation u of the 200 keys; cell X succeeds on key k iff u_k < SR(X)
(comonotone, exact counts, narrow CIs), so each SR table maps to the intended branch.

Coverage (protocol rev3 §8 list, C / K / L / bridge no longer exist)
  O four branches + boundary ...... A (OFT 现象保持, OFTi 现象未保持 -> checkpoint 条件性), B (OFT D 1-2 剔除版翻转 -> 不确定),
                                     Rules.test_o / test_drop_boundary (掉分不可测, 不确定, -6.999999999999995 P20)
  D three branches ................. A all 0; B OFT object G'' 2 keys (1-2); B OFTi object G'' 3 QACC keys -> kp20 config 拒绝读,
                                     its SR never printed
  two checkpoints disagree ......... A
  Δ_old missing keys ............... A (old OFTi object GT lacks 10 keys)
  grasp triggers, one each ......... A (new grasp_plaus_pf.py format: pooled kp20, per-cell kp20_G2, slice0 kp20_GTfc2, missing items)
  §7.2 background_identical ........ A old-format gate json (FAIL) -> read as NA with warning, cells OK; B new format (NA) no warning
  extra: armature mismatch -> FAIL (B), GPU mixing -> 拒绝（卡型混杂）(B), 3 slices -> 未完成 + 47/48 (D), d5 rec' (B),
         non-H200 gate json -> twin cells FAIL (C), existing --out refused (B), pad residual line + gate0 json printed (A),
         C / K / L not judged sentence (A)
"""
import contextlib
import io
import json
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_physfix as pp  # noqa: E402

H = os.path.expanduser
PUB_RUNS = H(os.environ.get("PHYSFIX_PUB_RUNS", "$PROJ_ROOT/runs"))
B_TEST = 10000
A100 = "NVIDIA A100-PCIE-40GB"
KEYS = [(t, e) for t in range(10) for e in range(20)]

_TPL = {}


def npz_template(suite):
    if suite in _TPL:
        return _TPL[suite]
    z = np.load(os.path.join(PUB_RUNS, "sham_cg" if suite == "spatial" else "obj_s", "t0_e0_i42.npz"), allow_pickle=True)
    d = {}
    for k in z.files:
        v = z[k]
        if k in ("action", "eef_pos", "eef_quat", "gripper_qpos", "qpos"):
            v = v[:3]
        if k.startswith(("contact_", "twin_", "host", "gpu", "align_json", "pf_")):
            continue
        d[k] = v
    _TPL[suite] = d
    return d


def init_of(t, e):
    return int(np.random.RandomState(5478 + t).permutation(50)[e])


def outcomes(sr_table, seed):
    perm = np.random.RandomState(seed).permutation(len(KEYS))
    u = {k: (perm[i] + 0.5) / len(KEYS) for i, k in enumerate(KEYS)}
    return {c: {k: int(u[k] < sr) for k in KEYS} for c, sr in sr_table.items()}


def write_plain(path, suite, outc, gpu=None, drop=()):
    os.makedirs(path, exist_ok=True)
    tpl = npz_template(suite)
    for (t, e), s in outc.items():
        if (t, e) in drop:
            continue
        d = dict(tpl)
        d.update(success=bool(s), task=t, ep=e, init=init_of(t, e))
        if gpu:
            d.update(gpu=gpu, host="node21")
        np.savez(os.path.join(path, f"t{t}_e{e}_i{init_of(t, e)}.npz"), **d)


def write_pf(path, suite, code, outc, md):
    os.makedirs(path, exist_ok=True)
    tpl = npz_template(suite)
    spec = pp.CELLS[code]
    for i, ((t, e), s) in enumerate(sorted(outc.items())):
        if e >= md.get("keep_eps", 20):
            continue
        d = dict(tpl)
        al = {"pos_mm": 0.0016, "rot_deg": 0.0013, "fixture_mm": 14.739, "missing": [], "quat_sign": 1.0, "copy_grip_cmd": 1}
        gpu = md["gpu_half"] if ("gpu_half" in md and i % 2) else A100
        d.update(success=bool(s), task=t, ep=e, init=init_of(t, e), host="node21", gpu=gpu, align_json=json.dumps(al),
                 contact_arm_env_contact_steps=0, contact_first_contact=-1, contact_stall_steps=0, contact_pairs="{}",
                 contact_n_steps=3, contact_first_grip_obj=1, pf_blind=1,
                 pf_qacc_count=2 if (t, e) in md.get("qacc", ()) else 0,
                 pf_qacc_first_step=55 if (t, e) in md.get("qacc", ()) else -1,
                 pf_r85_armature=(1e-3 if spec["r85"] == 0 else 0.0) if (t, e) in md.get("arm_bad", ()) else spec["r85"],
                 pf_grip_armature=np.full(spec["grip"][0], spec["grip"][1]))
        if spec["robotiq"]:   # Robotiq finger joint appended at the end of qpos (Panda template has none)
            d["joint_names"] = np.array(list(tpl["joint_names"]) + ["gripper0_finger_joint"])
            d["qpos"] = np.concatenate([tpl["qpos"], np.array([[0.0], [0.70], [0.71]])], axis=1)
        if spec["twin"]:
            n = 17
            d.update(twin_pos_err_mm=np.full(n, 0.01, np.float32), twin_rot_err_deg=np.full(n, 0.02, np.float32), twin_stale=0,
                     twin_cam_err_mm=np.zeros(n, np.float32), twin_cam_err_deg=np.zeros(n, np.float32))
        np.savez(os.path.join(path, f"t{t}_e{e}_i{init_of(t, e)}.npz"), **d)


# ----------------------------------------------------------------------------------------------- old config + SRC
OLD_SR = {("OFT", "spatial"): {"S": .95, "T0": .95, "G": .50, "GT": .86},
          ("OFT", "object"): {"S": .96, "T0": .955, "G": .80, "GT": .74},
          ("OFTi", "spatial"): {"S": .96, "T0": .95, "G": .66, "GT": .875},
          ("OFTi", "object"): {"S": .96, "T0": .95, "G": .84, "GT": .75}}
OLD_DROP = {(t, 19) for t in range(10)}          # 10 keys missing in old OFTi object GT -> Δ_fix 配对缺键
SEEDS = {("OFT", "spatial"): 1, ("OFT", "object"): 2, ("OFTi", "spatial"): 3, ("OFTi", "object"): 4}


def build_old(root):
    pub, zoo = os.path.join(root, "pub_runs"), os.path.join(root, "zoo_runs")
    for suite, d in (("spatial", "src"), ("object", "obj_src")):
        write_plain(os.path.join(pub, d), suite, outcomes({"SRC": .98}, 9)["SRC"])
    for (ck, suite), tab in OLD_SR.items():
        oc = outcomes(tab, SEEDS[(ck, suite)] + 50)
        kind, sub, cells = pp.CKPTS[ck]["old"][suite]
        base = pub if kind == "pub" else os.path.join(zoo, sub)
        for c, d in cells.items():
            drop = OLD_DROP if (ck, suite, c) == ("OFTi", "object", "GT") else ()
            write_plain(os.path.join(base, d), suite, oc[c], gpu=None if kind == "pub" else A100, drop=drop)
    km1 = os.path.join(root, "kold_m1.json")
    kpub = os.path.join(root, "kold_pub.json")
    with open(km1, "w") as f:
        json.dump({"oft_obj|libero_object": [[t, e] for t in range(10) for e in (0, 1, 2)] + [[0, 3]], "oft_sp|libero_spatial": []}, f)
    with open(kpub, "w") as f:
        json.dump({"oft_pub|libero_object": [[t, 4] for t in range(10)] + [[t, 5] for t in range(4)], "oft_pub|libero_spatial": [[3, 19]]}, f)
    return pub, zoo, km1, kpub


# ----------------------------------------------------------------------------------------------- new pf data
SC_A = {
    ("OFT", "spatial"): {"S": .95, "T0": .95},
    ("OFTi", "spatial"): {"S": .96, "T0": .95},
    ("OFT", "object"): {"S": .96, "T0": .95, "G''": .80, "GTfc''": .76},
    ("OFTi", "object"): {"S": .96, "T0": .95, "G''": .84, "GTfc''": .90},
}
SC_B = {k: dict(v) for k, v in SC_A.items()}
SC_B[("OFT", "object")].update({"T0": .85, "G''": .89, "GTfc''": .95})
SC_B[("OFTi", "object")].update({"G''": .82, "GTfc''": .79})
MODS_B = {("OFT", "spatial", "T0"): {"arm_bad": [(5, 5)]},
          ("OFTi", "spatial", "T0"): {"gpu_half": "NVIDIA H100 80GB HBM3"},
          ("OFTi", "object", "G''"): {"qacc": [(1, 1), (2, 2), (3, 3)]}}
MODS_D = {("OFT", "object", "GTfc''"): {"keep_eps": 15}}


def build_new(root, spec, mods=None, bd_auto=None):
    """bd_auto: (ck, suite, code) whose breakdown json gets 2 keys where S succeeds and G'' fails."""
    mods = mods or {}
    pf = os.path.join(root, "pf_runs")
    bd = {"_by_cell": {}, "_meta": {}}
    for (ck, suite), tab in spec.items():
        oc = outcomes(tab, SEEDS[(ck, suite)])
        for code, sr in tab.items():
            d = os.path.join(pf, pp.CKPTS[ck]["pf"][suite], pp.cell_dir(code, suite))
            md = mods.get((ck, suite, code), {})
            write_pf(d, suite, code, oc[code], md)
            keys = []
            if bd_auto == (ck, suite, code):
                keys = [k for k in KEYS if oc["S"][k] == 1 and oc["G''"][k] == 0][:2]
            bd["_by_cell"][f"{pp.CKPTS[ck]['pf'][suite]}|libero_x|{pp.cell_dir(code, suite)}"] = {
                "dir": d, "keys": 200, "no_qpos": 0, "shards": [0, 5, 10, 15], "breakdown": [list(k) for k in keys],
                "A_arm": {f"{k[0]},{k[1]}": {"first_step": 100, "max_jump_deg": 200.0, "beyond_limit": True} for k in keys},
                "B_finger": {}, "breakdown_loose": [], "passive_overlimit_steps": {"p50": 0}}
    bdp = os.path.join(root, "breakdown.json")
    with open(bdp, "w") as f:
        json.dump(bd, f)
    return pf, bdp


def side_jsons(root):
    pl = os.path.join(root, "padlr.json")
    one = lambda a, b, c: {"p50": a, "p90": b, "max": c, "min": a}  # noqa: E731
    st = {"n": 160, "w_R": one(12.3, 12.6, 12.6), "w_P": one(-0.7, -0.6, -0.6), "dw": one(13.0, 13.2, 13.2),
          "res_p": one(6.5, 6.6, 6.7), "res_m": one(6.6, 6.7, 6.7)}
    with open(pl, "w") as f:
        json.dump({"sets": {"free_closed_end": st, "free": dict(st, n=1040), "blocked_object": dict(st, n=2000)}}, f)
    gp = os.path.join(root, "grasp.json")
    base = {"no_pad_contact_carried_frac": 0.0, "penetration_p90": 2.0, "carry_frac_median": 0.9, "slip_p90": 3.0}
    old = {"no_pad_contact_carried_frac": 0.2, "penetration_p90": 2.0, "carry_frac_median": 0.9}
    g = {"slices": {
        "full": {"old_normal": old,
                 "OFT/object": {"kp20": {**base, "no_pad_contact_carried_frac": 0.02}, "kp20_G2": dict(base), "kp20_GTfc2": dict(base), "old_normal": old},
                 "OFTi/object": {"kp20": dict(base), "kp20_G2": {**base, "penetration_p90": 6.0}, "kp20_GTfc2": dict(base), "old_normal": old}},
        "slice0": {"old_normal": old,
                   "OFT/object": {"kp20": dict(base), "kp20_G2": dict(base), "kp20_GTfc2": {**base, "carry_frac_median": 0.6}, "old_normal": old},
                   "OFTi/object": {"kp20": {"slip_p90": 1.0}, "old_normal": old}}}}
    with open(gp, "w") as f:
        json.dump(g, f)
    g0 = os.path.join(root, "gate0.json")
    with open(g0, "w") as f:
        json.dump({"groups": {"OFT/object": {"cells": {"S": {"slices": {"e0-4": {"status": "PASS"}}}},
                                             "T0_vs_S_agreement_slice0": {"keys": 50, "rate": 0.98, "status": "PASS"}}},
                   "all_slices_no_FAIL": True}, f)
    return pl, gp, g0


def gate_json(path, renderer="NVIDIA H200 NVL/PCIe/SSE2", old_bg=False):
    cells = {}
    for code, spec in pp.CELLS.items():
        if not spec["gj"]:
            continue
        for suite in spec["suites"]:
            d = pp.cell_dir(code, suite)
            st = "PASS"
            if code == "T0":
                items = {"t0_vs_s_agent": {"status": "PASS", "value": 0.01}, "t0_vs_s_wrist": {"status": "PASS", "value": 0.04},
                         "keys": {"status": "NA", "value": 50}}
            else:
                items = {"wrist_both_hidden_mean": {"status": "PASS", "value": 0.0},
                         "agent_both_hidden_mean": {"status": "NA", "value": 1.4636}, "agent_lb": {"status": "NA", "value": 1.4581},
                         "agent_excess": {"status": "PASS", "value": 0.0055}, "agent_max_key_excess": {"status": "NA", "value": 0.014},
                         "background_identical": {"status": "FAIL" if old_bg else "NA", "value": 81},
                         "twin_phys_joint_diff": {"status": "PASS", "value": 0.0}, "keys": {"status": "NA", "value": 50}}
                st = "FAIL" if old_bg else "PASS"
            cells[d] = {"status": st, "items": items}
    with open(path, "w") as f:
        json.dump({"gl_renderer": renderer, "source": [], "regression": {"status": "PASS", "items": {}}, "cells": cells}, f)
    return path


def run_main(root, pf, bdp, gj, old, tag, extra=()):
    pub, zoo, km1, kpub = old
    out, vj = os.path.join(root, f"res_{tag}.txt"), os.path.join(root, f"v_{tag}.json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(io.StringIO()):
        rc = pp.main(["--out", out, "--verdict-json", vj, "--B", str(B_TEST), "--pf-runs", pf, "--zoo-runs", zoo, "--pub-runs", pub,
                      "--zoo-logs", os.path.join(root, "nologs"), "--gate-json", gj, "--breakdown-json", bdp,
                      "--kold-m1", km1, "--kold-pub", kpub, "--protocol", os.path.join(root, "proto.md"),
                      "--cells-next", os.path.join(root, "no.sh"), "--body-env", os.path.join(root, "no.py"),
                      "--robots", os.path.join(root, "no2.py"), *extra])
    txt, v = "", None
    if os.path.exists(out):
        txt = open(out).read()
    if os.path.exists(vj):
        v = json.load(open(vj))
    return rc, v, txt


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="physfix_test_")
        cls.old = build_old(cls.tmp)
        cls.res = {}
        ra = os.path.join(cls.tmp, "A")
        pfa, bda = build_new(ra, SC_A)
        pl, gp, g0 = side_jsons(ra)
        cls.res["A"] = run_main(ra, pfa, bda, gate_json(os.path.join(ra, "gate_oldbg.json"), old_bg=True), cls.old, "A",
                                ["--padlr-json", pl, "--grasp-json", gp, "--gate0-json", g0])
        cls.res["C"] = run_main(ra, pfa, bda, gate_json(os.path.join(ra, "gate_a100.json"), "NVIDIA A100-PCIE-40GB/PCIe/SSE2"), cls.old, "C")
        rb = os.path.join(cls.tmp, "B")
        pfb, bdb = build_new(rb, SC_B, MODS_B, bd_auto=("OFT", "object", "G''"))
        cls.b_args = (rb, pfb, bdb, gate_json(os.path.join(rb, "gate.json")))
        cls.res["B"] = run_main(*cls.b_args, cls.old, "B")
        rd = os.path.join(cls.tmp, "D")
        pfd, bdd = build_new(rd, SC_A, MODS_D)
        cls.res["D"] = run_main(rd, pfd, bdd, gate_json(os.path.join(rd, "gate.json")), cls.old, "D")

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("PHYSFIX_KEEP"):
            shutil.rmtree(cls.tmp, ignore_errors=True)
        else:
            print("kept", cls.tmp)

    def v(self, tag):
        rc, v, txt = self.res[tag]
        self.assertEqual(rc, 0, txt[-3000:])
        return v, txt

    def test_A_cells_gates_bg_override(self):
        v, txt = self.v("A")
        self.assertEqual(len(v["cells"]), 12)
        for k, c in v["cells"].items():
            self.assertEqual(c["status"], "OK", (k, c))
        self.assertEqual(v["slices_complete"], 48)
        self.assertIn("12 格 48 片）：48/48", txt)
        self.assertEqual(v["cells"]["OFT/object/GTfc''"]["gates"]["第 0 步一致性"], "PASS")
        self.assertIn("旧版 gate json 把描述量当门槛，已按 NA 读", txt)
        self.assertIn("M_套件（有限 IK 搜索最小值，不是理论下界）", txt)
        self.assertIn("被动内指关节越限（只印）", txt)
        self.assertIn("合拢平台中位", txt)
        self.assertIn("MISSING", txt)
        self.assertTrue(all(t == "0" for t in v["D"].values()))
        self.assertEqual(set(v["D"]), {"OFT/object/kp20", "OFTi/object/kp20"})

    def test_A_O(self):
        v, txt = self.v("A")
        L = pp.O_LBL
        self.assertEqual(v["O"]["OFT"], L["keep"])
        self.assertEqual(v["O"]["OFTi"], L["lost"])
        self.assertEqual(v["O"]["family"], "checkpoint 条件性")
        self.assertIn(pp.TXT["ckpt_cond"], txt)
        self.assertIn(pp.TXT["O_keep"].format(k=1), txt)
        self.assertIn(pp.TXT["O_lost"].format(k=1), txt)
        self.assertIn("该 kp20 物理及既定开度映射下的完整重绘管线收益", txt)
        self.assertIn(pp.PADNOTE, txt)
        self.assertIn("右指垫残差 6.5/6.6/6.7 mm", txt)
        self.assertIn(pp.TXT["NOT_JUDGED"], txt)
        self.assertNotIn("决策 C：spatial", txt)
        self.assertNotIn("决策 L", txt)
        self.assertIn("数字写于 XML 分支", txt)

    def test_A_delta_kold(self):
        v, txt = self.v("A")
        self.assertEqual(v["delta_old"]["OFTi/object/recovery"]["missing_pairs"], 10)
        self.assertEqual(v["delta_old"]["OFTi/object/recovery"]["n"], 190)
        self.assertEqual(v["delta_old"]["OFTi/object/drop_nat"]["missing_pairs"], 0)
        self.assertNotIn("OFT/spatial/drop_nat", v["delta_old"])
        self.assertIn("（旧配置缺配对键 10）", txt)
        self.assertIn("配置不同", txt)
        self.assertIn("Δ_old 不解释为失效修复的效应", txt)
        self.assertEqual(v["kold"]["OFTi/object/S"][1], 31)
        self.assertEqual(v["kold"]["OFTi/object/S"][1] + v["kold"]["OFTi/object/S"][3], 200)
        self.assertEqual(v["kold"]["OFT/object/G''"][1], 14)
        self.assertEqual(v["kold"]["OFT/spatial/T0"][1], 1)

    def test_A_grasp_gate0(self):
        v, txt = self.v("A")
        g = v["grasp_triggers"]
        t1, t2, t3 = list(pp.grasp_triggers({}, {}).keys())
        self.assertEqual(g["full/OFT/object/kp20"], {t1: True, t2: False, t3: False})
        self.assertEqual(g["full/OFTi/object/kp20"], {t1: False, t2: False, t3: False})
        self.assertEqual(g["full/OFTi/object/kp20_G2"], {t1: False, t2: True, t3: False})
        self.assertEqual(g["slice0/OFT/object/kp20_GTfc2"], {t1: False, t2: False, t3: True})
        self.assertEqual(g["slice0/OFTi/object/kp20"], {t1: None, t2: None, t3: None})
        self.assertTrue(v["grasp_trigger_any"])
        self.assertIn("触发 -> 交 Astra", txt)
        self.assertIn("全部片无 FAIL", txt)
        self.assertIn("T0 对 S 第 0 片一致率 0.98", txt)

    def test_B_D_sens(self):
        v, txt = self.v("B")
        self.assertEqual(v["D"]["OFT/object/kp20"], "1-2")
        self.assertEqual(v["O"]["OFT"], pp.O_LBL["unsure"])
        self.assertIn("与主读数不同 -> 降为不确定（对残余失效敏感）", txt)
        self.assertIn("§5.5 剔除版", txt)
        self.assertEqual(v["cells"]["OFT/object/G''"]["gates"]["检测器 D + QACC"], "WARN")
        self.assertNotIn("旧版 gate json 把描述量当门槛", txt)

    def test_B_D_refuse_no_leak(self):
        v, txt = self.v("B")
        self.assertEqual(v["D"]["OFTi/object/kp20"], ">=3")
        self.assertEqual(v["cells"]["OFTi/object/G''"]["status"], "REFUSED")
        self.assertEqual(v["cells"]["OFTi/object/GTfc''"]["status"], "REFUSED")   # same configuration
        self.assertEqual(v["O"]["OFTi"], "拒绝读")
        self.assertEqual(v["O"]["family"], "未能判定")
        self.assertNotIn("82.0%", txt)
        self.assertNotIn("164/200", txt)
        for k, x in v["est"].items():
            if k.startswith("OFTi/object/kp20/") and k != "OFTi/object/kp20/d5":
                self.assertEqual(x, "refused", k)
        self.assertNotIn("OFTi/object/G''", v["kold"])
        self.assertNotIn("OFTi/object/GTfc''", v["kold"])
        self.assertIn("拒绝读（检测器 D，配置 kp20）", txt)
        self.assertIn(pp.TXT["D_refuse"], txt)
        self.assertNotEqual(v["cells"]["OFTi/object/S"]["status"], "REFUSED")   # Panda configuration still read

    def test_B_armature_gpu_d5_rerun(self):
        v, txt = self.v("B")
        self.assertEqual(v["cells"]["OFT/spatial/T0"]["gates"]["armature"], "FAIL")
        self.assertEqual(v["cells"]["OFT/spatial/T0"]["status"], "FAIL")
        self.assertEqual(v["cells"]["OFTi/spatial/T0"]["gates"]["GPU 单一型号"], "FAIL")
        self.assertIsNone(v["d5_trigger"]["OFTi/spatial"])
        self.assertIn("拒绝（卡型混杂", txt)
        self.assertTrue(v["d5_trigger"]["OFT/object"])
        self.assertIn("rec′(GTfc'')", txt)
        self.assertIn("替身管线代价 d5 =", txt)
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _, _ = run_main(*self.b_args, self.old, "B")
        self.assertEqual(rc, 3)

    def test_C_non_h200(self):
        v, txt = self.v("C")
        self.assertIn("不含 H200", txt)
        for k, c in v["cells"].items():
            code = k.split("/")[-1]
            if pp.CELLS[code]["gj"]:
                self.assertEqual(c["gates"]["第 0 步一致性"], "FAIL", k)
                self.assertEqual(c["status"], "FAIL", k)
            self.assertEqual(c["gates"]["回归（§7.1）"], "UNK", k)
        self.assertEqual(v["O"]["OFT"], "does not count")

    def test_D_incomplete_slices(self):
        v, txt = self.v("D")
        self.assertEqual(v["cells"]["OFT/object/GTfc''"]["slices"], [0, 1, 2])
        self.assertEqual(v["slices_complete"], 47)
        self.assertIn("47/48；按整片规则读已齐的", txt)
        self.assertEqual(v["est"]["OFT/object/kp20/recovery(G'';GTfc'')"][3], "未完成")
        self.assertNotIn(v["O"]["OFT"], pp.O_LBL.values())


def R(pt, lo, hi):
    return {"point": pt, "ci95": (lo, hi)}


class Rules(unittest.TestCase):
    def test_drop_boundary(self):
        self.assertTrue(pp.drop_exists(R(-7.0, -13, -1)))
        self.assertTrue(pp.drop_exists(R(100 * (0.89 - 0.96), -13, -1)))   # -6.999999999999995 (P20)
        self.assertFalse(pp.drop_exists(R(-6.5, -13, -1)))
        self.assertFalse(pp.drop_exists(R(-12, -18, 0.0)))   # upper bound 0 -> not < 0
        self.assertTrue(pp.drop_exists(R(-12, -18, -0.01)))

    def test_o(self):
        L = pp.O_LBL
        self.assertEqual(pp.o_verdict(R(-3, -10, 5.9), R(-12, -18, -6)), L["keep"])
        self.assertEqual(pp.o_verdict(R(-3, -10, 6.0), R(-12, -18, -6)), L["unsure"])
        self.assertEqual(pp.o_verdict(R(8, 1, 15), R(-12, -18, -6)), L["lost"])
        self.assertEqual(pp.o_verdict(R(5, 1, 12), R(-12, -18, -6)), L["unsure"])     # point < half
        self.assertEqual(pp.o_verdict(R(20, 10, 30), R(-6, -12, 0)), L["vanish"])
        self.assertEqual(pp.o_family({"OFT": L["keep"], "OFTi": L["keep"]})[0], L["keep"] + "（2/2 checkpoint）")
        self.assertEqual(pp.o_family({"OFT": L["vanish"], "OFTi": L["vanish"]})[0], L["vanish"] + "（2/2 checkpoint）")
        self.assertEqual(pp.o_family({"OFT": L["keep"], "OFTi": L["lost"]})[0], "checkpoint 条件性")
        self.assertEqual(pp.o_family({"OFT": L["keep"], "OFTi": "拒绝读"})[0], "未能判定")
        self.assertIn("2/2 checkpoint", pp.o_text(L["keep"], 2))

    def test_no_cancelled_cells(self):
        self.assertEqual(set(pp.CELLS), {"S", "T0", "G''", "GTfc''"})
        self.assertEqual([f["id"] for f in pp.FAMILIES], ["kp20"])
        self.assertEqual(pp.N_SLICES_EXPECTED, 48)
        self.assertIn("background_identical", pp.GJ_DESCRIPTIVE)

    def test_final_wording(self):   # blind review 2026-11-07, final text 2026-11-08
        self.assertIn("整帧替身渲染，包含轮廓混合像素及适用场景中的机器人反射", pp.TXT["LIMIT"])
        self.assertIn("撤回旧的", pp.TXT["O_lost"])
        self.assertIn("旧数字进附录", pp.TXT["O_keep"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
