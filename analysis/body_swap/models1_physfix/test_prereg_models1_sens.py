"""Synthetic-data tests for prereg_models1_sens.py (Fable add1 §2.3 item 2). No vla_zoo / models1 data is read.

Run:  PYTHONDONTWRITEBYTECODE=1 python test_prereg_models1_sens.py
Data: exactly the synthetic scenarios of the frozen test_prereg_models1.py (same builders, same read-only published
templates whose success field is rewritten). That module is imported unchanged; its module-level name `pm` is rebound
to prereg_models1_sens, so
  1. every original test (Scenarios A-D end to end + all Rules unit tests) runs against the sens script without
     --exclude-keys / --bounds;
  2. on the same synthetic data the frozen prereg_models1.main is run too: the output text must be identical line by
     line after removing the sens header block (first line .. the original first line) and the two "（本脚本）" sha lines;
     the verdict json must be identical after removing the "sens_physics" key;
  3. --exclude-keys with 3 OFTi object keys (chosen as S=1, G=0 keys so drop_nat(G) must move): every OFTi object
     estimand row has n = 197 and the point equals a hand computation on the remaining keys; Δ_model OFTi object rows
     n = 197; one dense OFTi spatial key (ep 25) only moves the 400-key rows (n = 399); every estimand row of the other
     models and every gate section is identical to the run without exclusion; an unmatched json key is harmless;
  4. --bounds: π0 / OVLA estimand rows end with "失效集上界 ±x.x" with x = 100 Σ|coef| w / n (rounded up to 0.1),
     "缺（…）" when a cell is absent; OFTi rows carry no bound; the rest of the text is unchanged; sens_bound only sums
     the slices the estimand uses.
"""
import contextlib
import io
import json
import os
import re
import sys
import types
import unittest

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import prereg_models1 as pm_orig  # noqa: E402
import prereg_models1_sens as ps  # noqa: E402
import test_prereg_models1 as T  # noqa: E402

T.pm = ps            # the frozen test now drives the sens script
Rules = T.Rules      # original unit tests, collected here (they use T.pm = ps)

HEAD = "物理失效集剔除版，diff 见 prereg_models1_sens.diff"
ORIG_FIRST = "prereg_models1.py：models1 预注册分析"
EST_RX = re.compile(r"^  (\S+)\s+([+-]\d+\.\d+) pp  95% \[.*?\]  90% \[.*?\]  n=(\d+) 片=(\d+)(.*)$")
DELTA_RX = re.compile(r"^  (.+?)\s+([+-]\d+\.\d+)  95% \[.*?\]  90% \[.*?\]  n=(\d+)(.*)$")
BOUND_RX = re.compile(r"  失效集上界 (±\d+\.\d|缺（.*）)$")


def run_mod(mod, root, zr, pr, p5, tag, extra=()):
    out, vj = os.path.join(root, f"res_{tag}.txt"), os.path.join(root, f"v_{tag}.json")
    with contextlib.redirect_stdout(io.StringIO()):
        rc = mod.main(["--out", out, "--verdict-json", vj, "--B", str(T.B_TEST), "--zoo-runs", zr, "--pub-runs", pr, "--pi05", p5,
                       "--zoo-logs", os.path.join(root, "nologs"), "--protocol", os.path.join(root, "proto.md"),
                       "--decisions", os.path.join(root, "nodec.md"), "--zoo-cells", os.path.join(root, "nocells.sh"), *extra])
    with open(out) as fh:
        txt = fh.read()
    with open(vj) as fh:
        v = json.load(fh)
    return rc, v, txt


def strip_sens(txt):
    L = txt.split("\n")
    assert L[0] == HEAD, L[0]
    i = next(i for i, x in enumerate(L) if x.startswith(ORIG_FIRST))
    return [x for x in L[i:] if "（本脚本）" not in x and not x.startswith("  [剔除版]")]


def strip_orig(txt):
    return [x for x in txt.split("\n") if "（本脚本）" not in x]


def sections(lines):
    """{(section header, row label): (point, n, rest)} for §4 / R2 estimand rows and Δ_model rows; and {header: [lines]}."""
    rows, secs, cur = {}, {}, None
    for x in lines:
        if x.startswith(("==== ", "######## ")):
            cur = x
            secs[cur] = []
            continue
        if cur is None:
            continue
        secs[cur].append(x)
        if "Δ_model = est(" in cur:
            m = DELTA_RX.match(x)
            if m:
                rows[(cur, m.group(1))] = (float(m.group(2)), int(m.group(3)), m.group(4))
        else:
            m = EST_RX.match(x)
            if m:
                rows[(cur, m.group(1))] = (float(m.group(2)), int(m.group(3)), m.group(5))
    return rows, secs


def is_ofti_obj(h):
    return "（OFTi，object）" in h or ("est(OFTi)" in h and "（object，" in h)


class SensScenarios(T.Scenarios):
    """All original scenario tests (inherited) + equivalence / exclusion / bounds on the same synthetic data."""

    @classmethod
    def setUpClass(cls):
        super().setUpClass()                    # runs ps.main on A-D (T.pm = ps) -> cls.res
        cls.orig = {}
        for tag in "ABCD":
            root = os.path.join(cls.tmp, tag)
            zr = cls.d_zr if tag == "D" else os.path.join(root, "zoo", "runs")
            cls.orig[tag] = run_mod(pm_orig, root, zr, cls.pr, cls.p5, "orig_" + tag)
        # exclusion: 3 OFTi object keys with S = 1, G = 0 (seed of T.build_new: OFTi 11 + 100 for object)
        cls.oc_obj = T.outcomes(T.SC_A[("OFTi", "object")], 111)
        cls.ex_obj = [k for k in sorted(cls.oc_obj["S"]) if cls.oc_obj["S"][k] == 1 and cls.oc_obj["G"][k] == 0][:3]
        root = os.path.join(cls.tmp, "A")
        cls.zrA = os.path.join(root, "zoo", "runs")
        exf = os.path.join(root, "exclude.json")
        with open(exf, "w") as fh:
            json.dump({"_comment": "synthetic", "oft_obj|libero_object": [list(k) for k in cls.ex_obj],
                       "oft_sp|libero_spatial": [[2, 25]], "nosuch|libero_object": [[0, 0]]}, fh)
        cls.ex = run_mod(ps, root, cls.zrA, cls.pr, cls.p5, "A_ex", ["--exclude-keys", exf])
        bdf = os.path.join(root, "bounds.json")
        z4 = {"0": 0, "5": 0, "10": 0, "15": 0}
        with open(bdf, "w") as fh:
            json.dump({"_comment": "synthetic",
                       "pi0|obj_grpkp100": {"0": 1, "5": 0, "10": 2, "15": 0}, "pi0|obj_sham_cg": z4, "pi0|obj_sham_twin": z4,
                       "pi0|obj_g_tfc": {"0": 4, "5": 0, "10": 0, "15": 0},
                       "openvla|g_tw_fc": {"0": 1, "5": 0, "10": 0, "15": 0}, "openvla|grpkp100": z4, "openvla|g_ta_fc": z4,
                       "openvla|sham_cg": z4, "openvla|g_tfc": z4}, fh)
        cls.bd = run_mod(ps, root, cls.zrA, cls.pr, cls.p5, "A_bd", ["--bounds", bdf])

    # ------------------------------------------------------------ 2. identical to the frozen script without flags
    def test_equivalence_with_frozen(self):
        for tag in "ABCD":
            rc_o, v_o, t_o = self.orig[tag]
            rc_s, v_s, t_s = self.res[tag]
            self.assertEqual((rc_o, rc_s), (0, 0), tag)
            a, b = strip_orig(t_o), strip_sens(t_s)
            self.assertEqual(len(a), len(b), tag)
            for i, (x, y) in enumerate(zip(a, b)):
                self.assertEqual(x, y, f"{tag} line {i}")
            self.assertGreater(len(a), 300, tag)
            self.assertIn("sens_physics", v_s)
            v_s = {k: x for k, x in v_s.items() if k != "sens_physics"}
            self.assertEqual(v_o, v_s, tag)
            L = t_s.split("\n")
            self.assertEqual(L[0], HEAD)
            self.assertIn("prereg_models1.py（冻结原版，同目录）", L[2])
            self.assertTrue(any(x.strip().startswith("--exclude-keys") and x.rstrip().endswith("未给") for x in L[:8]))
            self.assertFalse(any("失效集上界" in x for x in b))          # body only (the header 口径 line names it)
            self.assertNotIn("  [剔除版]", t_s)

    # ------------------------------------------------------------ 3. exclusion
    def test_exclusion(self):
        rc, v, txt = self.ex
        self.assertEqual(rc, 0)
        self.assertEqual(len(self.ex_obj), 3)
        rows0, secs0 = sections(strip_sens(self.res["A"][2]))
        rows1, secs1 = sections(strip_sens(txt))
        self.assertEqual(set(rows0), set(rows1))
        keep = [k for k in sorted(self.oc_obj["S"]) if k not in self.ex_obj]
        n_obj = n_dense = n_same = 0
        for key, (pt, n, rest) in rows1.items():
            h, nm = key
            pt0, n0, rest0 = rows0[key]
            if is_ofti_obj(h):
                self.assertEqual(n, 197, key)
                self.assertEqual(n0, 200, key)
                if "Δ_model" not in h:
                    terms = ps.EST[nm]
                    exp = 100 * sum(a * sum(self.oc_obj[c][k] for k in keep) / len(keep) for a, c in terms)
                    self.assertAlmostEqual(pt, exp, delta=0.006, msg=key)
                n_obj += 1
            elif "R2 通道加密估计量（400 键）（OFTi，spatial）" in h:
                self.assertEqual((n0, n), (400, 399), key)
                n_dense += 1
            else:
                self.assertEqual((pt, n, rest), (pt0, n0, rest0), key)   # other models / 200-key OFTi spatial: unchanged
                n_same += 1
        self.assertEqual(n_dense, 3)
        self.assertGreaterEqual(n_obj, len(ps.EST) + 3)
        self.assertGreater(n_same, 40)
        # drop_nat(G) moved: 3 keys with S = 1, G = 0 removed
        h = next(h for h, nm in rows1 if nm == "drop_nat(G)" and "§4 估计量（200 键）（OFTi，object）" in h)
        self.assertNotEqual(rows1[(h, "drop_nat(G)")][0], rows0[(h, "drop_nat(G)")][0])
        # gate sections and task tables untouched (gates are not re-licensed by the exclusion)
        for hh, ls in secs0.items():
            if hh.startswith("==== 门槛（") or "10 任务表" in hh or "在线接触汇总" in hh:
                self.assertEqual(secs1[hh], ls, hh)
        self.assertEqual(v["cells"], self.res["A"][1]["cells"])
        self.assertEqual(v["sens_physics"]["exclude_n"], {"oft_obj|libero_object": 3, "oft_sp|libero_spatial": 1, "nosuch|libero_object": 1})
        self.assertIn(f"[剔除版] 物理失效键 3 个在整片内", txt)
        self.assertIn(f"剔除表 oft_obj|libero_object: 3 键 {sorted(self.ex_obj)}", txt)
        for k in ("R1", "R2", "R3", "F1", "F2", "O2", "E1", "E3", "E4", "E5"):   # verdicts of untouched models
            self.assertEqual(v[k], self.res["A"][1][k], k)

    # ------------------------------------------------------------ 4. bounds
    def test_bounds(self):
        rc, v, txt = self.bd
        self.assertEqual(rc, 0)
        rows, _ = sections(strip_sens(txt))
        got = {}
        for (h, nm), (pt, n, rest) in rows.items():
            m = BOUND_RX.search(rest)
            if "Δ_model" in h:
                self.assertIsNone(m, (h, nm))
                continue
            if "（OFTi，" in h:
                self.assertIsNone(m, (h, nm))
                continue
            self.assertIsNotNone(m, (h, nm))
            got[(h.split("（")[-1].split("）")[0], nm)] = m.group(1)
        exp = {("π0，object", "drop_nat(G)"): "±1.5", ("π0，object", "d5"): "±0.0", ("π0，object", "rec_fc"): "±3.5",
               ("π0，object", "recovery(G)"): "±3.5", ("OVLA，spatial", "rec_w"): "±0.5", ("OVLA，spatial", "channel"): "±0.5",
               ("OVLA，spatial", "drop_nat(G)"): "±0.0", ("OVLA，spatial", "rec_a"): "±0.0", ("OVLA，spatial", "rec_fc"): "±0.0"}
        for k, x in exp.items():
            self.assertEqual(got.get(k), x, k)
        self.assertTrue(got[("π0，object", "rec_w")].startswith("缺（pi0|obj_g_tw_fc"))
        self.assertTrue(got[("π0，spatial", "drop_nat(G)")].startswith("缺（"))
        self.assertTrue(got[("OVLA，spatial", "d5")].startswith("缺（openvla|sham_twin"))
        # nothing else changed
        a = [BOUND_RX.sub("", x) for x in strip_sens(txt)]
        self.assertEqual(a, strip_sens(self.res["A"][2]))
        self.assertIn("警告表 pi0|obj_grpkp100: e0=1, e5=0, e10=2, e15=0（合计 3）", txt)

    def test_bound_uses_only_estimand_slices(self):
        keys200 = [(t, e) for t in range(10) for e in range(20)]
        m = types.SimpleNamespace(published=False, kind="jsonl", keys200=keys200, keys400=[(t, e) for t in range(10) for e in range(40)],
                                  spec={c: (c, d, None, None) for c, d in (("G", "grpkp100"), ("S", "sham_cg"))}, sens_model_dir="pi0")
        r = {"mask": [k[1] >= 5 for k in keys200], "n": 150}
        old = ps.SENS["bounds"]
        try:
            ps.SENS["bounds"] = {"pi0|grpkp100": {"0": 7, "5": 1, "10": 1, "15": 1}, "pi0|sham_cg": {"0": 0, "5": 0, "10": 1, "15": 0}}
            self.assertEqual(ps.sens_bound(m, "drop_nat(G)", r), "  失效集上界 ±2.7")   # 100 * 4 / 150 = 2.67 -> 2.7
            ps.SENS["bounds"]["pi0|sham_cg"] = {"0": 0, "5": 0, "10": 1}
            self.assertEqual(ps.sens_bound(m, "drop_nat(G)", r), "  失效集上界 缺（pi0|sham_cg@15）")
            m.kind = "npz"
            self.assertEqual(ps.sens_bound(m, "drop_nat(G)", r), "")
            m.kind, ps.SENS["bounds"] = "jsonl", None
            self.assertEqual(ps.sens_bound(m, "drop_nat(G)", r), "")
        finally:
            ps.SENS["bounds"] = old


if __name__ == "__main__":
    unittest.main(verbosity=2)
