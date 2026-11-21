"""Synthetic-data tests for prereg_models1.py (no vla_zoo data is read).

Run:  PYTHONDONTWRITEBYTECODE=1 python test_prereg_models1.py        (unittest; ~ a few minutes on 4 CPUs)
Templates: one published OFT npz per suite ($PROJ_ROOT/runs/sham_cg/t0_e0_i42.npz, $PROJ_ROOT/runs/obj_s/t0_e0_i42.npz) and one
published pi0.5 jsonl record ($PROJ_ROOT/pi05/v2/grpkp100_tw_e0.jsonl, line 1), read only; the success field and the
models1 fields are rewritten, time series truncated to 3 rows. Override with MODELS1_PUB_RUNS / MODELS1_PI05.

Outcome coupling: per (model, suite) one fixed permutation u of the 200 (and 200 dense) keys; cell X succeeds on key k
iff u_k < SR(X). Cells are therefore comonotone, success counts are exact and CIs are narrow, so each scenario's
SR table maps to the intended branch.

Scenarios (end to end through main()):
  A  every line in its 复现 / 符合 branch; d5 not triggered; F reading = strong sentence; E5 整体小于
  B  every line in its 不复现 / 不符 branch; OFTi spatial d5 triggered (rec' rows); R3 vs recovery_GT differ line; F reading OFT
  C  R2 R3 F1 F2 O1 O2 E3 E4 不确定 (R1 不确定 is unreachable at n = 200, covered by unit tests)
  D  slice rule (3 slices -> 未完成, 2 slices -> 不印, R2 7/8 slices -> 未完成), error records dropped, duplicate clean
     record (first kept), copy_grip_cmd = 0 -> does not count, replan != 5 -> does not count, E1 < 95% -> E3/E4 只印不读,
     E5 相当或更大, existing --out refused
Unit tests: every pure decision rule in all three branches.
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
import prereg_models1 as pm  # noqa: E402

H = os.path.expanduser
PUB_RUNS = H(os.environ.get("MODELS1_PUB_RUNS", "$PROJ_ROOT/runs"))
PI05 = H(os.environ.get("MODELS1_PI05", "$PROJ_ROOT/pi05/v2"))
B_TEST = 10000

# ----------------------------------------------------------------------------------------------- templates
_TPL = {}


def npz_template(suite):
    if suite in _TPL:
        return _TPL[suite]
    f = os.path.join(PUB_RUNS, "sham_cg" if suite == "spatial" else "obj_s", "t0_e0_i42.npz")
    z = np.load(f, allow_pickle=True)
    d = {}
    for k in z.files:
        v = z[k]
        if k in ("action", "eef_pos", "eef_quat", "gripper_qpos", "qpos"):
            v = v[:3]
        if k.startswith(("contact_", "twin_", "host", "gpu", "align_json")):
            continue
        d[k] = v
    _TPL[suite] = d
    return d


def jsonl_template():
    if "jsonl" in _TPL:
        return _TPL["jsonl"]
    with open(os.path.join(PI05, "grpkp100_tw_e0.jsonl")) as fh:
        r = json.loads(fh.readline())
    _TPL["jsonl"] = r
    return r


def init_of(t, e):
    return int(np.random.RandomState(5478 + t).permutation(50)[e])


def fix_from_tpl(suite):
    z = npz_template(suite)
    names = [str(x) for x in z["ref_body_names"]]
    pos = np.asarray(z["ref_body_pos"], float)
    return {n: [round(float(x), 5) for x in pos[i]] for i, n in enumerate(names)
            if n != "world" and not n.startswith(("robot0", "gripper0", "mount0"))}


def objq_from_tpl(suite):
    z = npz_template(suite)
    names = [str(x) for x in z["joint_names"]]
    ad, tot = pm.qpos_layout(names)
    q0 = np.asarray(z["qpos"][0], float)
    return {n: [round(float(x), 5) for x in q0[a:a + s]] for n, (a, s) in ad.items() if not n.startswith(("robot0", "gripper0"))}


# ----------------------------------------------------------------------------------------------- outcome generator
def outcomes(sr_table, seed, eps=20, ep0=0):
    """{cell: {key: 0/1}} comonotone, exact counts per 200-key block."""
    keys = [(t, e) for t in range(10) for e in range(ep0, ep0 + eps)]
    perm = np.random.RandomState(seed).permutation(len(keys))
    u = {k: (perm[i] + 0.5) / len(keys) for i, k in enumerate(keys)}
    return {c: {k: int(u[k] < sr) for k in keys} for c, sr in sr_table.items() if sr is not None}


def twin_fields(kind, n=17):
    tw = {"twin_pos_err_mm": np.full(n, 0.01, np.float32), "twin_rot_err_deg": np.full(n, 0.02, np.float32),
          "twin_posture_deg": np.zeros(0, np.float32), "twin_stale": 0, "twin_t0": np.zeros(4, np.float32)}
    cam = 6.4 if kind == "move" else 0.0
    tw["twin_cam_err_mm"] = np.full(n, cam, np.float32)
    tw["twin_cam_err_deg"] = np.zeros(n, np.float32)
    return tw


def write_npz_cell(path, suite, outc, twin=None, copy_override=None, published=False, gpu="NVIDIA A100-PCIE-40GB"):
    os.makedirs(path, exist_ok=True)
    tpl = npz_template(suite)
    for (t, e), s in outc.items():
        init = init_of(t, e)
        d = dict(tpl)
        d.update(success=bool(s), task=t, ep=e, init=init)
        if not published:
            cg = (copy_override or {}).get((t, e), 1)
            al = None if twin == "src" else {"pos_mm": 0.0016, "rot_deg": 0.0013, "fixture_mm": 14.739, "missing": [],
                                              "quat_sign": 1.0, "copy_grip_cmd": cg, "forearm_geoms_off": None}
            d.update(host="node21", gpu=gpu, align_json=json.dumps(al), contact_arm_env_contact_steps=0,
                     contact_first_contact=-1, contact_stall_steps=0, contact_pairs="{}", contact_n_steps=3,
                     contact_first_grip_obj=1)
            if twin not in (None, "src"):
                d.update(twin_fields(twin))
        np.savez(os.path.join(path, f"t{t}_e{e}_i{init}.npz"), **d)


def jsonl_rec(suite, t, e, s, cell_twin=None, replan=5, a0=None, published=False, err=None, copy=1):
    tpl = jsonl_template()
    r = {k: v for k, v in tpl.items()}
    r.update(task=t, episode=e, init=init_of(t, e), success=None if s is None else bool(s), suite=suite, error=err,
             fix=fix_from_tpl(suite), objq=objq_from_tpl(suite) if suite == "object" else None,
             contacts={"arm_env_contact_steps": 0, "first_contact": -1, "stall_steps": 0, "pairs": {}, "n_steps": 3, "first_grip_obj": 1})
    if cell_twin == "src":
        r["align"] = None
    else:
        r["align"] = {"pos_mm": 0.0038, "rot_deg": 0.004, "fixture_mm": 14.739, "missing": [], "quat_sign": 1.0, "copy_grip_cmd": copy}
    if cell_twin in (None, "src"):
        r["twin"] = None
    else:
        r["twin"] = {"pos_err_mm_max": 0.01, "rot_err_deg_max": 0.05, "n_paint": 15, "miss": 0, "stale": 0, "over_1mm": 0,
                     "pos_err_mm_q": [0.01] * 3}
        cam = 6.4 if cell_twin == "move" else 0.0
        r["twin"].update(cam_err_mm_max=cam, cam_err_deg_max=0.0, cam_err_mm_mean=cam, cam_err_mm_q=[cam] * 3, cam_err_deg_q=[0.0] * 3)
    if not published:
        r.update(replan=replan, chunk_len=50 if replan == 5 else 1, a0=a0 if a0 is not None else [[0.1 * t, 0.01 * e, 0, 0, 0, 0, -1]] * replan,
                 server_host="node37", server_gpu="NVIDIA H100 80GB HBM3", host="node12", gpu="Tesla T4", rtt_ms_med=90.0)
    else:
        for k in ("replan", "chunk_len", "a0", "server_host", "server_gpu"):
            r.pop(k, None)
        r.update(host="node41", gpu="NVIDIA H200 NVL")
    return r


def write_jsonl_cell(root, name, suite, outc, twin=None, replan=5, published=False, extra=None, a0_map=None, replan_override=None):
    """One file per 5-episode slice (<name>_e<off>.jsonl); extra: {off: [records appended]}"""
    os.makedirs(root, exist_ok=True)
    by = {}
    for (t, e), s in outc.items():
        rp = (replan_override or {}).get((t, e), replan)
        by.setdefault(e // 5 * 5, []).append(jsonl_rec(suite, t, e, s, twin, rp, (a0_map or {}).get((t, e)), published))
    for off, recs in (extra or {}).items():
        by.setdefault(off, [])
        by[off] = recs[0] + by[off] + recs[1]  # (prepend, append)
    for off, recs in by.items():
        with open(os.path.join(root, f"{name}_e{off}.jsonl"), "w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")


# ----------------------------------------------------------------------------------------------- published (fixed)
PUB_SR = {
    ("OFT", "spatial"): {"SRC": .98, "S": .95, "T0": .95, "G": .50, "G_T": .86, "G_Tw": .825, "G_Ta": .715},
    ("pi05", "spatial"): {"SRC": .97, "S": .90, "T0": .865, "G": .73, "G_T": .81, "G_Tw": .735, "G_Ta": .83},
    ("OFT", "object"): {"SRC": .98, "S": .96, "T0": .955, "G": .80, "G_T": .74},
    ("pi05", "object"): {"SRC": .97, "S": .95, "T0": .975, "G": .78, "G_T": .84},
}


def build_published(root):
    pr, p5 = os.path.join(root, "pub_runs"), os.path.join(root, "pi05")
    for (mdl, suite), tab in PUB_SR.items():
        cells = dict(pm.PUB_SP if suite == "spatial" else pm.PUB_OBJ)
        oc = outcomes(tab, seed={("OFT", "spatial"): 1, ("pi05", "spatial"): 2, ("OFT", "object"): 3, ("pi05", "object"): 4}[(mdl, suite)])
        for c, d in cells.items():
            if mdl == "OFT":
                write_npz_cell(os.path.join(pr, d), suite, oc[c], published=True)
            else:
                write_jsonl_cell(p5, d, suite, oc[c], twin=None if c in ("SRC", "S", "G") else "move", published=True)
    return pr, p5


CELL_TWIN = {"SRC": "src", "S": None, "T0": "sync", "G": None, "GTfc": "sync", "GTw": "sync", "GTa": "agent", "G_T": "move"}


def build_new(root, spec, mods=None):
    """spec: {(model, suite): {cell: SR}}; mods: per-cell tweaks for scenario D."""
    mods = mods or {}
    zr = os.path.join(root, "zoo", "runs")
    for (mdl, suite), tab in spec.items():
        seed = {"OFTi": 11, "pi0": 22, "OVLA": 33}[mdl] + (0 if suite == "spatial" else 100)
        oc = outcomes(tab, seed)
        oc2 = outcomes({c: tab[c] for c in pm.DENSE if c in tab}, seed + 7, eps=20, ep0=20) if (mdl, suite) == ("OFTi", "spatial") else {}
        for c, d, _, _ in pm.NEW_CELLS:
            if c not in oc:
                continue
            md = mods.get((mdl, suite, c), {})
            o = dict(oc[c])
            if c in oc2:
                o.update(oc2[c])
            if "flip_from" in md:                       # E1 disagreement: copy another cell, flip keys pairwise
                o = dict(oc[md["flip_from"]])
                on = [k for k in sorted(o) if o[k] == 1][:md["nflip"]]
                off = [k for k in sorted(o) if o[k] == 0][:md["nflip"]]
                for k in on:
                    o[k] = 0
                for k in off:
                    o[k] = 1
            if "keep_eps" in md:
                o = {k: v for k, v in o.items() if k[1] < md["keep_eps"]}
            name = ("obj_" if suite == "object" else "") + d
            if mdl == "OFTi":
                write_npz_cell(os.path.join(zr, "oft_sp" if suite == "spatial" else "oft_obj", name), suite, o, twin=CELL_TWIN[c],
                               copy_override=md.get("copy0"))
            else:
                a0_map = None
                if "flip_from" in md:
                    a0_map = {k: [[9.0] * 7] for k in on + off}
                write_jsonl_cell(os.path.join(zr, "pi0" if mdl == "pi0" else "openvla"), name, suite, o, twin=CELL_TWIN[c],
                                 replan=5 if mdl == "pi0" else 1, extra=md.get("extra"), a0_map=a0_map,
                                 replan_override=md.get("replan"))
    return zr


def run_main(root, zr, pr, p5, tag, extra_args=()):
    out = os.path.join(root, f"res_{tag}.txt")
    vj = os.path.join(root, f"v_{tag}.json")
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        rc = pm.main(["--out", out, "--verdict-json", vj, "--B", str(B_TEST), "--zoo-runs", zr, "--pub-runs", pr, "--pi05", p5,
                      "--zoo-logs", os.path.join(root, "nologs"), "--protocol", os.path.join(root, "proto.md"),
                      "--decisions", os.path.join(root, "nodec.md"), "--zoo-cells", os.path.join(root, "nocells.sh"), *extra_args])
    txt, v = "", None
    if os.path.exists(out):
        with open(out) as fh:
            txt = fh.read()
    if os.path.exists(vj):
        with open(vj) as fh:
            v = json.load(fh)
    return rc, v, txt


# ----------------------------------------------------------------------------------------------- scenario SR tables
SC_A = {
    ("OFTi", "spatial"): {"SRC": .98, "S": .96, "T0": .95, "G": .52, "GTfc": .81, "GTw": .82, "GTa": .70, "G_T": .83},
    ("pi0", "spatial"): {"SRC": .96, "S": .92, "T0": .90, "G": .74, "GTfc": .84, "GTw": .70, "GTa": .80, "G_T": .84},
    ("OFTi", "object"): {"SRC": .98, "S": .96, "T0": .95, "G": .78, "GTfc": .73, "GTw": .75, "GTa": .76, "G_T": .73},
    ("pi0", "object"): {"SRC": .95, "S": .94, "T0": .93, "G": .77, "GTfc": .79, "GTw": .78, "GTa": .78, "G_T": .79},
    ("OVLA", "spatial"): {"SRC": .84, "S": .80, "T0": .79, "G": .60, "GTfc": .70, "GTw": .60, "GTa": .70},
}
SC_B = {
    ("OFTi", "spatial"): {"SRC": .98, "S": .96, "T0": .85, "G": .81, "GTfc": .62, "GTw": .80, "GTa": .90, "G_T": 1.0},
    ("pi0", "spatial"): {"SRC": .96, "S": .80, "T0": .78, "G": .90, "GTfc": .78, "GTw": .85, "GTa": .70, "G_T": .78},
    ("OFTi", "object"): {"SRC": .98, "S": .96, "T0": .95, "G": .70, "GTfc": .89, "GTw": .80, "GTa": .80, "G_T": .89},
    ("pi0", "object"): {"SRC": .95, "S": .80, "T0": .79, "G": .90, "GTfc": .79, "GTw": .80, "GTa": .80, "G_T": .79},
    ("OVLA", "spatial"): {"SRC": .84, "S": .80, "T0": .80, "G": .81, "GTfc": .70, "GTw": .81, "GTa": .70},
}
SC_C = {
    ("OFTi", "spatial"): {"SRC": .98, "S": .96, "T0": .95, "G": .52, "GTfc": .66, "GTw": .80, "GTa": .80, "G_T": .88},
    ("pi0", "spatial"): {"SRC": .96, "S": .92, "T0": .90, "G": .74, "GTfc": .64, "GTw": .75, "GTa": .75, "G_T": .80},
    ("OFTi", "object"): {"SRC": .98, "S": .96, "T0": .95, "G": .76, "GTfc": .83, "GTw": .80, "GTa": .80, "G_T": .83},
    ("pi0", "object"): {"SRC": .95, "S": .94, "T0": .93, "G": .77, "GTfc": .69, "GTw": .78, "GTa": .78, "G_T": .69},
    ("OVLA", "spatial"): {"SRC": .84, "S": .80, "T0": .79, "G": .73, "GTfc": .73, "GTw": .73, "GTa": .73},
}
SC_D = {k: dict(v) for k, v in SC_A.items()}
SC_D[("OVLA", "spatial")] = {"SRC": .84, "S": .80, "T0": .79, "G": .30, "GTfc": .40, "GTw": .30, "GTa": .40}
ERR = "ConnectionClosedError()"
MODS_D = {
    ("OFTi", "spatial", "S"): {"keep_eps": 15},                     # 3 slices -> 未完成
    ("OFTi", "spatial", "GTw"): {"keep_eps": 35},                   # 7/8 dense slices -> R2 未完成
    ("pi0", "spatial", "G"): {"keep_eps": 10},                      # 2 slices -> 不印
    ("OFTi", "object", "GTfc"): {"copy0": {(3, 7): 0}},             # copy_grip_cmd 0 -> does not count
    ("pi0", "object", "G"): {"replan": {(4, 4): 8}},                # replan != 5 -> does not count
    ("OVLA", "spatial", "GTw"): {"flip_from": "G", "nflip": 10},    # agree 90% -> E3/E4 只印不读
}


def d_extra():
    """pi0 spatial S slice e0: error record before the clean one for (0,0); (1,1) only an error record (its clean
    record is removed in the test); a second clean record for (2,2) with flipped success after the first."""
    return {0: ([jsonl_rec("spatial", 0, 0, 1, None, err=ERR)],
                [jsonl_rec("spatial", 1, 1, 0, None, err=ERR), jsonl_rec("spatial", 2, 2, None, None)])}


class Scenarios(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="models1_test_")
        cls.pr, cls.p5 = build_published(cls.tmp)
        cls.res = {}
        for tag, sc, mods in (("A", SC_A, None), ("B", SC_B, None), ("C", SC_C, None), ("D", SC_D, MODS_D)):
            root = os.path.join(cls.tmp, tag)
            os.makedirs(root)
            m = dict(mods or {})
            if tag == "D":
                m[("pi0", "spatial", "S")] = {"extra": d_extra()}
            zr = build_new(root, sc, m)
            if tag == "D":   # (2,2) duplicate: second clean record must have the opposite outcome; drop clean (1,1)
                f = os.path.join(zr, "pi0", "sham_cg_e0.jsonl")
                with open(f) as fh:
                    lines = [json.loads(x) for x in fh]
                first = next(r for r in lines if (r["task"], r["episode"]) == (2, 2) and not r["error"] and r["success"] is not None)
                for r in lines:
                    if (r["task"], r["episode"]) == (2, 2) and r["success"] is None:
                        r["success"] = not first["success"]
                lines = [r for r in lines if not ((r["task"], r["episode"]) == (1, 1) and not r["error"])]
                with open(f, "w") as fh:
                    for r in lines:
                        fh.write(json.dumps(r) + "\n")
                cls.d_first_22 = float(bool(first["success"]))
                cls.d_zr = zr
            cls.res[tag] = run_main(root, zr, cls.pr, cls.p5, tag)

    @classmethod
    def tearDownClass(cls):
        if not os.environ.get("MODELS1_KEEP"):
            shutil.rmtree(cls.tmp, ignore_errors=True)
        else:
            print("kept", cls.tmp)

    def v(self, tag):
        rc, v, txt = self.res[tag]
        self.assertEqual(rc, 0)
        return v, txt

    # ---------------- published synthetic recompute and protocol comparison
    def test_published_mismatch_warning(self):
        v, txt = self.v("A")
        self.assertIn("协议文字比对", txt)
        # a protocol number that the data do not give -> warning (patch the constant, published-only run)
        old = pm.PROTO_PUB[("OFT", "spatial")]["drop_nat(G)"]
        pm.PROTO_PUB[("OFT", "spatial")]["drop_nat(G)"] = (-30, -37, -23)
        try:
            root = os.path.join(self.tmp, "A")
            rc, v2, txt2 = run_main(root, os.path.join(root, "zoo", "runs"), self.pr, self.p5, "pubonly", ["--published-only"])
        finally:
            pm.PROTO_PUB[("OFT", "spatial")]["drop_nat(G)"] = old
        self.assertEqual(rc, 0)
        self.assertIn("!!! 警告：与协议不一致", txt2)
        self.assertTrue(any("OFT/spatial/drop_nat(G)" in x for x in v2["proto_mismatch"]))
        self.assertNotIn("门槛（", txt2)   # published-only reads no vla_zoo data
        self.assertAlmostEqual(v["published"]["OFT/spatial/drop_nat(G)"][0], -45.0)
        self.assertAlmostEqual(v["published"]["OFT/spatial/recovery(G)"][0], 36.0)
        self.assertAlmostEqual(v["published"]["pi05/spatial/channel"][0], -9.5)
        self.assertIn("MISSING", txt)  # sha256 of absent files

    # ---------------- A: 复现 / 符合
    def test_A(self):
        v, txt = self.v("A")
        self.assertEqual(v["R1"], "复现")
        self.assertEqual(v["R2"], "复现")
        self.assertEqual(v["R3"], "复现")
        self.assertEqual(v["recGT_P7"], "复现")
        self.assertEqual(v["R_family"], "复现")
        self.assertIn(pm.TXT_R_YES, txt)
        self.assertFalse(v["R3_vs_GT_differ"])
        self.assertEqual(v["F1"], "像 π0.5")
        self.assertEqual(v["F2"], "像 π0.5")
        self.assertEqual(v["F_reading"], "strong")
        self.assertIn(pm.TXT_F_STRONG, txt)
        self.assertEqual(v["O1"], "复现")
        self.assertIn(pm.TXT_O1_YES, txt)
        self.assertEqual(v["O2"], "复现")
        self.assertEqual(v["E1"], 100.0)
        self.assertEqual(v["E2"], 100.0)
        self.assertTrue(v["E_read"])
        self.assertEqual(v["E3"], "符合")
        self.assertEqual(v["E4"], "符合")
        self.assertEqual(v["E5"], "整体小于")
        self.assertIn(pm.TXT_E5_SMALL, txt)
        self.assertIn(pm.TXT_E5_CAVEAT, txt)
        self.assertFalse(v["d5_trigger"]["OFTi/spatial"])
        self.assertNotIn("rec'_w", txt)
        self.assertIn("首块动作 a0 逐位相同 200/200", txt)
        for k, c in v["cells"].items():
            self.assertEqual(c["status"], "OK", (k, c))
        self.assertIn("Holm p", txt)
        self.assertIn("recovery_GT(G) − recovery(G)", txt)

    # ---------------- B: 不复现 / 不符 + d5
    def test_B(self):
        v, txt = self.v("B")
        self.assertEqual(v["R1"], "不复现")
        self.assertEqual(v["R2"], "不复现")
        self.assertEqual(v["R3"], "不复现")
        self.assertEqual(v["R_family"], "不复现")
        self.assertIn(pm.TXT_R_NO, txt)
        self.assertEqual(v["recGT_P7"], "复现")
        self.assertTrue(v["R3_vs_GT_differ"])
        self.assertIn(pm.TXT_R_CAM, txt)
        self.assertEqual(v["F1"], "像 OFT")
        self.assertEqual(v["F2"], "像 OFT")
        self.assertEqual(v["F_reading"], "oft")
        self.assertIn(pm.TXT_F_OFT, txt)
        self.assertEqual(v["O1"], "不复现")
        self.assertIn(pm.TXT_O1_NO, txt)
        self.assertEqual(v["O2"], "不复现")
        self.assertEqual(v["E3"], "不符")
        self.assertEqual(v["E4"], "不符")
        # d5 branch
        self.assertTrue(v["d5_trigger"]["OFTi/spatial"])
        self.assertIn("rec'_w", txt)
        self.assertIn("替身管线代价 d5 =", txt)
        self.assertIn("（与 recovery(G) 恒等）", txt)

    # ---------------- C: 不确定
    def test_C(self):
        v, txt = self.v("C")
        self.assertEqual(v["R1"], "复现")
        self.assertEqual(v["R2"], "不确定")
        self.assertEqual(v["R3"], "不确定")
        self.assertEqual(v["R_family"], "不确定")
        self.assertEqual(v["F1"], "不确定")
        self.assertEqual(v["F2"], "不确定")
        self.assertIsNone(v["F_reading"])
        self.assertEqual(v["O1"], "不确定")
        self.assertEqual(v["O2"], "不确定")
        self.assertEqual(v["E3"], "不确定")
        self.assertEqual(v["E4"], "不确定")

    # ---------------- D: slices, errors, copy_grip_cmd, replan, E1 < 95%
    def test_D(self):
        v, txt = self.v("D")
        c = v["cells"]
        self.assertEqual(c["OFTi/spatial/S"]["slices"], [0, 1, 2])
        self.assertEqual(v["printed"]["OFTi/spatial/drop_nat(G)"], "未完成")
        self.assertEqual(v["R1"], "未完成")
        self.assertEqual(v["R3"], "未完成")
        self.assertEqual(c["OFTi/spatial/GTw"]["slices"], [0, 1, 2, 3, 4, 5, 6])
        self.assertEqual(v["printed"]["OFTi/spatial/channel@400"], "未完成")
        self.assertEqual(v["R2"], "未完成")
        self.assertEqual(v["R_family"], "未能判定")
        self.assertIn("【未完成：共同整片 3/4，只印不判】", txt)
        # < 3 slices: not printed
        self.assertEqual(c["pi0/spatial/G"]["slices"], [0, 1])
        self.assertEqual(v["printed"]["pi0/spatial/drop_nat(G)"], "不印")
        self.assertIn("不印（共同整片 2 < 3）", txt)
        self.assertEqual(v["F1"], "缺")
        self.assertEqual(v["F2"], "缺")
        # error records dropped; key with only error records breaks slice 0; duplicate clean record
        meta = c["pi0/spatial/S"]["meta"]
        self.assertEqual(meta["error_dropped"], 2)
        self.assertEqual(meta["error_only_keys"], 1)
        self.assertEqual(meta["dup_clean"], 1)
        self.assertEqual(c["pi0/spatial/S"]["slices"], [1, 2, 3])
        rec, _ = pm.load_jsonl_cell(os.path.join(self.d_zr, "pi0"), "sham_cg", 20, dup="first")
        self.assertEqual(rec[(2, 2)]["succ"], self.d_first_22)
        self.assertNotIn((1, 1), rec)
        self.assertIn((0, 0), rec)
        # copy_grip_cmd = 0 -> does not count
        self.assertEqual(c["OFTi/object/GTfc"]["gates"]["copy_grip_cmd"], "FAIL")
        self.assertEqual(c["OFTi/object/GTfc"]["status"], "FAIL")
        self.assertEqual(v["O1"], "does not count")
        self.assertIn("不判定（does not count：GTfc 门槛不过）", txt)
        # replan != 5 -> does not count
        self.assertEqual(c["pi0/object/G"]["gates"]["replan"], "FAIL")
        self.assertEqual(v["O2"], "does not count")
        # E1 < 95% -> E3/E4 只印不读
        self.assertEqual(v["E1"], 90.0)
        self.assertFalse(v["E_read"])
        self.assertTrue(v["E3"].startswith("只印不读:"))
        self.assertTrue(v["E4"].startswith("只印不读:"))
        self.assertIn("首块动作 a0 逐位相同 180/200", txt)
        self.assertEqual(v["E5"], "相当或更大")
        self.assertIn(pm.TXT_E5_BIG, txt)
        # single read-out: existing --out refused
        root = os.path.join(self.tmp, "D")
        with contextlib.redirect_stderr(io.StringIO()):
            rc, _, _ = run_main(root, self.d_zr, self.pr, self.p5, "D")
        self.assertEqual(rc, 3)


# ----------------------------------------------------------------------------------------------- unit tests
def R(pt, lo, hi):
    return {"point": pt, "ci95": (lo, hi)}


class Rules(unittest.TestCase):
    def test_r1(self):
        pub = R(-45, -52, -38)
        self.assertEqual(pm.r1_verdict(R(-44, -51, -37), pub), "复现")
        self.assertEqual(pm.r1_verdict(R(-15, -20, -10), pub), "不复现")
        self.assertEqual(pm.r1_verdict(R(-20, -40, -5), pub), "不确定")   # overlaps, upper >= -7 (needs CI width >= 31)

    def test_r2(self):
        self.assertEqual(pm.r2_verdict(R(12, 6, 18), R(30, 22, 38), -44), "复现")
        self.assertEqual(pm.r2_verdict(R(-8, -14, -2), R(30, 22, 38), -44), "不复现")
        self.assertEqual(pm.r2_verdict(R(5, -2, 12), R(5, -3, 10), -44), "不复现")   # rec_w upper < 11 (wrist ineffective)
        self.assertEqual(pm.r2_verdict(R(12, 6, 18), R(18, 10, 26), -44), "不确定")   # rec_w point < 22
        self.assertEqual(pm.r2_verdict(R(3, -3, 9), R(30, 22, 38), -44), "不确定")

    def test_r3(self):
        self.assertEqual(pm.r3_verdict(R(30, 22, 38), -44), "复现")
        self.assertEqual(pm.r3_verdict(R(2, -5, 9), -44), "不复现")
        self.assertEqual(pm.r3_verdict(R(15, 7, 23), -44), "不确定")

    def test_f(self):
        self.assertEqual(pm.f1_verdict(R(-10, -15, -4)), "像 π0.5")
        self.assertEqual(pm.f1_verdict(R(10, 4, 15)), "像 OFT")
        self.assertEqual(pm.f1_verdict(R(0, -6, 6)), "不确定")
        self.assertEqual(pm.f2_verdict("复现", "复现"), "像 π0.5")
        self.assertEqual(pm.f2_verdict("不复现", "不复现"), "像 OFT")
        self.assertEqual(pm.f2_verdict("复现", "不复现"), "不确定")
        self.assertEqual(pm.f2_verdict("不确定", "复现"), "不确定")

    def test_o(self):
        self.assertEqual(pm.o1_verdict(R(-4, -10, 2), -18), "复现")
        self.assertEqual(pm.o1_verdict(R(20, 12, 28), -26), "不复现")
        self.assertEqual(pm.o1_verdict(R(8, 1, 15), -20), "不确定")
        self.assertEqual(pm.o1_verdict(R(5, 1, 12), -26), "复现")   # upper 12 < 13
        self.assertEqual(pm.o2_verdict("复现", "复现"), "复现")
        self.assertEqual(pm.o2_verdict("不复现", "不复现"), "不复现")
        self.assertEqual(pm.o2_verdict("复现", "不确定"), "不确定")

    def test_p7(self):
        ref = R(-17, -22.5, -12)
        self.assertEqual(pm.p7_verdict(ref, R(-18, -24, -12)), "复现")
        self.assertEqual(pm.p7_verdict(ref, R(10, 5, 15)), "不复现")
        self.assertEqual(pm.p7_verdict(ref, R(-5, -9, -1)), "不确定")    # same sign, no overlap
        self.assertEqual(pm.p7_verdict(ref, R(3, -2, 8)), "不确定")      # opposite sign, CI has 0
        self.assertEqual(pm.p7_verdict(ref, R(0, -3, 3)), "不确定")      # D4: zero point has no sign

    def test_e(self):
        self.assertEqual(pm.e3_verdict(R(-20, -26, -14)), "符合")
        self.assertEqual(pm.e3_verdict(R(1, -1, 3)), "不符")
        self.assertEqual(pm.e3_verdict(R(-7, -10, -4)), "不确定")
        self.assertEqual(pm.e4_verdict(R(10, 6, 14)), "符合")
        self.assertEqual(pm.e4_verdict(R(-11, -15, -7)), "不符")
        self.assertEqual(pm.e4_verdict(R(0, -3, 3)), "不确定")
        self.assertEqual(pm.e5_verdict(R(-20, -26, -14), R(-44, -51, -37)), "整体小于")
        self.assertEqual(pm.e5_verdict(R(-40, -47, -33), R(-44, -51, -37)), "相当或更大")
        self.assertEqual(pm.abs_interval(R(1, -2, 5)), (0.0, 5))

    def test_d5_slices_family_holm(self):
        self.assertTrue(pm.d5_trigger(R(-11, -15, -7)))
        self.assertFalse(pm.d5_trigger(R(-3.5, -6.5, -1)))
        self.assertEqual(pm.slice_status(4, 4), "完成")
        self.assertEqual(pm.slice_status(3, 4), "未完成")
        self.assertEqual(pm.slice_status(2, 4), "不印")
        self.assertEqual(pm.slice_status(8, 8), "完成")
        self.assertEqual(pm.slice_status(7, 8), "未完成")
        self.assertEqual(pm.r_family({"R1": "复现", "R2": "复现", "R3": "复现"}), "复现")
        self.assertEqual(pm.r_family({"R1": "复现", "R2": "不复现", "R3": "未完成"}), "不复现")
        self.assertEqual(pm.r_family({"R1": "复现", "R2": "未完成", "R3": "复现"}), "未能判定")
        self.assertEqual(pm.r_family({"R1": "复现", "R2": "不确定", "R3": "复现"}), "不确定")
        adj = pm.holm({"a": 0.01, "b": 0.04, "c": 0.03})
        self.assertAlmostEqual(adj["a"], 0.03)
        self.assertAlmostEqual(adj["c"], 0.06)
        self.assertAlmostEqual(adj["b"], 0.06)


if __name__ == "__main__":
    unittest.main(verbosity=2)
