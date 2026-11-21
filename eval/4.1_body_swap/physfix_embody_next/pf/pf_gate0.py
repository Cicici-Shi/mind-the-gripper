"""physfix post-hoc slice gates (protocol_physfix.md rev3 7.1 / 7.2 / 9.1; 2026-11-08). BLIND: never prints or writes a
success value, a success rate, an episode length or any per-key step count.

Gate implementations are the FROZEN prereg_models1.py ones (sha256 29faca6a..., verified before import), called on each
slice separately: ModelData.fixcheck / copy_gate / align_gate / twin_gates / contact_gate, with pf cells mapped to the
models1 codes S, T0, G (G''), GTfc (GTfc''): fixcheck vs the published OFT SRC of the same key (ref_body_pos +
object-suite free-joint first-frame qpos, <= 0.00005 mm, init id equal), copy_grip_cmd 1, first-frame alignment S / T0
<= 0.01 mm / 0.05 deg (PandaR85 print only), IK query frames > 1 mm < 1 %, stale 0, GTfc camera <= 0.01 mm / 0.01 deg,
contact log present. Added here: every slice (ep 0-4, 5-9, 10-14, 15-19) of every cell, not only slice 0 (rev3 9.1:
a100 LIBERO-Plus reset() retry can shift fixtures); align_json fixture_mm (fixture correction size, printed only); detector D (imported
physics_breakdown_scan.detect, thresholds unchanged) and BADQACC (npz pf_qacc_count) per slice; pf_blind == 1; GPU
model per cell. T0 vs S same-key outcome agreement (7.2, >= 0.90 WARN only) on slice 0 only: success is used inside
this process for the agreement rate and discarded; only the rate and the key count are printed.
Usage: python pf_gate0.py --out gate0_physfix.json
"""
import argparse, glob, hashlib, json, os, re, sys
from collections import Counter
import numpy as np

H = os.path.expanduser
ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--zoo", default=H("$PROJ_ROOT/vla_zoo"))
ap.add_argument("--pub", default=H("$PROJ_ROOT/runs"))
a = ap.parse_args()
AN = os.path.join(a.zoo, "analysis")
FROZEN_SHA = "29faca6ae9d63233b21378b62b61f40655a052cab924bd0c20e451f1e1c346e5"
assert hashlib.sha256(open(os.path.join(AN, "prereg_models1.py"), "rb").read()).hexdigest() == FROZEN_SHA
sys.dont_write_bytecode = True
sys.path.insert(0, AN)
sys.path.insert(0, os.path.join(a.zoo, "tools"))
import prereg_models1 as pm  # noqa: E402
import physics_breakdown_scan as pbs  # noqa: E402

GROUPS = {"OFT/object": ("pf_oftc_obj", "object"), "OFTi/object": ("pf_ofti_obj", "object"),
          "OFT/spatial": ("pf_oftc_sp", "spatial"), "OFTi/spatial": ("pf_ofti_sp", "spatial")}
CELLS = {"object": [("S", "pf_obj_sham_cg", 1, None), ("T0", "pf_obj_sham_twin", 1, "sync"),
                    ("G", "pf_obj_grpkp20", 1, None), ("GTfc", "pf_obj_g_tfc_kp20", 1, "sync")],
         "spatial": [("S", "pf_sham_cg", 1, None), ("T0", "pf_sham_twin", 1, "sync")]}
NAME = {"S": "S", "T0": "T0", "G": "G''", "GTfc": "GTfc''"}
SLICES = [(0, 5), (5, 10), (10, 15), (15, 20)]


class View:  # duck-typed stand-in for prereg_models1.ModelData (only the attributes the gate methods use)
    def __init__(self, suite, cells, data):
        self.kind, self.suite, self.spec, self.data = "npz", suite, {c[0]: c for c in cells}, data


def pf_fields(path):
    """Per key: pf_qacc_count, pf_blind, D (A / B / loose) -- no outcome, no length."""
    out = {}
    for f in sorted(glob.glob(os.path.join(path, "t*_e*_i*.npz"))):
        m = re.search(r"t(\d+)_e(\d+)_i(\d+)\.npz$", f)
        k = (int(m.group(1)), int(m.group(2)))
        z = np.load(f, allow_pickle=True)
        q = np.asarray(z["qpos"], float)
        names = [str(n) for n in z["joint_names"]]
        ai = [i for i, n in enumerate(names) if re.match(r"robot0_joint\d$", n)]
        fl = [i for i, n in enumerate(names) if n == "gripper0_finger_joint"]
        A, B, lo = pbs.detect(q, ai, fl[0] if fl else None, -0.5, 1.2)
        al = json.loads(str(z["align_json"])) if "align_json" in z.files else {}
        out[k] = {"qacc": int(z["pf_qacc_count"]) if "pf_qacc_count" in z.files else None,
                  "blind": int(z["pf_blind"]) if "pf_blind" in z.files else None,
                  "D": bool(A or B), "D_arm": bool(A), "D_finger": bool(B), "D_loose": bool(lo),
                  "fixture_mm": al.get("fixture_mm")}
    return out


rep = {"script_sha256": hashlib.sha256(open(os.path.abspath(__file__), "rb").read()).hexdigest(),
       "frozen_prereg_models1": FROZEN_SHA, "pbs_sha256": hashlib.sha256(open(pbs.__file__, "rb").read()).hexdigest(),
       "groups": {}}
src_cache = {}
allpass = True
for g, (mdir, suite) in GROUPS.items():
    sd = "obj_src" if suite == "object" else "src"
    if sd not in src_cache:
        src_cache[sd] = pm.load_npz_cell(os.path.join(a.pub, sd), 20, suite == "object")[0]
    src = src_cache[sd]
    cells = CELLS[suite]
    raw, pff = {}, {}
    for code, d, _, _ in cells:
        p = os.path.join(a.zoo, "runs", mdir, d)
        raw[code] = pm.load_npz_cell(p, 20, suite == "object")[0]
        pff[code] = pf_fields(p)
    G = {"dir": mdir, "suite": suite, "cells": {}}
    for code, d, _, _ in cells:
        C = {"dir": os.path.join(a.zoo, "runs", mdir, d), "keys": len(raw[code]), "slices": {}}
        gpus = Counter(r.get("gpu") for r in raw[code].values())
        C["gpu"] = dict(gpus)
        C["gpu_single"] = len(gpus) == 1 and None not in gpus
        for lo, hi in SLICES:
            sub = {k: r for k, r in raw[code].items() if lo <= k[1] < hi}
            v = View(suite, cells, {code: sub})
            S = {"n_keys": len(sub), "complete": len(sub) == 10 * (hi - lo)}
            gate = {"fixcheck": pm.ModelData.fixcheck(v, code, src), "copy_grip_cmd": pm.ModelData.copy_gate(v, code),
                    "first_frame_align": pm.ModelData.align_gate(v, code), "contacts": pm.ModelData.contact_gate(v, code)}
            gate.update(pm.ModelData.twin_gates(v, code))
            pf = {k: x for k, x in pff[code].items() if lo <= k[1] < hi}
            fx = [x["fixture_mm"] for x in pf.values()]
            nfx = sum(1 for x in fx if x is None)
            bad_fx = sum(1 for x in fx if x is not None and float(x) != 0.0)
            fv = [float(x) for x in fx if x is not None]
            # align_json fixture_mm = size of the static-fixture correction BodyEnv applies to match the reference env
            # (apply_static_body_pose), not a residual: descriptive only (2026-11-08 run <job> had wrongly gated it == 0)
            gate["align_fixture_correction_mm"] = ("NA", f"nonzero {bad_fx}, missing {nfx}, median {np.median(fv) if fv else None}, max {max(fv) if fv else None}")
            dk = sorted(k for k, x in pf.items() if x["D"])
            qk = sorted(k for k, x in pf.items() if (x["qacc"] or 0) > 0)
            qmiss = sum(1 for x in pf.values() if x["qacc"] is None)
            gate["D_0"] = ("PASS" if not dk else "FAIL", f"D keys {len(dk)} {dk} (arm {sum(x['D_arm'] for x in pf.values())}, finger {sum(x['D_finger'] for x in pf.values())}); loose {sum(x['D_loose'] for x in pf.values())}")
            gate["QACC_0"] = ("PASS" if not qk and not qmiss else ("UNK" if not qk else "FAIL"), f"QACC keys {len(qk)} {qk}, unreadable {qmiss}")
            nb = sum(1 for x in pf.values() if x["blind"] != 1)
            gate["pf_blind"] = ("PASS" if nb == 0 else "FAIL", f"pf_blind != 1: {nb}")
            S["gates"] = {k: {"status": s, "detail": det} for k, (s, det) in gate.items()}
            S["status"] = "FAIL" if any(s == "FAIL" for s, _ in gate.values()) else (
                "UNK" if any(s == "UNK" for s, _ in gate.values()) else "PASS")
            allpass &= S["status"] != "FAIL"
            C["slices"][f"e{lo}-{hi - 1}"] = S
        G["cells"][NAME[code]] = C
    # T0 vs S same-key outcome agreement, slice 0 only (rate printed, never the success values)
    ks = sorted(k for k in set(raw["T0"]) & set(raw["S"]) if k[1] < 5)
    agr = sum(raw["T0"][k]["succ"] == raw["S"][k]["succ"] for k in ks) / max(len(ks), 1)
    G["T0_vs_S_agreement_slice0"] = {"keys": len(ks), "rate": round(agr, 4), "status": "PASS" if agr >= 0.90 else "WARN"}
    for code in raw:
        for r in raw[code].values():
            r.pop("succ", None)
    rep["groups"][g] = G
rep["all_slices_no_FAIL"] = bool(allpass)
json.dump(rep, open(a.out, "w"), indent=1, default=str)
for g, G in rep["groups"].items():
    for c, C in G["cells"].items():
        for s, S in C["slices"].items():
            bad = {k: v["detail"] for k, v in S["gates"].items() if v["status"] not in ("PASS", "NA")}
            print(f"{g:13s} {c:7s} {s:7s} n {S['n_keys']:3d} {S['status']:4s} " + (json.dumps(bad, ensure_ascii=False) if bad else ""))
        print(f"{g:13s} {c:7s} gpu {C['gpu']}")
    print(f"{g:13s} T0 vs S slice-0 agreement {G['T0_vs_S_agreement_slice0']}")
print("ALL_SLICES_NO_FAIL", rep["all_slices_no_FAIL"])
