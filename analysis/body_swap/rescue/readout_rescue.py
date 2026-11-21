"""rescue pilot readout (protocol_rescue_pilot.md 78c1e1d3a20166ca..., frozen 2026-11-08), section 3. Run ONCE, after
all three cells x two checkpoints are complete (200 keys each). Not run by the execution agent.

Per checkpoint (合训 OFT = rs_oftc_sp, OFTi = rs_ofti_sp), cells S = pf_sham_cg, G2 = pf_grpkp20, GTfc2 = pf_g_tfc_kp20:
  drop_nat = SR(G2) - SR(S); recovery = SR(GTfc2) - SR(G2)  (percentage points)
  paired cluster bootstrap over (task, ep) keys, B = 10000, seed 0, percentile 95%: the Boot class is IMPORTED from the
  frozen $PROJ_ROOT/vla_zoo/analysis/prereg_models1.py (sha256 29faca6a..., import refused on mismatch), one draw matrix
  per checkpoint shared by all its cells (same as prereg_physfix).
Verdict (protocol 3, verbatim lines):
  值得做: drop point <= -15 and 95% upper < 0 (per checkpoint; that checkpoint enters the formal rescue cells)
  边缘:   -15 < drop point <= -7 and 95% upper < 0: enters ONLY if recovery 95% lower > 0, report says 「可救幅度小」
  不值得做: everything else. If no checkpoint enters, the fixed sentence of protocol 3 is printed.
Refuses to read (exit 3, prints no SR) unless every cell dir has exactly the 200 keys t0..9 x e0..19, one file per
key, pf_blind == 1 and pf_cell_env equal to the cells_next.sh switches. GPU type, QACC are printed (descriptive).
Usage: python readout_rescue.py [--runs $PROJ_ROOT/vla_zoo/runs] [--B 10000] --out <results.txt> [--json <verdict.json>]
       [--cells-env <json file with {cell: env}> (tests only; default = from cells_next.sh)]
"""
import argparse, glob, hashlib, json, os, re, subprocess, sys
import numpy as np

H = os.path.expanduser
FROZEN = H("$PROJ_ROOT/vla_zoo/analysis/prereg_models1.py")
FROZEN_SHA = "29faca6ae9d63233b21378b62b61f40655a052cab924bd0c20e451f1e1c346e5"
CKPTS = [("合训 OFT", "rs_oftc_sp"), ("OFTi", "rs_ofti_sp")]
CELLS = {"S": "pf_sham_cg", "G2": "pf_grpkp20", "GTfc2": "pf_g_tfc_kp20"}
KEYS = [(t, e) for t in range(10) for e in range(20)]
WORTH, EDGE = -15.0, -7.0
NOT_WORTH_TXT = "kp20 仿真配置下 Spatial 换 Robotiq 没有可测的掉分，仿真中轮廓救援没有对象"


def boot_cls():
    sha = hashlib.sha256(open(FROZEN, "rb").read()).hexdigest()
    if sha != FROZEN_SHA:
        raise SystemExit(f"prereg_models1.py sha256 {sha} != {FROZEN_SHA}; refusing to import")
    sys.dont_write_bytecode = True
    sys.path.insert(0, os.path.dirname(FROZEN))
    import prereg_models1 as pm  # noqa: E402
    return pm.Boot


def cells_env():
    out = {}
    for c in CELLS.values():
        v = subprocess.check_output(["bash", H("$PROJ_ROOT/vla_zoo/next/cells_next.sh"), "vars", c], text=True).split()
        d = {"BODY": v[0], "EMBODY_DIR": H("$PROJ_ROOT/embody_next"), "M1_FASTSYNC": "1"}
        d.update(dict(x.split("=", 1) for x in v[1:]))
        out[c] = d
    return out


def integrity(runs, want_env):
    """Return (problems, desc). Looks only at file names and non-outcome fields."""
    probs, desc = [], {}
    for lab, m in CKPTS:
        for nm, c in CELLS.items():
            d = os.path.join(runs, m, c)
            fs = sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz")))
            ks = [tuple(int(x) for x in re.search(r"t(\d+)_e(\d+)_i\d+\.npz$", f).groups()) for f in fs]
            if sorted(ks) != KEYS:
                probs.append(f"{m}/{c}: {len(ks)} files, unique {len(set(ks))}, missing {len(set(KEYS) - set(ks))}, extra {len(set(ks) - set(KEYS))}")
                continue
            gpus, qacc, envbad, blindbad = set(), 0, 0, 0
            for f in fs:
                z = np.load(f, allow_pickle=True)
                gpus.add(str(z["gpu"]) if "gpu" in z.files else "NA")
                qacc += int(z["pf_qacc_count"]) if "pf_qacc_count" in z.files else 0
                envbad += json.loads(str(z["pf_cell_env"])) != want_env[c]
                blindbad += int(z["pf_blind"]) != 1
            if envbad or blindbad:
                probs.append(f"{m}/{c}: cell_env mismatch {envbad}, pf_blind != 1 {blindbad}")
            desc[(m, c)] = {"gpu": sorted(gpus), "qacc_total": qacc}
    return probs, desc


def load_y(runs, m):
    Y = {}
    for nm, c in CELLS.items():
        y = np.full(len(KEYS), np.nan)
        for f in glob.glob(os.path.join(runs, m, c, "t*_e*_i*.npz")):
            t, e = (int(x) for x in re.search(r"t(\d+)_e(\d+)_i\d+\.npz$", f).groups())
            y[KEYS.index((t, e))] = float(bool(np.load(f, allow_pickle=True)["success"]))
        Y[nm] = y
    return Y


def verdict(drop, rec):
    """drop / rec: dicts with point, ci95 (lo, hi). Returns (tier, enters, note)."""
    # 200 binary keys: points are multiples of 0.5 pp; round away float noise (e.g. -15.000000000000002) so that the
    # protocol boundaries <= -15 / <= -7 / < 0 / > 0 are applied to the exact values.
    p, hi, rlo = round(drop["point"], 6), round(drop["ci95"][1], 6), round(rec["ci95"][0], 6)
    if p <= WORTH and hi < 0:
        return "值得做", True, ""
    if WORTH < p <= EDGE and hi < 0:
        if rlo > 0:
            return "边缘", True, "可救幅度小"
        return "边缘", False, "recovery 95% 下界不 > 0，不进入正式格"
    return "不值得做", False, ""


def fmt(r):
    return f"{r['point']:+.2f} [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] (n={r['n']})"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=H("$PROJ_ROOT/vla_zoo/runs"))
    ap.add_argument("--B", type=int, default=10000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--json")
    ap.add_argument("--cells-env")
    a = ap.parse_args(argv)
    for p in (a.out, a.json):
        if p and os.path.exists(p):
            print(f"refuse: {p} exists (one readout only)")
            return 2
    want = json.load(open(a.cells_env)) if a.cells_env else cells_env()
    probs, desc = integrity(a.runs, want)
    if probs:
        print("REFUSE (incomplete / inconsistent; no SR printed):")
        for x in probs:
            print("  " + x)
        return 3
    Boot = boot_cls()
    L, res = [], {}
    L.append(f"rescue pilot readout, protocol_rescue_pilot.md 78c1e1d3a20166ca, B={a.B} seed=0, runs={a.runs}")
    L.append(f"readout script sha256 {hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest()}")
    for lab, m in CKPTS:
        Y = load_y(a.runs, m)
        bt = Boot(KEYS, a.B, seed=0)
        sr = {nm: bt.est(Y, [(1, nm)]) for nm in CELLS}
        drop = bt.est(Y, [(1, "G2"), (-1, "S")])
        rec = bt.est(Y, [(1, "GTfc2"), (-1, "G2")])
        resid = bt.est(Y, [(1, "GTfc2"), (-1, "S")])
        tier, enters, note = verdict(drop, rec)
        L.append(f"\n==== {lab}（{m}，libero_spatial，200 键）")
        for nm, c in CELLS.items():
            L.append(f"  SR({nm} {c}) = {fmt(sr[nm])}   gpu={desc[(m, c)]['gpu']} qacc_total={desc[(m, c)]['qacc_total']}")
        L.append(f"  drop_nat = SR(G″) − SR(S)       = {fmt(drop)}")
        L.append(f"  recovery = SR(GTfc″) − SR(G″)   = {fmt(rec)}")
        L.append(f"  （描述，不判）SR(GTfc″) − SR(S) = {fmt(resid)}")
        L.append(f"  判定：{tier}；{'进入 rescue 正式格' if enters else '不进入'}{('；' + note) if note else ''}")
        res[lab] = {"model": m, "sr": {k: {"point": v["point"], "ci95": v["ci95"]} for k, v in sr.items()},
                    "drop_nat": {"point": drop["point"], "ci95": drop["ci95"]},
                    "recovery": {"point": rec["point"], "ci95": rec["ci95"]},
                    "resid_desc": {"point": resid["point"], "ci95": resid["ci95"]},
                    "tier": tier, "enters": enters, "note": note}
    ent = [k for k, v in res.items() if v["enters"]]
    L.append("\n==== 总判定")
    if ent:
        L.append("  进入 rescue 正式格的 checkpoint：" + "、".join(f"{k}（{res[k]['tier']}{'，可救幅度小' if res[k]['note'] == '可救幅度小' else ''}）" for k in ent))
    else:
        L.append("  不值得做：" + NOT_WORTH_TXT + "。不换物理、不换键、不换套件补救。")
    L.append("  kp100 旧数只作背景，不与本批合并。")
    txt = "\n".join(L)
    print(txt)
    with open(a.out, "w") as f:
        f.write(txt + "\n")
    if a.json:
        json.dump({"checkpoints": res, "enters": ent}, open(a.json, "w"), ensure_ascii=False, indent=1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
