"""Block-2 re-analysis with the COPY_GRIP_CMD=1 reruns, per protocol_block3.md Appendix A.

prereg_block2.py is not modified. This script copies its loaders, gates, estimators, bootstrap and decision
rules line for line (functions below are verbatim except that the cell table is a parameter), and runs them
twice per model:

  ORIGINAL : the frozen block-2 cell table. Its output must equal results_block2_final.txt exactly
             (checked by diffing the printed section against that file).
  ALIGNED  : the Appendix A.2 swap only: S sham -> sham_cg, IIWA iiwa -> iiwa_cg, UR5e ur5e -> ur5e_cg.
             T0, IIWA_T, UR5e_T, UR5e_R, G, G_T, SRC unchanged (so d_rev is unchanged by construction).

Then the Appendix A / R9 descriptive extras on one joint key set (all cells resampled jointly, B = 10000,
seed 0, percentile CIs, the same index draws as above because the key set is the same 200 keys):
  SR(S)-SR(SRC) split three ways (sham_cg, sham, and d_cg(sham)), d_cg(X) for X in {sham, iiwa, ur5e},
  drop_nat(G) with both S versions (Fable's note in A.1), and block-1 P2 GRP - iiwa_cg with prereg.py's method.

Usage: python prereg_block2_cg.py [RUNS_DIR] [PI05_V2_DIR]
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from summarize import load_dumps  # noqa: E402

RUNS = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "$PROJ_ROOT/runs")
P5 = os.path.expanduser(sys.argv[2] if len(sys.argv) > 2 else "$PROJ_ROOT/pi05/v2")
B, SEED = 10000, 0
CELLS = {"SRC": "src", "S": "sham", "T0": "sham_twin", "IIWA": "iiwa", "IIWA_T": "iiwa_twin",
         "UR5e": "ur5e", "UR5e_T": "ur5e_twin", "UR5e_R": "ur5e_rev", "G": "grpkp100", "G_T": "grpkp100_twin"}
TWIN = {"T0": "sham_twin", "IIWA_T": "iiwa_twin", "UR5e_T": "ur5e_twin", "UR5e_R": "ur5e_rev", "G_T": "grpkp100_twin"}
# Appendix A.2: only these three names change.
SWAP = {"S": "sham_cg", "IIWA": "iiwa_cg", "UR5e": "ur5e_cg"}
CELLS_CG = {c: SWAP.get(c, n) for c, n in CELLS.items()}


def load_oft(name):
    return {k: float(v[1]) for k, v in load_dumps(os.path.join(RUNS, name)).items()}


def load_pi05(name):
    out = {}
    for f in glob.glob(os.path.join(P5, f"{name}_e*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None:
                    out[(r["task"], r["episode"])] = float(r["success"])
    return out


def gate_oft(name):
    pe, re_ = [], []
    for f in glob.glob(os.path.join(RUNS, name, "*.npz")):
        z = np.load(f, allow_pickle=True)
        pe.append(z["twin_pos_err_mm"][::8])
        re_.append(z["twin_rot_err_deg"][::8])
    if not pe:
        return None
    pe, re_ = np.concatenate(pe), np.concatenate(re_)
    return 100.0 * float(((pe > 2) | (re_ > 1)).mean())


def gate_pi05(name):
    n = m = 0
    for f in glob.glob(os.path.join(P5, f"{name}_e*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                if r.get("error") is None and r.get("twin"):
                    n += r["twin"]["n_paint"]
                    m += r["twin"]["miss"]
    return 100.0 * m / n if n else None


def ci(bs, lo, hi):
    return tuple(np.percentile(bs, [lo, hi]))


def analyse(model, loader, gate, CELLS):
    data = {c: loader(n) for c, n in CELLS.items()}
    have = {c: d for c, d in data.items() if d}
    missing = [c for c in CELLS if c not in have]
    keys = sorted(set.intersection(*(set(d) for d in have.values())))
    print(f"\n######## {model}: common keys {len(keys)}; cells " +
          ", ".join(f"{c}={len(d)}" for c, d in data.items()) + (f"; MISSING {missing}" if missing else ""))
    if not keys:
        return {}
    Y = {c: np.array([d[k] for k in keys]) for c, d in have.items()}
    sr = {c: 100 * y.mean() for c, y in Y.items()}
    print("  SR %: " + ", ".join(f"{c} {v:.1f}" for c, v in sr.items()))
    for c, n in TWIN.items():
        g = gate(n)
        if g is not None:
            print(f"  gate {c}: query-frame IK misses {g:.2f}% -> {'PASS' if g < 1 else 'FAIL (condition does not count)'}")

    def est(f, idx=None):
        Z = {c: (y if idx is None else y[idx]) for c, y in Y.items()}
        return f(lambda c: 100 * Z[c].mean())

    contrasts = {
        "SR(S)-SR(SRC)": lambda s: s("S") - s("SRC"),
        "d5 = SR(T0)-SR(S)": lambda s: s("T0") - s("S"),
        "recovery(UR5e)": lambda s: (s("UR5e_T") - s("T0")) - (s("UR5e") - s("S")),
        "recovery(IIWA)": lambda s: (s("IIWA_T") - s("T0")) - (s("IIWA") - s("S")),
        "recovery(G)": lambda s: (s("G_T") - s("T0")) - (s("G") - s("S")),
        "d_rev = SR(UR5e_R)-SR(T0)": lambda s: s("UR5e_R") - s("T0"),
        "interaction(UR5e)": lambda s: (s("UR5e") - s("UR5e_T")) - (s("UR5e_R") - s("T0")),
    }
    rng = np.random.default_rng(SEED)
    n = len(keys)
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    res = {}
    for name, f in contrasts.items():
        try:
            point = est(f)
        except KeyError:
            continue
        bs = np.array([est(f, i) for i in idxs])
        res[name] = (point, ci(bs, 2.5, 97.5), ci(bs, 5, 95))
        p, (l95, h95), (l90, h90) = res[name]
        print(f"  {name:28s} {p:+6.1f} pp  95% [{l95:+6.1f}, {h95:+6.1f}]  90% [{l90:+6.1f}, {h90:+6.1f}]")
    return res


def decide(model, res):
    out = {}
    if not res:
        return out
    print(f"  -- decisions ({model})")
    if "d5 = SR(T0)-SR(S)" in res:
        _, _, (l90, h90) = res["d5 = SR(T0)-SR(S)"]
        out["P5"] = 'equivalent within +-5' if -5 < l90 and h90 < 5 else 'twin cost NOT excluded'
        print(f"  P5 twin pipeline: {out['P5']}")
    if "recovery(UR5e)" in res:
        p, (l, h), _ = res["recovery(UR5e)"]
        v = "confirmed" if (l > 0 and p >= 7) else "falsified" if h < 7 else "inconclusive"
        out["P3"] = v
        print(f"  P3 UR5e painting recovers >= half: {v}")
    if "recovery(G)" in res:
        p, (l, h), _ = res["recovery(G)"]
        v = "confirmed" if h < 21.75 else "falsified" if l > 21.75 else "inconclusive"
        out["P4"] = v
        print(f"  P4 Robotiq painting recovers < half: {v}")
    if "d_rev = SR(UR5e_R)-SR(T0)" in res:
        p, (l, h), _ = res["d_rev = SR(UR5e_R)-SR(T0)"]
        v = "confirmed" if l > -7 else "falsified" if h < -7 else "inconclusive"
        out["P6"] = v
        print(f"  P6 seeing a UR5e costs < half: {v}")
    if "SR(S)-SR(SRC)" in res:
        _, (l, h), _ = res["SR(S)-SR(SRC)"]
        if l > 0 or h < 0:
            print("  baseline note: SR(S) differs from SR(SRC); block-2 conditions are relative to 'Panda via transfer'")
    return out


def p7(oft, pi):
    out = {}
    print("\n######## P7 replication (sign agreement and CI overlap)")
    for k in oft:
        if k in pi:
            (po, (lo1, ho1), _), (pp, (lo2, ho2), _) = oft[k], pi[k]
            if np.sign(po) == np.sign(pp) and not (ho1 < lo2 or ho2 < lo1):
                v = "replicates"
            elif np.sign(po) != np.sign(pp) and (lo2 > 0 or ho2 < 0):
                v = "does not replicate"
            else:
                v = "indeterminate"
            out[k] = v
            print(f"  {k:28s} OFT {po:+6.1f}  pi0.5 {pp:+6.1f}  -> {v}")
    return out


# ---------------------------------------------------------------- extras (Appendix A / R9), joint key set
def extras(model, loader):
    names = {"SRC": "src", "sham": "sham", "sham_cg": "sham_cg", "iiwa": "iiwa", "iiwa_cg": "iiwa_cg",
             "ur5e": "ur5e", "ur5e_cg": "ur5e_cg", "G": "grpkp100"}
    data = {c: loader(n) for c, n in names.items()}
    keys = sorted(set.intersection(*(set(d) for d in data.values())))
    print(f"\n######## {model} extras: joint keys {len(keys)}; " + ", ".join(f"{c}={len(d)}" for c, d in data.items()))
    Y = {c: np.array([d[k] for k in keys]) for c, d in data.items()}
    print("  SR %: " + ", ".join(f"{c} {100 * y.mean():.1f}" for c, y in Y.items()))
    rng = np.random.default_rng(SEED)
    n = len(keys)
    idxs = [rng.integers(0, n, n) for _ in range(B)]
    con = {
        "SR(sham_cg)-SR(SRC)": ("sham_cg", "SRC"),
        "SR(sham)-SR(SRC)": ("sham", "SRC"),
        "d_cg(sham) = sham_cg-sham": ("sham_cg", "sham"),
        "d_cg(iiwa) = iiwa_cg-iiwa": ("iiwa_cg", "iiwa"),
        "d_cg(ur5e) = ur5e_cg-ur5e": ("ur5e_cg", "ur5e"),
        "drop_nat(G) vs sham": ("G", "sham"),
        "drop_nat(G) vs sham_cg": ("G", "sham_cg"),
        "drop_nat(IIWA) vs sham": ("iiwa", "sham"),
        "drop_nat(IIWA_cg) vs sham_cg": ("iiwa_cg", "sham_cg"),
        "drop_nat(UR5e) vs sham": ("ur5e", "sham"),
        "drop_nat(UR5e_cg) vs sham_cg": ("ur5e_cg", "sham_cg"),
    }
    res = {}
    for name, (a, b) in con.items():
        d = Y[a] - Y[b]
        point = 100 * d.mean()
        bs = np.array([100 * d[i].mean() for i in idxs])
        res[name] = (point, ci(bs, 2.5, 97.5), ci(bs, 5, 95))
        flips = int((Y[a] != Y[b]).sum())
        print(f"  {name:30s} {point:+6.1f} pp  95% [{res[name][1][0]:+6.1f}, {res[name][1][1]:+6.1f}]"
              f"  90% [{res[name][2][0]:+6.1f}, {res[name][2][1]:+6.1f}]  discordant keys {flips}"
              f" ({int(((Y[a] == 1) & (Y[b] == 0)).sum())} +, {int(((Y[a] == 0) & (Y[b] == 1)).sum())} -)")
    p, (l, h), _ = res["d_cg(sham) = sham_cg-sham"]
    print(f"  R9 prediction |d_cg(sham)| <= 3 (point): |{p:+.1f}| -> {'holds' if abs(p) <= 3 else 'FAILS'};"
          f" 95% CI inside [-3, +3]: {'yes' if -3 <= l and h <= 3 else 'no'}")
    print(f"  A.3 trigger |d_cg(sham)| > 5 (point): {'TRIGGERED' if abs(p) > 5 else 'not triggered'}")
    # per-task sham vs sham_cg
    tasks = sorted({k[0] for k in keys})
    row = []
    for t in tasks:
        m = np.array([k[0] == t for k in keys])
        row.append(f"t{t} {int(Y['sham'][m].sum())}->{int(Y['sham_cg'][m].sum())}/{int(m.sum())}")
    print("  per task sham -> sham_cg: " + ", ".join(row))
    return res


def p2_block1(runs):
    """Block-1 P2 (prereg.py, OFT only): SR(GRP) - SR(ARM), seed 0, B 10000, joint per pair."""
    def cells(names):
        d = {n: load_dumps(os.path.join(runs, n)) for n in names}
        keys = sorted(set.intersection(*(set(v) for v in d.values())))
        return keys, {n: np.array([d[n][k][1] for k in keys], float) for n in names}

    def boot(keys, f, seed=0):
        rng = np.random.default_rng(seed)
        n = len(keys)
        return np.array([f(rng.integers(0, n, n)) for _ in range(B)])

    print("\n######## block-1 P2 gripper vs arm: SR(GRP) - SR(ARM) (OFT; prereg.py method)")
    for arm, note in [("iiwa", "original"), ("ur5e", "original"), ("iiwa_cg", "Appendix A.2 extra"),
                      ("ur5e_cg", "not listed in A.2; shown for symmetry")]:
        keys, c = cells(["grp", arm])
        diff = c["grp"] - c[arm]
        bs = 100 * boot(keys, lambda i: diff[i].mean())
        lo, hi = np.percentile(bs, [2.5, 97.5])
        verdict = "gripper hurts LESS than arm" if lo > 0 else ("not separated (gripper hurts MORE: upper bound < 0)" if hi < 0 else "not separated")
        print(f"GRP - {arm.upper():8s} n={len(keys):3d} {100*diff.mean():+6.1f} pp  95% CI [{lo:+6.1f}, {hi:+6.1f}]  -> {verdict}  [{note}]")


if __name__ == "__main__":
    mode = os.environ.get("CG_MODE", "all")
    if mode in ("orig", "all"):
        # Section 1: must reproduce results_block2_final.txt byte for byte.
        oft = analyse("OpenVLA-OFT", load_oft, gate_oft, CELLS)
        decide("OpenVLA-OFT", oft)
        pi = analyse("pi0.5-LIBERO", load_pi05, gate_pi05, CELLS)
        decide("pi0.5-LIBERO", pi)
        if oft and pi:
            p7(oft, pi)
    if mode == "orig":
        sys.exit(0)
    print("\n\n==================== ALIGNED (_cg) re-analysis, Appendix A.2 swap: " +
          ", ".join(f"{c}: {CELLS[c]} -> {n}" for c, n in SWAP.items()) + " ====================")
    oft_cg = analyse("OpenVLA-OFT [_cg]", load_oft, gate_oft, CELLS_CG)
    dec_oft_cg = decide("OpenVLA-OFT [_cg]", oft_cg)
    pi_cg = analyse("pi0.5-LIBERO [_cg]", load_pi05, gate_pi05, CELLS_CG)
    dec_pi_cg = decide("pi0.5-LIBERO [_cg]", pi_cg)
    rep_cg = p7(oft_cg, pi_cg) if oft_cg and pi_cg else {}

    print("\n\n==================== Appendix A / R9 extras ====================")
    ex_oft = extras("OpenVLA-OFT", load_oft)
    ex_pi = extras("pi0.5-LIBERO", load_pi05)
    p2_block1(RUNS)

    # side by side
    oft_o = oft if mode == "all" else {}
    pi_o = pi if mode == "all" else {}
    if oft_o:
        dec_oft_o, dec_pi_o = ({}, {})
        import io
        import contextlib
        with contextlib.redirect_stdout(io.StringIO()):
            dec_oft_o = decide("x", oft_o)
            dec_pi_o = decide("x", pi_o)
            rep_o = p7(oft_o, pi_o)
        print("\n\n==================== Side by side: original | aligned (_cg) ====================")
        for model, a, b in [("OFT", oft_o, oft_cg), ("pi0.5", pi_o, pi_cg)]:
            print(f"  -- {model}")
            for k in a:
                (p1, (l1, h1), (m1, n1)), (p2, (l2, h2), (m2, n2)) = a[k], b[k]
                print(f"  {k:28s} {p1:+6.1f} [{l1:+6.1f}, {h1:+6.1f}] | {p2:+6.1f} [{l2:+6.1f}, {h2:+6.1f}]"
                      f"  (90%: [{m1:+5.1f}, {n1:+5.1f}] | [{m2:+5.1f}, {n2:+5.1f}])")
        for model, a, b in [("OFT", dec_oft_o, dec_oft_cg), ("pi0.5", dec_pi_o, dec_pi_cg)]:
            for k in a:
                print(f"  {model:6s} {k}: {a[k]:32s} | {b[k]:32s} {'(same)' if a[k] == b[k] else '(CHANGED)'}")
        for k in rep_o:
            print(f"  P7 {k:28s} {rep_o[k]:20s} | {rep_cg.get(k, '-'):20s} {'(same)' if rep_o[k] == rep_cg.get(k) else '(CHANGED)'}")
