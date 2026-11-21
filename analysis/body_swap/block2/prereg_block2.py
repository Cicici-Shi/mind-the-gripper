"""Block-2 (cross-paint) pre-registered analysis, exactly as frozen in project/protocol_block2.md
(sha256 788f6e62...). Per model: OFT from per-episode dumps, pi0.5 from v2 jsonl (records with an error
dropped, one clean record per (task, episode)). Paired cluster bootstrap over (task, episode) keys common
to all cells of that model, all cells resampled jointly, B = 10000, seed 0, percentile CIs.

Usage: python prereg_block2.py [RUNS_DIR] [PI05_V2_DIR]
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


def analyse(model, loader, gate):
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
    if not res:
        return
    print(f"  -- decisions ({model})")
    if "d5 = SR(T0)-SR(S)" in res:
        _, _, (l90, h90) = res["d5 = SR(T0)-SR(S)"]
        print(f"  P5 twin pipeline: {'equivalent within +-5' if -5 < l90 and h90 < 5 else 'twin cost NOT excluded'}")
    if "recovery(UR5e)" in res:
        p, (l, h), _ = res["recovery(UR5e)"]
        v = "confirmed" if (l > 0 and p >= 7) else "falsified" if h < 7 else "inconclusive"
        print(f"  P3 UR5e painting recovers >= half: {v}")
    if "recovery(G)" in res:
        p, (l, h), _ = res["recovery(G)"]
        v = "confirmed" if h < 21.75 else "falsified" if l > 21.75 else "inconclusive"
        print(f"  P4 Robotiq painting recovers < half: {v}")
    if "d_rev = SR(UR5e_R)-SR(T0)" in res:
        p, (l, h), _ = res["d_rev = SR(UR5e_R)-SR(T0)"]
        v = "confirmed" if l > -7 else "falsified" if h < -7 else "inconclusive"
        print(f"  P6 seeing a UR5e costs < half: {v}")
    if "SR(S)-SR(SRC)" in res:
        _, (l, h), _ = res["SR(S)-SR(SRC)"]
        if l > 0 or h < 0:
            print("  baseline note: SR(S) differs from SR(SRC); block-2 conditions are relative to 'Panda via transfer'")


oft = analyse("OpenVLA-OFT", load_oft, gate_oft)
decide("OpenVLA-OFT", oft)
pi = analyse("pi0.5-LIBERO", load_pi05, gate_pi05)
decide("pi0.5-LIBERO", pi)
if oft and pi:
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
            print(f"  {k:28s} OFT {po:+6.1f}  pi0.5 {pp:+6.1f}  -> {v}")
