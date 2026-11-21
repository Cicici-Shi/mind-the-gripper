#!/usr/bin/env python3
"""physics_breakdown_scan.py -- read-only physics-breakdown (failed-simulation) scan.

Fable ruling add1 (protocol_models1_gate_rulings_fable_add1.md, 2026-11-05) section 2.2 detector D and section 4
items 1 and 6. Based on Fable's appendix-A script (~/tmp/fable_add1_scan.py, sha256 c8ddd97e...f92d); thresholds and
the detector definition are unchanged. Login node / CPU only, numpy only, no MuJoCo.

NEVER reads any `success` / outcome field: npz members are loaded lazily and only `qpos`, `joint_names` (npz mode) or
`qpos` (states mode) are touched; log mode only counts lines containing "QACC". Episode length (n_steps) is NOT
written (it can reveal whether an episode ran to the step limit), nor the number of out-of-range steps (same
reason); only the first breakdown step is kept.

Detector D (per episode = key (task, ep)):
  D-A (arm):    any of the 7 robot0_joint* changes by |dq| > pi/2 rad between consecutive recorded steps, or any step is
                beyond the Panda joint range by > 0.1 rad.
  D-B (finger): gripper0_finger_joint at any step < FLO or > FHI (default [-0.5, 1.2] rad).
  breakdown := D-A or D-B. A looser finger range [-1, 2] is also reported per cell (threshold sensitivity only).
  Only valid for Panda-arm cells; cells whose name marks a non-Panda arm (k3, iiwa, ur5e, rt2) are skipped.
  For jsonl models (pi0, pi0.5, OpenVLA) no per-step qpos exists: --logs counts "QACC" warning lines per
  model x cell x shard (an upper bound of breakdown episodes; warnings cannot be attributed to episodes because
  stderr is flushed at process exit).

Modes and interface (fixed; the sensitivity readout job calls the first two exactly like this):
  python tools/physics_breakdown_scan.py --out out/physics_breakdown_m1.json <npz dir...>
  python tools/physics_breakdown_scan.py --logs --out out/physics_breakdown_m1_logs.json
  python tools/physics_breakdown_scan.py --preset m1 --out out/physics_breakdown_m1.json        (same dirs as below)
  python tools/physics_breakdown_scan.py --preset published --out out/physics_breakdown_published.json
  python tools/physics_breakdown_scan.py --preset published --logs --out out/physics_breakdown_published_logs.json
  python tools/physics_breakdown_scan.py --preset toy2 --out out/physics_breakdown_toy2.json          (states mode)
  python tools/physics_breakdown_scan.py --preset toy2 --logs --out out/physics_breakdown_toy2_logs.json
  python tools/physics_breakdown_scan.py --preset physfix --out out/physics_breakdown_physfix.json   (2026-11-08, 12 pf_ cells)
  --logs without --preset and without files = preset m1 logs.

models1 npz dirs (preset m1; every cell of zoo_cells.sh except smoke dirs, all shards present at scan time):
  $PROJ_ROOT/vla_zoo/runs/oft_sp/{src,sham_cg,sham_twin,grpkp100,g_tfc,g_tw_fc,g_ta_fc,g_t}
  $PROJ_ROOT/vla_zoo/runs/oft_obj/obj_{src,sham_cg,sham_twin,grpkp100,g_tfc,g_tw_fc,g_ta_fc,g_t}
models1 logs (preset m1): $PROJ_ROOT/vla_zoo/logs/c_pi0_*.log, c_openvla_*.log (all attempts of a shard summed, as in
  Fable add1 section 1 table: pi0 total 28).
published (preset published): third-batch OFT Panda-arm cells in $PROJ_ROOT/runs (spatial G family + controls; object
  obj_src, obj_s, obj_t0, obj_g, obj_g_t); pi0.5 logs $PROJ_ROOT/pi05/logs/v2_*.log.
toy2 (preset toy2): $PROJ_ROOT/embody_toy/out/toy2a1/states/t<task>_e<ep>/q<step>.npz (fork_run2 FORK_SAVE: full MjData
  qpos at every policy query of the G baseline, i.e. snapshots are one query apart, not one control step; D-A's jump
  test is then applied between consecutive snapshots). qpos layout taken from a toy1 toy_g dump (same body, same
  qpos length, the 8 joints used are the first 8 qpos entries). Logs: $PROJ_ROOT/embody_toy/logs/t2a1_<job>_p<i>.log.

Output (npz / states mode):
  {"<model>|<suite>": [[task, ep], ...]   union over ALL cells of that model x suite (exclusion list: drop these keys
                                          from every cell, keeping keys paired),
   "_by_cell": {"<model>|<suite>|<cell>": {dir, keys, no_qpos, shards, breakdown, A_arm, B_finger, breakdown_loose}},
   "_meta": {thresholds, dirs, keys per cell, skipped, script sha256, time}}
  model label: vla_zoo runs -> parent dir (oft_sp / oft_obj); $PROJ_ROOT/runs -> oft_pub; embody_toy/out/runs -> oft_toy1;
  toy2 states -> toy2a1.
Output (logs mode): {"<model>|<cell>": {"<ep offset>": n_QACC_lines, ...}, "_by_log": {...}, "_meta": {...}}
  model: pi0 / openvla / groot (c_<model>_<cell>_<job>_e<off>.log), pi05 (v2_<cell>_<job>_e<off>.log),
  toy2a1 (t2a1_<job>_p<i>.log, key "toy2a1|<job>", shard "p<i>").
"""
import argparse, glob, hashlib, json, os, re, sys, time
import numpy as np

HOME = os.path.expanduser("~")
P = HOME + "/proj"
LIM = np.array([[-2.8973, 2.8973], [-1.7628, 1.7628], [-2.8973, 2.8973], [-3.0718, -0.0698], [-2.8973, 2.8973],
                [-0.0175, 3.7525], [-2.8973, 2.8973]])
NON_PANDA = re.compile(r"(^|_)(k3|iiwa|ur5e|rt2)")
M1_CELLS = ["src", "sham_cg", "sham_twin", "grpkp100", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"]
PRESET_DIRS = {
    "m1": [f"{P}/vla_zoo/runs/oft_sp/{c}" for c in M1_CELLS] + [f"{P}/vla_zoo/runs/oft_obj/obj_{c}" for c in M1_CELLS],
    "published": [f"{P}/runs/{c}" for c in ("src", "sham", "sham_cg", "sham_twin", "grpkp100", "grpkp100_twin",
                                             "grpkp100_tw", "grpkp100_ta", "g_r", "grp", "p6j3")]
                 + [f"{P}/runs/{c}" for c in ("obj_src", "obj_s", "obj_t0", "obj_g", "obj_g_t")],
    "toy2": [f"{P}/embody_toy/out/toy2a1/states"],
    # physfix rev3 (2026-11-08): final cell table, 12 cells = <job>-<job>
    "physfix": [f"{P}/vla_zoo/runs/pf_{m}_obj/pf_obj_{c}" for m in ("oftc", "ofti") for c in ("sham_cg", "sham_twin", "grpkp20", "g_tfc_kp20")]
                 + [f"{P}/vla_zoo/runs/pf_{m}_sp/pf_{c}" for m in ("oftc", "ofti") for c in ("sham_cg", "sham_twin")],
}
PRESET_LOGS = {
    "m1": [f"{P}/vla_zoo/logs/c_pi0_*.log", f"{P}/vla_zoo/logs/c_openvla_*.log"],
    "published": [f"{P}/pi05/logs/v2_*.log"],
    "toy2": [f"{P}/embody_toy/logs/t2a1_*_p*.log"],
    "physfix": [],  # OFT in-process: QACC is per episode in the npz (pf_qacc_count), see pf_gate0.py
}
TOY_REF = f"{P}/embody_toy/out/runs/toy_g"


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def label(d):
    d = os.path.abspath(d.rstrip("/"))
    cell, parent = os.path.basename(d), os.path.dirname(d)
    pb = os.path.basename(parent)
    if pb in ("oft_sp", "oft_obj"):
        model = pb
    elif parent == f"{P}/runs":
        model = "oft_pub"
    elif parent == f"{P}/embody_toy/out/runs":
        model = "oft_toy1"
    elif cell == "states" and pb.startswith("toy2"):
        model, cell = pb, "G_baseline"
    else:
        model = pb
    suite = "libero_object" if (pb == "oft_obj" or cell.startswith(("obj_", "pf_obj_"))) else "libero_spatial"  # pf_obj_: 2026-11-08
    return model, suite, cell


def detect(q, ai, fi, flo, fhi):
    """q: (T, nq) qpos rows in time order; ai: 7 arm qpos indices; fi: finger index or None.
    Returns (A dict or None, B dict or None, loose-B flag)."""
    A = B = None
    qa = q[:, ai]
    if len(q) > 1:
        dq = np.abs(np.diff(qa, axis=0)).max(axis=1)
        jump = np.nonzero(dq > np.pi / 2)[0]
        mj = round(float(np.degrees(dq.max())), 1)
    else:
        jump, mj = np.zeros(0, int), 0.0
    oob = np.nonzero(((qa < LIM[:, 0] - 0.1) | (qa > LIM[:, 1] + 0.1)).any(axis=1))[0]
    if len(jump) or len(oob):
        # no count of steps beyond the limit: after an explosion the arm can stay out of range to the last step,
        # so that count plus first_step would reveal the episode length (outcome proxy)
        A = {"first_step": int(min(([jump[0] + 1] if len(jump) else []) + ([oob[0]] if len(oob) else []))),
             "max_jump_deg": mj, "beyond_limit": bool(len(oob))}
    loose = False
    if fi is not None:
        fq = q[:, fi]
        bad = np.nonzero((fq < flo) | (fq > fhi))[0]
        if len(bad):
            B = {"first_step": int(bad[0]), "extreme": round(float(fq[np.abs(fq).argmax()]), 2)}
        loose = bool(((fq < -1.0) | (fq > 2.0)).any())
    return A, B, loose


def scan_npz_dir(d, flo, fhi):
    fs = sorted(glob.glob(os.path.join(d, "t*_e*_i*.npz")))
    A, B, L, nk, nostep, shards = {}, {}, set(), 0, 0, set()
    for f in fs:
        m = re.search(r"t(\d+)_e(\d+)_i", os.path.basename(f))
        k = (int(m.group(1)), int(m.group(2)))
        try:
            z = np.load(f, allow_pickle=True)
            if "qpos" not in z.files:
                nostep += 1
                continue
            q = np.asarray(z["qpos"], float)
            names = [str(n) for n in z["joint_names"]]
        except Exception:
            nostep += 1
            continue
        nk += 1
        shards.add(k[1] // 5 * 5)
        ai = [i for i, n in enumerate(names) if re.match(r"robot0_joint\d$", n)]
        fl = [i for i, n in enumerate(names) if n == "gripper0_finger_joint"]
        if len(ai) != 7:
            continue
        a, b, lo = detect(q, ai, fl[0] if fl else None, flo, fhi)
        if a: A[k] = a
        if b: B[k] = b
        if lo or a: L.add(k)
    return nk, nostep, sorted(shards), A, B, L


def toy_layout():
    f = sorted(glob.glob(os.path.join(TOY_REF, "t*_e*_i*.npz")))[0]
    z = np.load(f, allow_pickle=True)
    names = [str(n) for n in z["joint_names"]]
    nq = z["qpos"].shape[1]
    want = [f"robot0_joint{i}" for i in range(1, 8)] + ["gripper0_finger_joint"]
    assert names[:8] == want, names[:8]  # first 8 joints are hinge joints -> qpos address == index
    return list(range(7)), 7, nq, f


def scan_states_dir(d, flo, fhi):
    ai, fi, nq, ref = toy_layout()
    A, B, L, nk, nostep = {}, {}, set(), 0, 0
    for kd in sorted(glob.glob(os.path.join(d, "t*_e*"))):
        m = re.match(r"t(\d+)_e(\d+)$", os.path.basename(kd))
        if not m:
            continue
        k = (int(m.group(1)), int(m.group(2)))
        fs = glob.glob(os.path.join(kd, "q*.npz"))
        fs.sort(key=lambda p: int(re.search(r"q(\d+)\.npz$", p).group(1)))
        try:
            q = np.stack([np.asarray(np.load(p)["qpos"], float) for p in fs])
        except Exception:
            nostep += 1
            continue
        if q.ndim != 2 or q.shape[1] != nq or not len(q):
            nostep += 1
            continue
        nk += 1
        a, b, lo = detect(q, ai, fi, flo, fhi)
        steps = [int(re.search(r"q(\d+)\.npz$", p).group(1)) for p in fs]
        if a: a["first_step"] = steps[a["first_step"]]; A[k] = a  # snapshot index -> policy step
        if b: b["first_step"] = steps[b["first_step"]]; B[k] = b
        if lo or a: L.add(k)
    return nk, nostep, [], A, B, L, ref


def run_npz(dirs, a):
    out, by_cell, meta_dirs, skipped = {}, {}, [], []
    for d in dirs:
        d = os.path.abspath(os.path.expanduser(d.rstrip("/")))
        model, suite, cell = label(d)
        if NON_PANDA.search(cell):
            skipped.append({"dir": d, "reason": "non-Panda arm: Panda joint range does not apply"})
            continue
        if not os.path.isdir(d):
            skipped.append({"dir": d, "reason": "missing"})
            continue
        ref = None
        if cell == "G_baseline":
            nk, nostep, shards, A, B, L, ref = scan_states_dir(d, a.flo, a.fhi)
        else:
            nk, nostep, shards, A, B, L = scan_npz_dir(d, a.flo, a.fhi)
        keys = sorted(set(A) | set(B))
        g = f"{model}|{suite}"
        out.setdefault(g, set()).update(keys)
        by_cell[f"{g}|{cell}"] = {
            "dir": d, "keys": nk, "no_qpos": nostep, "shards": shards, "breakdown": [list(k) for k in keys],
            "A_arm": {f"{k[0]},{k[1]}": v for k, v in sorted(A.items())},
            "B_finger": {f"{k[0]},{k[1]}": v for k, v in sorted(B.items())},
            "breakdown_loose": [list(k) for k in sorted(L)]}
        if ref:
            by_cell[f"{g}|{cell}"]["qpos_layout_ref"] = ref
            by_cell[f"{g}|{cell}"]["note"] = "snapshots one policy query apart; D-A jump test between snapshots"
        meta_dirs.append(d)
        print(f"{g}|{cell}: keys {nk} no_qpos {nostep} shards {shards} | breakdown {len(keys)} {keys} | "
              f"A {sorted(A)} | B {dict((k, v['extreme']) for k, v in sorted(B.items()))} | loose {len(L)}", flush=True)
    res = {g: [list(k) for k in sorted(v)] for g, v in sorted(out.items())}
    for g, v in res.items():
        print(f"== {g}: union {len(v)} {v}", flush=True)
    res["_by_cell"] = by_cell
    res["_meta"] = {"mode": "npz", "finger_range": [a.flo, a.fhi], "finger_range_loose": [-1.0, 2.0],
                    "arm_jump_rad": float(np.pi / 2), "arm_limit_margin_rad": 0.1, "panda_limits": LIM.tolist(),
                    "dirs": meta_dirs, "keys_per_cell": {c: v["keys"] for c, v in by_cell.items()},
                    "skipped": skipped, "preset": a.preset}
    return res


def run_logs(files, a):
    pats = [(re.compile(r"^c_(pi0|openvla|groot)_(.+)_(\d+)_e(\d+)\.log$"), lambda m: (m[1], m[2], m[3], m[4])),
            (re.compile(r"^v2_(.+)_(\d+)_e(\d+)\.log$"), lambda m: ("pi05", m[1], m[2], m[3])),
            (re.compile(r"^t2a1_(\d+)_p(\d+)\.log$"), lambda m: ("toy2a1", m[1], m[1], "p" + m[2]))]
    res, by_log, unparsed = {}, {}, []
    for f in files:
        b = os.path.basename(f)
        for rx, fn in pats:
            m = rx.match(b)
            if m:
                model, cell, job, sh = fn(m)
                break
        else:
            unparsed.append(f)
            continue
        with open(f, "rb") as fh:
            n = sum(1 for ln in fh if b"QACC" in ln)
        key = f"{model}|{cell}"
        res.setdefault(key, {})
        res[key][sh] = res[key].get(sh, 0) + n
        by_log.setdefault(f"{key}|{sh}", {})[job] = n
    res = {k: dict(sorted(v.items(), key=lambda kv: (len(kv[0]), kv[0]))) for k, v in sorted(res.items())}
    tot = {}
    for k, v in res.items():
        tot[k.split("|")[0]] = tot.get(k.split("|")[0], 0) + sum(v.values())
        if sum(v.values()):
            print(f"{k}: {v}", flush=True)
    print(f"== totals per model: {tot}  (logs {len(files)}, unparsed {len(unparsed)})", flush=True)
    res["_by_log"] = by_log
    res["_meta"] = {"mode": "logs", "files": len(files), "unparsed": unparsed, "totals": tot, "preset": a.preset,
                    "note": "count of lines containing QACC per model x cell x shard, all attempts (job ids) summed; "
                            "upper bound of breakdown episodes; not attributable to episodes"}
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--logs", action="store_true")
    ap.add_argument("--preset", choices=sorted(PRESET_DIRS))
    ap.add_argument("--flo", type=float, default=-0.5)
    ap.add_argument("--fhi", type=float, default=1.2)
    ap.add_argument("paths", nargs="*", help="npz dirs (npz mode) or log files (--logs)")
    a = ap.parse_args()
    if a.logs:
        pats = a.paths or PRESET_LOGS[a.preset or "m1"]
        files = sorted(set(f for p in pats for f in glob.glob(os.path.expanduser(p))))
        res = run_logs(files, a)
        res["_meta"]["globs"] = pats
    else:
        dirs = list(a.paths) or (PRESET_DIRS[a.preset] if a.preset else [])
        if not dirs:
            ap.error("give npz dirs or --preset")
        res = run_npz(dirs, a)
    me = os.path.abspath(__file__)
    res["_meta"].update({"script": me, "script_sha256": sha256(me), "time": time.strftime("%Y-%m-%d %H:%M:%S"),
                         "host": os.uname().nodename, "argv": sys.argv[1:]})
    os.makedirs(os.path.dirname(os.path.abspath(a.out)) or ".", exist_ok=True)
    tmp = a.out + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(res, fh, indent=1)
    os.replace(tmp, a.out)
    print(f"wrote {a.out}  sha256 {sha256(a.out)}", flush=True)


if __name__ == "__main__":
    main()
