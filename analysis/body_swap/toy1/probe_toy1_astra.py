"""TOY1 probe analysis under protocol_toy1_astra_probe_rule.md (Astra 6, 2026-11-03 afternoon). Supersedes the
section 5.3 / 5.4 part of the frozen analysis/prereg_toy1.py (65e6c173...), which was written for the original
5-chunk-boundary frame rule; that script is not modified (closed-loop readout unchanged). Written and frozen BEFORE any
probe output was read.

Inputs: shift.jsonl / parts.jsonl / missing.jsonl from tools/probe_toy.py --frame-rule astra (one or more dirs).
Each frame row: appearance (panda | robotiq), task, ep, step, phase, dist_cm, stratum (main < 7.5 cm, mid 7.5-15,
far > 15), u_x, u_y, R_base, actions{cond: 8x7 processed} for cond in 0, 0p, x-10, x-5, x+5, x+10, y-5, y+5,
cam-5, cam+5; centroid_px / expected_px; same_0_0p.

Quantities (world frame: a_w = R_base @ a[:3]); PRIMARY = first step of the re-queried chunk (k = 1), 8-step sums
descriptive:
  D_x   = (a_w(x+5) - a_w(x-5)) . u_x      prediction under H_ruler: D_x < 0 (compensation against the shift)
  D_x10 = (a_w(x+10) - a_w(x-10)) . u_x    dose
  L     = (a_w(x+5) - a_w(x-5)) . u_y      orthogonal leak of the x shift
  D_y   = (a_w(y+5) - a_w(y-5)) . u_y      descriptive
  D_cam = (a_w(cam+5) - a_w(cam-5)) . u_x  descriptive only (parallax, decision (f))
Aggregation (rule 4): per episode and stratum the median over its frames; then over episodes.
Screening (per appearance, main stratum), all three needed for "detected" (检出), otherwise "not detected" (未检出):
  sign: share of episodes with median D_x < 0 >= 0.70
  specificity: median_ep |D_x| >= 2 x median_ep |L|
  dose: share of episodes with median |D_x10| > median |D_x| >= 0.60
Main stratum must have >= 30 episodes covering >= 5 tasks, else Q2 is descriptive only (rule 3; no back-filling from
mid / far). Bootstrap: episodes resampled (paired across appearances, keys = (task, ep) present in both), B = 10000,
seed 0, percentile 95% CI for median D_x per appearance and for the panda - robotiq difference.
Interpretation table (original 5.3 with "检出 / 未检出" wording): panda not detected -> ruler not detected (not
"absent"); panda detected & robotiq not detected -> supports H_ruler (Robotiq breaks the reading); both detected and
robotiq median |D_x| >= half of panda's -> reading works on Robotiq, the drop is not a reading error; both detected but
robotiq < half -> not covered by the table, printed as such. Caveat printed (rule 5): counterfactual "if re-planned
now" response; does not show the executed 8-step chunk uses this feedback, nor localise to the fingertips; far-layer
non-compensation does not refute the ruler.
Manipulation checks (printed): 0 vs 0p pixel / action identity rate; wrist centroid shift for x+5 vs the projection
of the shift (expected_px): share with |shift| in 5-12 px and share with cosine > 0.
Parts (5.4, descriptive): d(ci, cj) = mean over the 8 steps of |pos_i - pos_j| (also first step only);
r_i = d(ci, c1) / d(c0, c1); per-episode median over frames (all strata and main stratum), then median over episodes
with bootstrap CI; coverage counts for c5, c6.
Usage: python analysis/probe_toy1_astra.py --dirs DIR[,DIR] --out FILE [--expect-rule-sha SHA]
"""
import argparse, glob, hashlib, json, os, sys
from collections import defaultdict
import numpy as np

RULE_SHA = "df3053eb4a0d5018328aaafd158953a8f90d274ff0d53010e79f327cc2381dc9"
SIGN_MIN, SPEC_K, DOSE_MIN, MIN_EPS, MIN_TASKS = 0.70, 2.0, 0.60, 30, 5
B, SEED = 10000, 0

ap = argparse.ArgumentParser()
ap.add_argument("--dirs", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--rule", default=os.path.expanduser("$PROJ_ROOT/embody_toy/protocol_toy1_astra_probe_rule.md"))
ap.add_argument("--expect-rule-sha", default=RULE_SHA)
args = ap.parse_args()
if os.path.exists(args.out):
    sys.exit(f"refusing to overwrite {args.out}")
sha = lambda p: hashlib.sha256(open(p, "rb").read()).hexdigest()
if os.path.exists(args.rule) and sha(args.rule) != args.expect_rule_sha:
    sys.exit(f"rule file sha mismatch: {sha(args.rule)}")
lines = []
P = lambda *a: lines.append(" ".join(str(x) for x in a))
P(f"probe_toy1_astra.py sha256 {sha(os.path.abspath(__file__))}; rule {sha(args.rule) if os.path.exists(args.rule) else 'file missing'}")

rows, parts, missing = [], [], []
for d in args.dirs.split(","):
    for f in glob.glob(os.path.join(d, "**", "shift.jsonl"), recursive=True):
        rows += [json.loads(l) for l in open(f)]
    for f in glob.glob(os.path.join(d, "**", "parts.jsonl"), recursive=True):
        parts += [json.loads(l) for l in open(f)]
    for f in glob.glob(os.path.join(d, "**", "missing.jsonl"), recursive=True):
        missing += [json.loads(l) for l in open(f)]


def aw(r, c, k=0):
    R = np.asarray(r["R_base"])
    a = np.asarray(r["actions"][c])
    return R @ a[k, :3] if k is not None else R @ a[:, :3].sum(0)


def frame_q(r, k=0):
    ux, uy = np.asarray(r["u_x"]), np.asarray(r["u_y"])
    d = lambda p, m, u: float((aw(r, p, k) - aw(r, m, k)) @ u)
    return {"D_x": d("x+5", "x-5", ux), "D_x10": d("x+10", "x-10", ux), "L": d("x+5", "x-5", uy),
            "D_y": d("y+5", "y-5", uy), "D_cam": d("cam+5", "cam-5", ux)}


def per_episode(app, stratum, k=0):
    """{(task, ep): {q: median over frames}} for one appearance and stratum (None = all)."""
    g = defaultdict(list)
    for r in rows:
        if r["appearance"] == app and (stratum is None or r.get("stratum") == stratum):
            g[(r["task"], r["ep"])].append(frame_q(r, k))
    return {key: {q: float(np.median([f[q] for f in fs])) for q in fs[0]} for key, fs in g.items()}


def screen(E):
    if not E:
        return None
    dx = np.array([e["D_x"] for e in E.values()])
    d10 = np.array([e["D_x10"] for e in E.values()])
    lk = np.array([e["L"] for e in E.values()])
    sign = float((dx < 0).mean())
    spec = float(np.median(np.abs(dx))), float(np.median(np.abs(lk)))
    dose = float((np.abs(d10) > np.abs(dx)).mean())
    det = sign >= SIGN_MIN and spec[0] >= SPEC_K * spec[1] and dose >= DOSE_MIN
    return {"n_eps": len(E), "n_tasks": len({k[0] for k in E}), "sign_share": sign, "median_abs_Dx": spec[0],
            "median_abs_L": spec[1], "dose_share": dose, "median_Dx": float(np.median(dx)), "detected": det}


def boot_median(vals, keys_sets):
    rng = np.random.default_rng(SEED)
    keys = sorted(keys_sets)
    out = []
    for _ in range(B):
        idx = rng.integers(0, len(keys), len(keys))
        out.append([np.median([v[keys[i]] for i in idx]) for v in vals])
    out = np.asarray(out)
    return np.percentile(out, [2.5, 97.5], axis=0)


P("== frames")
P("rows", len(rows), "parts rows", len(parts), "missing episodes", len(missing), [(m["task"], m["ep"], m.get("missing")) for m in missing])
for app in ("panda", "robotiq"):
    rs = [r for r in rows if r["appearance"] == app]
    cnt = defaultdict(int)
    for r in rs:
        cnt[r.get("stratum")] += 1
    ph = defaultdict(int)
    for r in rs:
        ph[r.get("phase")] += 1
    P(f"{app}: frames {len(rs)} episodes {len({(r['task'], r['ep']) for r in rs})} strata {dict(cnt)} phases {dict(ph)}")
fp = {(r["task"], r["ep"], r["step"]) for r in rows if r["appearance"] == "panda"}
fr = {(r["task"], r["ep"], r["step"]) for r in rows if r["appearance"] == "robotiq"}
P("same frames in both appearances:", fp == fr, len(fp), len(fr))

P("== manipulation checks")
for app in ("panda", "robotiq"):
    rs = [r for r in rows if r["appearance"] == app]
    if not rs:
        continue
    same = np.mean([r["same_0_0p"]["pixels"] for r in rs]), np.mean([r["same_0_0p"]["actions"] for r in rs])
    P(f"{app}: 0 vs 0p identical pixels {same[0]:.3f} actions {same[1]:.3f}")
    ok_mag, ok_dir, n = 0, 0, 0
    for r in rs:
        c, e = r["centroid_px"].get("x+5"), r["expected_px"].get("x+5")
        if c is None or e is None:
            continue
        n += 1
        c, e = np.asarray(c), np.asarray(e)
        ok_mag += 5 <= np.linalg.norm(c) <= 12
        ok_dir += float(c @ e) > 0
    P(f"{app}: x+5 wrist centroid shift in 5-12 px {ok_mag}/{n}, direction agrees with projection {ok_dir}/{n}")
    ce = max(max(r["cam_err_mm"][c] for c in r["cam_err_mm"] if not c.startswith("cam")) for r in rs)
    P(f"{app}: max twin-vs-physical wrist camera error (non-camera conditions) {ce:.2e} mm")

res = {}
for kname, k in (("first step (PRIMARY)", 0), ("8-step sum (descriptive)", None)):
    P(f"== Q2 {kname}")
    for st in ("main", "mid", "far", None):
        Es = {app: per_episode(app, st, k) for app in ("panda", "robotiq")}
        for app in ("panda", "robotiq"):
            s = screen(Es[app])
            P(f"  stratum={st or 'all'} {app}: {s}")
            if k == 0 and st == "main":
                res[app] = s
        common = set(Es["panda"]) & set(Es["robotiq"])
        if common and k == 0:
            vp = {key: Es["panda"][key]["D_x"] for key in common}
            vr = {key: Es["robotiq"][key]["D_x"] for key in common}
            vd = {key: vp[key] - vr[key] for key in common}
            ci = boot_median([vp, vr, vd], common)
            P(f"  stratum={st or 'all'} paired eps {len(common)}: median D_x panda {np.median(list(vp.values())):.4f} [{ci[0,0]:.4f}, {ci[1,0]:.4f}]"
              f" robotiq {np.median(list(vr.values())):.4f} [{ci[0,1]:.4f}, {ci[1,1]:.4f}] diff {np.median(list(vd.values())):.4f} [{ci[0,2]:.4f}, {ci[1,2]:.4f}]")
        if k == 0:
            for app in ("panda", "robotiq"):
                E = Es[app]
                if E:
                    P(f"  stratum={st or 'all'} {app} descriptive medians: D_y {np.median([e['D_y'] for e in E.values()]):.4f}"
                      f" D_cam {np.median([e['D_cam'] for e in E.values()]):.4f} |D_x10| {np.median([abs(e['D_x10']) for e in E.values()]):.4f}")

P("== Q2 screening (main stratum, first step)")
enough = {app: (res.get(app) and res[app]["n_eps"] >= MIN_EPS and res[app]["n_tasks"] >= MIN_TASKS) for app in ("panda", "robotiq")}
if not all(enough.values()):
    P("main stratum too small (need >= 30 episodes and >= 5 tasks per appearance):",
      {a: (res[a]["n_eps"], res[a]["n_tasks"]) if res.get(a) else None for a in res}, "-> Q2 DESCRIPTIVE ONLY")
else:
    p, r = res["panda"]["detected"], res["robotiq"]["detected"]
    P(f"panda: {'检出' if p else '未检出'}; robotiq: {'检出' if r else '未检出'}")
    if not p:
        P("reading: ruler not detected on Panda pixels (未检出, not 'absent'); if Q1 says cue, the cue is a template")
    elif not r:
        P("reading: detected on Panda, not on Robotiq -> direct support for H_ruler; full stage branch B with fingertip geometry")
    elif res["robotiq"]["median_abs_Dx"] >= 0.5 * res["panda"]["median_abs_Dx"]:
        P("reading: detected on both, Robotiq magnitude >= half of Panda -> the reading works on Robotiq; drop is not a reading error")
    else:
        P("reading: detected on both but Robotiq < half of Panda -> not covered by the interpretation table")
P("caveat: counterfactual 'if re-planned now' response; does not show that the executed 8-step chunk uses this feedback,"
  " does not localise the effect to the fingertips; non-compensation in far layers does not refute the ruler; passing"
  " the screen cannot exclude a reading bias.")

P("== Q3 parts (descriptive)")
def dist(r, a, b, k=None):
    A, Bm = np.asarray(r["actions"][a])[:, :3], np.asarray(r["actions"][b])[:, :3]
    return float(np.linalg.norm(A - Bm, axis=1).mean()) if k is None else float(np.linalg.norm(A[k] - Bm[k]))
for kname, k in (("8-step mean", None), ("first step", 0)):
    for st in (None, "main"):
        g = defaultdict(lambda: defaultdict(list))
        for r in parts:
            if st and r.get("stratum") != st:
                continue
            d01 = dist(r, "c0", "c1", k)
            if d01 <= 0:
                continue
            for c in ("c2", "c3", "c4", "c5", "c6"):
                if c in r["actions"]:
                    g[c][(r["task"], r["ep"])].append(dist(r, c, "c1", k) / d01)
        for c in ("c2", "c3", "c4", "c5", "c6"):
            E = {key: float(np.median(v)) for key, v in g[c].items()}
            if not E:
                P(f"  {kname} stratum={st or 'all'} r_{c}: missing (门槛未核实)")
                continue
            ci = boot_median([E], set(E))
            P(f"  {kname} stratum={st or 'all'} r_{c}: median {np.median(list(E.values())):.3f} [{ci[0,0]:.3f}, {ci[1,0]:.3f}] eps {len(E)}")
cov = defaultdict(list)
for r in parts:
    for c, v in r.get("coverage", {}).items():
        for cam, x in v.items():
            cov[(c, cam)].append((x["twin_px"], x["on_phys_kept_px"]))
for key, v in sorted(cov.items()):
    v = np.asarray(v)
    P(f"  coverage {key}: twin px median {np.median(v[:,0]):.0f}, landing on kept physical gripper px median {np.median(v[:,1]):.0f}")

open(args.out, "w").write("\n".join(lines) + "\n")
print("\n".join(lines))
