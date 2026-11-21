"""physfix rev3 impl-list item 14 / 7.1 (d) / 7.2 second layer (2026-11-08): kp20 table on kp20 physics, left / right pad
alignment residuals of the GTfc'' twin. Descriptive only, NO threshold. Reads the <job> geo_grasp_pf.py jsons
(physics only, no policy, no pf_ outcome); no simulation is run.

Twin geometry reproduced from embody_next body_env.py (sha 38a1b81a...):
  * twin Panda grip site = physical R85 grip site shifted by R @ (0, 0, -R85_PAD_BACK = -6.4 mm), same orientation
    (IK residual assumed 0; t0check <job> gates it separately), so twin Panda pads sit at Panda-site z -3.6 - 6.4 =
    -10.0 mm in the R85 site frame;
  * twin Panda finger qpos = [0.039 f, -0.039 f], f = clip(1 - finger_joint / 0.50) (old kp20 table, _grp_proprio).
Per row (R85 site frame, mm; opening axis = site x, pads paired by side):
  w_R          physical R85 pad net gap (from the json, box-support face distance along the hand opening axis)
  w_P          net gap of the twin Panda pads at the mapped f (Panda pad-face model fitted on the <job> Panda sweeps)
  dw           w_R - w_P
  res_{side}   3-D distance between the twin Panda pad inner-face point and the R85 pad inner-face point:
               axis component = inner-face x (pad centre x -/+ half the summed box support, equal halves by mirror
               symmetry); y, z components = pad centre y, z (the face point's off-axis coordinates are approximated by
               the pad centre; same convention as the PAD_Z table).
Sets: free (4 kp20 free-air sweeps x 260 steps, open20 close120 open60 close60), free_closed_end (last 20 steps of
each close segment), blocked_object (scripted pick, phases close+lift+hold, 30 object-suite trials), blocked_bowl
(same, 12 bowl-rim trials). Usage: python kp20_pad_lr_pf.py --indir out/pf/geo_<job>/j --out kp20_pad_lr.json
"""
import argparse, glob, json, os
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--indir", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
PANDA_OPEN, R85_CLOSED, PAD_BACK, PANDA_PAD_Z = 0.039, 0.50, 0.0064, -0.0036
L = lambda p: json.load(open(p))

# --- Panda pad-face model from the Panda free-air sweeps: centre |x| = c0 + c1 * q, summed half-thickness hP ---
X, Y, H, Z = [], [], [], []
for p in sorted(glob.glob(os.path.join(a.indir, "sweep_*_panda.json"))):
    for r in L(p)["rows"]:
        q = np.asarray(r["q"]); pp, pm = np.asarray(r["pad_p_site"]), np.asarray(r["pad_m_site"])
        for c, qq in ((pp, abs(q[0])), (pm, abs(q[1]))):
            X.append(qq); Y.append(abs(c[0])); Z.append(c[2])
        H.append(abs(pp[0] - pm[0]) - r["gap"])
X, Y = np.asarray(X), np.asarray(Y)
c1, c0 = np.polyfit(X, Y, 1)
fit_res = float(np.max(np.abs(c0 + c1 * X - Y)))
hP = float(np.median(H))
panda_model = {"centre_abs_x_mm": f"{c0*1000:.4f} + {c1:.5f} * q_mm", "fit_max_abs_res_mm": fit_res * 1000,
               "summed_half_thickness_mm": hP * 1000, "summed_half_thickness_range_mm": [min(H) * 1000, max(H) * 1000],
               "pad_centre_z_mm_range": [min(Z) * 1000, max(Z) * 1000], "n_rows": len(H)}
assert fit_res < 5e-4 and max(H) - min(H) < 2e-4 and max(abs(z - PANDA_PAD_Z) for z in Z) < 1e-5, panda_model


def row(r):
    fj = r["q"][0]
    f = float(r["f_kp20"])  # exact mapped value stored by geo_grasp_pf.py (json q is rounded to 1e-6 rad)
    assert abs(f - float(np.clip(1.0 - fj / R85_CLOSED, 0.0, 1.0))) < 1e-5
    qP = PANDA_OPEN * f
    xc = c0 + c1 * qP                      # Panda pad centre |x| (site frame)
    xf_P = xc - hP / 2                     # Panda inner face |x|
    w_P = 2 * xf_P
    pp, pm = np.asarray(r["pad_p_site"]), np.asarray(r["pad_m_site"])
    hR = abs(pp[0] - pm[0]) - r["gap"]
    out = {"fj": fj, "f": f, "w_R": r["gap"] * 1000, "w_P": w_P * 1000, "dw": (r["gap"] - w_P) * 1000}
    zP = PANDA_PAD_Z - PAD_BACK
    for nm, c in (("p", pp), ("m", pm)):
        s = np.sign(c[0])
        fR = np.array([c[0] - s * hR / 2, c[1], c[2]])
        fP = np.array([s * xf_P, 0.0, zP])
        d = (fP - fR) * 1000
        out["res_" + nm] = float(np.linalg.norm(d)); out["ax_" + nm] = float(abs(d[0])); out["z_" + nm] = float(d[2])
    out["mid_z_R"] = float((pp[2] + pm[2]) / 2 * 1000); out["mid_x_R"] = float((pp[0] + pm[0]) / 2 * 1000)
    return out


def summ(R):
    if not R:
        return {"n": 0}
    o = {"n": len(R)}
    for k in ("w_R", "w_P", "dw", "res_p", "res_m", "ax_p", "ax_m", "z_p", "z_m", "mid_z_R", "mid_x_R", "f", "fj"):
        v = np.asarray([x[k] for x in R])
        o[k] = {"p50": float(np.percentile(v, 50)), "p90": float(np.percentile(v, 90)), "max": float(v.max()),
                "min": float(v.min())}
    for side in ("p", "m"):  # |.| for the residual summaries
        v = np.abs([x["z_" + side] for x in R]); o["absz_" + side] = {"p50": float(np.percentile(v, 50)),
                                                                      "p90": float(np.percentile(v, 90)), "max": float(v.max())}
    both = np.asarray([x["res_p"] for x in R] + [x["res_m"] for x in R])
    o["res_either_pad"] = {"p50": float(np.percentile(both, 50)), "p90": float(np.percentile(both, 90)), "max": float(both.max())}
    return o


sets = {"free": [], "free_closed_end": [], "blocked_object": [], "blocked_bowl": []}
files = {}
for p in sorted(glob.glob(os.path.join(a.indir, "sweep_*_kp20.json"))):
    d = L(p); assert d["seq"] == "open20 close120 open60 close60" and len(d["rows"]) == 260
    rr = [row(r) for r in d["rows"]]
    sets["free"] += rr
    sets["free_closed_end"] += rr[120:140] + rr[240:260]
    files[os.path.basename(p)] = len(rr)
for p in sorted(glob.glob(os.path.join(a.indir, "grasp_*_kp20.json"))):
    d = L(p)
    rr = [row(r) for r, ph in zip(d["rows"], d["phases"]) if ph in ("close", "lift", "hold")]
    sets["blocked_bowl" if d["suite"] == "libero_spatial" else "blocked_object"] += rr
    files[os.path.basename(p)] = len(rr)
res = {"what": "kp20 table on kp20 physics: w_R vs mapped twin Panda gap, and left/right pad residuals (mm); descriptive, no line",
       "protocol": "protocol_physfix.md rev3 7.1 (d), 7.2 second layer, impl list item 14", "source_dir": a.indir,
       "files": files, "panda_model": panda_model, "pad_names": {"p": "R85 right_fingerpad / Panda finger1 (site -x)",
                                                                  "m": "R85 left_fingerpad / Panda finger2 (site +x)"},
       "sets": {k: summ(v) for k, v in sets.items()}}
json.dump(res, open(a.out, "w"), indent=1)
for k, s in res["sets"].items():
    if not s["n"]:
        print(k, "n 0"); continue
    f = lambda q: "/".join(f"{s[q][x]:.2f}" for x in ("p50", "p90", "max"))
    print(f"{k:16s} n {s['n']:5d}  w_R {f('w_R')}  w_P {f('w_P')}  dw {f('dw')}  res_R(p) {f('res_p')}  res_L(m) {f('res_m')}"
          f"  ax_p {f('ax_p')} ax_m {f('ax_m')}  |z| {f('absz_p')}  (p50/p90/max mm)")
print("panda model", panda_model)
