"""Summarise tools/t0check_m1.py output against protocol_models1.md 7.2 step-0 gates (2026-11-04)."""
import json, sys
import numpy as np
A, W = "agentview_image", "robot0_eye_in_hand_image"
for f in sys.argv[1:]:
    R = [json.loads(l) for l in open(f) if l.strip()]
    print(f"== {f}: {len(R)} keys, hosts {sorted({(r['host'], r['gpu']) for r in R})}")
    t0a = np.mean([r["t0_vs_s"][A]["mean"] for r in R]); t0w = np.mean([r["t0_vs_s"][W]["mean"] for r in R])
    print(f"T0 vs S: per-cell average agent {t0a:.4f} wrist {t0w:.4f} (GATE < 0.11 each) -> {'PASS' if t0a < 0.11 and t0w < 0.11 else 'FAIL'}")
    for n, same, changed in (("g_tfc", None, (A, W)), ("g_tw_fc", A, (W,)), ("g_ta_fc", W, (A,)), ("g_t", None, (A, W))):
        k = f"{n}_vs_g"
        out, ok = [], True
        if same:
            mx = max(r[k][same]["max"] for r in R)
            nz = sum(r[k][same]["max"] > 0 for r in R)
            out.append(f"unreplaced {same[:5]} max |diff| {mx} (keys nonzero {nz}; GATE 0)")
            mm = float(np.mean([r[k][same]["mean"] for r in R])); out.append(f"mean {mm:.2e} (render-noise rule: max <= 1 and mean < 0.11)")
            ok &= bool(mx <= 1 and mm < 0.11)
        for c in changed:
            kh = f"{n}_vs_g_both_hidden"
            m = np.mean([r[kh][c]["mean"] for r in R]); lim = 1.0 if c == A else 8.0
            mv = np.mean([r[k][c]["mean"] for r in R])
            out.append(f"replaced {c[:5]} both-grippers-hidden avg mean {m:.4f} max-key {max(r[kh][c]['mean'] for r in R):.4f} "
                       f"(GATE < {lim}); grippers visible avg {mv:.3f} (descriptive)")
            ok &= bool(m < lim)
        print(f"{n} vs G: " + "; ".join(out) + f" -> {'PASS' if ok else 'FAIL'}")
    for n in ("sham_twin", "g_tfc", "g_tw_fc", "g_ta_fc", "g_t"):
        cm = [r["twin"][n]["cam_err_mm0"] for r in R if r["twin"].get(n) and r["twin"][n]["cam_err_mm0"] is not None]
        cd = [r["twin"][n]["cam_err_deg0"] for r in R if r["twin"].get(n) and r["twin"][n]["cam_err_deg0"] is not None]
        pe = [r["twin"][n]["pos_err_mm0"] for r in R if r["twin"].get(n)]
        print(f"  {n}: first-paint twin cam err max {max(cm):.5f} mm / {max(cd):.5f} deg, mean {np.mean(cm):.4f} mm; IK pad residual max {max(pe):.4f} mm")
