"""Compare two regress_img.py output dirs episode by episode: qpos max diff, image hashes per step, and for differing
images (when both kept raw images) the mean / max abs pixel difference. Usage: python regress_img_cmp.py A B"""
import glob, json, os, sys
import numpy as np
A, B = sys.argv[1], sys.argv[2]
tot = {"eps": 0, "steps": 0, "img_diff_steps": 0, "qpos_max": 0.0}
for fa in sorted(glob.glob(os.path.join(A, "t*_e*.npz"))):
    fb = os.path.join(B, os.path.basename(fa))
    if not os.path.exists(fb):
        print("missing", fb); continue
    a, b = np.load(fa), np.load(fb)
    m = min(len(a["qpos"]), len(b["qpos"]))
    dq = float(np.abs(a["qpos"][:m] - b["qpos"][:m]).max()) if m else 0.0
    ha, hb = a["img_md5"], b["img_md5"]
    k = min(len(ha), len(hb))
    neq = [i for i in range(k) if list(ha[i]) != list(hb[i])]
    rec = {"ep": os.path.basename(fa), "steps": [len(a["qpos"]), len(b["qpos"])], "qpos_max_diff": dq,
           "img_steps": k, "img_diff_steps": len(neq), "first_img_diff": neq[0] if neq else None}
    if neq and "imgs" in a.files and "imgs" in b.files:
        j = [i for i in neq if i < min(len(a["imgs"]), len(b["imgs"]))]
        if j:
            d = np.abs(a["imgs"][j].astype(int) - b["imgs"][j].astype(int))
            rec["pix_mean"], rec["pix_max"] = float(d.mean()), int(d.max())
    for key in ("toy_px",):
        if key in a.files or key in b.files:
            rec[key] = [key in a.files, key in b.files]
    print(json.dumps(rec))
    tot["eps"] += 1; tot["steps"] += k; tot["img_diff_steps"] += len(neq); tot["qpos_max"] = max(tot["qpos_max"], dq)
print("TOTAL", json.dumps(tot))
