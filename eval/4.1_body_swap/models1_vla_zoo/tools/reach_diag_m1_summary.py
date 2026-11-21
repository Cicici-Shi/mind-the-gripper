import json, sys, collections
import numpy as np
d = json.load(open(sys.argv[1]))
rows = d["rows"]
for cell in ["obj_g_ta_fc", "obj_g_tfc"]:
    for kind in ["over1", "control"]:
        R = [r for r in rows if r["cell"] == cell and r["kind"] == kind]
        if not R: continue
        rec = np.array([r["rec_pos_mm"] for r in R]); fb = np.array([r["full_best_pos_mm"] for r in R])
        fr = np.array([r["full_best_rot_deg"] for r in R]); po = np.array([r["posonly_mm"] for r in R])
        q = np.array([r["query"] for r in R])
        lim = collections.Counter(j for r in R for j in r["twin_joints_at_limit"])
        plim5 = collections.Counter(j for r in R for j in r["phys_joints_lt5deg"])
        print(f"== {cell} {kind}: frames {len(R)} (query {q.sum()}) keys {sorted(set(tuple(r['key']) for r in R))}")
        print(f"  recorded pos mm p50/p90/max {np.percentile(rec,[50,90,100]).round(2)}")
        print(f"  position-only reachable (<=0.01 mm) {np.sum(po<=0.01)}/{len(R)}; <=1 mm {np.sum(po<=1)}; pos-only max {po.max():.3f}")
        print(f"  full pose 50 seeds: best pos <=1 mm {np.sum(fb<=1)}/{len(R)}; still >1 mm {np.sum(fb>1)}; best pos p50/p90/max {np.percentile(fb,[50,90,100]).round(2)}; rot p50/max {np.percentile(fr,[50,100]).round(2)}")
        print(f"  50-seed best vs recorded (pos mm): best<rec-0.1 {np.sum(fb<rec-0.1)}; |diff|<=0.1 {np.sum(np.abs(fb-rec)<=0.1)}; best>rec+0.1 {np.sum(fb>rec+0.1)}")
        print(f"  frames where any seed reached <=1 mm full pose: {sum(1 for r in R if r['full_seeds_le1mm']>0)}")
        print(f"  twin joints at limit (<1 deg) in best full-pose: {dict(sorted(lim.items()))}; frames with none {sum(1 for r in R if not r['twin_joints_at_limit'])}")
        print(f"  physical arm joints <5 deg from limit: {dict(sorted(plim5.items()))}")
        yaw = np.array([r["target_yaw_deg"] for r in R]); tilt = np.array([r["target_tilt_deg"] for r in R]); dist = np.array([r["target_dist_from_base_m"] for r in R])
        print(f"  target yaw range [{yaw.min():.0f},{yaw.max():.0f}] tilt p50/max {np.percentile(tilt,[50,100]).round(0)} dist-from-base m p50/max {np.percentile(dist,[50,100]).round(3)}")
        z = np.array([r["target_xyz"][2] for r in R]); print(f"  target z range [{z.min():.3f},{z.max():.3f}]")
