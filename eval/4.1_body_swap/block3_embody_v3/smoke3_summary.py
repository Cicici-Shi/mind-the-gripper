import json, glob, sys, numpy as np
for name in sys.argv[2:]:
    v = [json.loads(l) for l in open(sys.argv[1] + f"/{name}.jsonl")]
    a = [r["align"] for r in v]
    print(f"{name:9s} n={len(v)} SR={sum(r['replay_success'] for r in v)}/{len(v)} pos_mm_max={max(x['pos_mm'] for x in a):.4f} rot_max={max(x['rot_deg'] for x in a):.4f} "
          f"fix={max(r['fixture_vs_src_mm'] for r in v):.4f} finite={all(r['finite'] for r in v)} qvel={max(r['qvel_arm_max'] for r in v):.2f} warn={sum(r['mj_warnings'] for r in v)}")
    if v[0].get("twin"):
        t = [r["twin"] for r in v]
        pa = [x["t0"]["agentview_image"][1] for x in t]; pw = [x["t0"]["robot0_eye_in_hand_image"][1] for x in t]
        print(f"   pad residual: query frames >1mm {sum(x['over_1mm_q'] for x in t)}/{sum(x['nq'] for x in t)}, all frames >2mm {sum(x['miss'] for x in t)}/{sum(x['n'] for x in t)}, "
              f"max {max(x['pos_max'] for x in t):.4f} mm, hand rot max {max(x['rot_max'] for x in t):.4f} deg, cam err {max(x['cam_err_mm_max'] or 0 for x in t):.4f} mm, stale {sum(x['stale'] for x in t)}; "
              f"t0 vs ref Panda agent mean {np.mean(pa):.3f} (max {max(pa):.3f}) wrist mean {np.mean(pw):.3f} (max {max(pw):.3f})")
