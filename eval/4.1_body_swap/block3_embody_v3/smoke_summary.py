import json,glob,numpy as np
for f in sorted(glob.glob("$PROJ_ROOT/embody_v3/out/smoke/*.jsonl")):
    v=[json.loads(l) for l in open(f)]; n=v[0]["name"]; a=[r["align"] for r in v]
    print(f"{n:6s} n={len(v)} SR={sum(r['replay_success'] for r in v)}/{len(v)} pos_mm_max={max(x['pos_mm'] for x in a):.4f} rot_deg_max={max(x['rot_deg'] for x in a):.4f} "
          f"fix_vs_src={max(r['fixture_vs_src_mm'] for r in v):.4f} cam_pos_mm={max(x['wrist_cam_pos_mm'] for x in a):.3f} cam_rot={max(x['wrist_cam_rot_deg'] for x in a):.3f} "
          f"hand={max(x['hand_in_site_deg'] for x in a):.4f} finite={all(r['finite'] for r in v)} qvel={max(r['qvel_arm_max'] for r in v):.2f} warn={sum(r['mj_warnings'] for r in v)} trk={np.median([r['track_err_mm_mean'] for r in v]):.2f}")
    if v[0].get("twin"):
        t=[r["twin"] for r in v]
        print(f"        twin qmiss={sum(x['qmiss'] for x in t)}/{sum(x['nq'] for x in t)} allmiss={sum(x['miss'] for x in t)}/{sum(x['n'] for x in t)} pos_max={max(x['pos_max'] for x in t):.4f} rot_max={max(x['rot_max'] for x in t):.4f} "
              f"stale={sum(x['stale'] for x in t)} t0 agent_mean_max={max(x['t0']['agentview_image'][1] for x in t):.3f} wrist_mean_max={max(x['t0']['robot0_eye_in_hand_image'][1] for x in t):.3f} "
              f"posture_p95={max((x['posture_p95'] or 0) for x in t):.2f} cam_err_mm={max((x['cam_err_mm_max'] or 0) for x in t):.4f} retries={sum(x['retries'] for x in t)}")
