import json, glob, numpy as np
O = "out/replay_b3"
for n in ["Panda", "IIWA", "UR5e", "UR5e_nc", "PandaL3", "Kinova3", "PandaRT2"]:
    v = [json.loads(l) for f in glob.glob(f"{O}/{n}_t[0-9].jsonl") for l in open(f)]
    no3 = [r for r in v if r["task"] != 3]; t3 = [r for r in v if r["task"] == 3]
    s = (f"{n:9s} n={len(v)} SR={100*np.mean([r['replay_success'] for r in v]):.1f} SR_not3={100*np.mean([r['replay_success'] for r in no3]):.1f} "
         f"t3={sum(r['replay_success'] for r in t3)}/{len(t3)} trk={np.median([r['track_err_mm_mean'] for r in v]):.2f} sat={np.mean([r['sat_substep_any'] for r in v]):.4f} "
         f"align_max={max(r['align_pos_mm'] for r in v):.4f} rot_max={max(r['align_rot_deg'] for r in v):.4f} contact_eps={sum(r['contacts']['arm_env_contact_steps'] > 0 for r in v)} "
         f"finite={all(r['finite'] for r in v)} warn={sum(r['mj_warnings'] for r in v)}")
    if n == "UR5e_nc":
        s += (f"\n          pen_eps_t3={sum(r['contacts']['forearm_pen_steps'] > 0 for r in t3)}/{len(t3)} pen_max_t3={max(r['contacts']['forearm_pen_max_mm'] for r in t3):.1f}mm "
              f"pen_eps_other={sum(r['contacts']['forearm_pen_steps'] > 0 for r in no3)} geoms_off={v[0]['contacts']['forearm_geoms_off']} "
              f"pairs={sorted({k for r in t3 for k in r['contacts']['forearm_pen_pairs']})}")
    print(s)
