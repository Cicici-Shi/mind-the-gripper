import json,glob,os,numpy as np
O=os.path.expanduser("$PROJ_ROOT/embody_v3/out/replay_closed"); OLD=os.path.expanduser("$PROJ_ROOT/replay/closed")
def load(d,name):
    R={}
    for f in glob.glob(f"{d}/{name}_t[0-9].jsonl"):
        for l in open(f): r=json.loads(l); R[(r["task"],r["init"])]=r
    return R
print("body            n  SR%   trk_mm med  rot_deg med  sat_any(substep) sat_joint  per-joint sat    finite  qvel_max  warn  align_pos_max fixture_max | old SR / same outcome / max |trk diff|")
for name in ["Panda","IIWA","UR5e","PandaR85_kp100","PandaL3","Kinova3","PandaRT"]:
    R=load(O,name); v=list(R.values())
    sr=100*np.mean([r["replay_success"] for r in v])
    pj=np.mean([r["sat_per_joint"] for r in v],axis=0)
    old=load(OLD,name); s=""
    if old:
        ks=set(old)&set(R); same=sum(old[k]["replay_success"]==R[k]["replay_success"] for k in ks)
        dd=max(abs(old[k]["track_err_mm_mean"]-R[k]["track_err_mm_mean"]) for k in ks)
        s=f"| {100*np.mean([old[k]['replay_success'] for k in ks]):.1f} / {same}/{len(ks)} / {dd:.2e}"
    print(f"{name:15s} {len(v)} {sr:5.1f} {np.median([r['track_err_mm_mean'] for r in v]):8.2f} {np.median([r['track_rot_deg_mean'] for r in v]):8.2f} "
          f"{np.mean([r['sat_substep_any'] for r in v]):10.4f} {np.mean([r['sat_joint_frac'] for r in v]):8.4f}  {np.round(pj,3).tolist()} "
          f"{all(r['finite'] for r in v)} {max(r['qvel_arm_max'] for r in v):6.2f} {sum(r['mj_warnings'] for r in v)} "
          f"{max(r['align_pos_mm'] for r in v):.4f} {max(r['fixture_mm'] for r in v):.4f} {s}")
