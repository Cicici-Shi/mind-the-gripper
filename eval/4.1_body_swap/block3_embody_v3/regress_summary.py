"""Summarize tools/regress.py outputs: per run vs dump, and old-code vs v3 trajectory identity."""
import json, os, sys, glob
import numpy as np
O = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1 else "$PROJ_ROOT/embody_v3/out/regress")
def load(d):
    return {(r["task"], r["ep"]): r for r in map(json.loads, open(os.path.join(O, d, "result.jsonl")))}
runs = sorted(os.path.basename(p.rstrip("/")) for p in glob.glob(os.path.join(O, "*/")))
for d in runs:
    R = load(d); v = list(R.values())
    agree = sum(r["dump_success"] == r["replay_success"] for r in v)
    exact = sum((r["qpos_max_diff"] or 0) <= 1e-9 and r["steps"] == r["dump_steps"] for r in v)
    fd = [r["first_diverge"] for r in v if r["first_diverge"] is not None]
    print(f"{d:14s} n={len(v)} dumpSR={sum(r['dump_success'] for r in v)} replaySR={sum(r['replay_success'] for r in v)} "
          f"agree={agree} exact_traj={exact} first_div(min/med)={(min(fd), int(np.median(fd))) if fd else '-'} "
          f"fixture_mm_max={max(r['fixture_mm'] for r in v):.4f} natural_mm_max={max(r['natural_fixture_mm'] for r in v):.4f} "
          f"align_pos_max={max((r['align'] or {}).get('pos_mm', 0) for r in v):.4f}")
pairs = [("src_prod", "src_v3"), ("sham_prod", "sham_v3"), ("sham_dev", "sham_v3copy"), ("sham_prod", "sham_dev"),
         ("rec_prod", "rec_v3"), ("iiwa_prod", "iiwa_v3"), ("ur5e_prod", "ur5e_v3"), ("apps_prod", "apps_v3"), ("grp_dev", "grp_v3")]
pairs += [(a, b) for a, b in [("shamtw_dev", "shamtw_v3"), ("grptw_dev", "grptw_v3")] if a in runs and b in runs]
print("\nold vs v3 (replayed qpos trajectories)")
for a, b in pairs:
    if a not in runs or b not in runs: continue
    A, B = load(a), load(b); keys = sorted(set(A) & set(B)); worst = 0.0; same_outcome = 0; ndiff = 0
    for k in keys:
        x = np.load(os.path.join(O, a, f"t{k[0]}_e{k[1]}.npz")); y = np.load(os.path.join(O, b, f"t{k[0]}_e{k[1]}.npz"))
        qa, qb = x["qpos"], y["qpos"]
        if qa.shape != qb.shape: ndiff += 1; worst = float("inf")
        else:
            dd = float(np.abs(qa - qb).max()); worst = max(worst, dd); ndiff += dd > 0
        for t in ("twin_pos", "twin_rot"):
            if t in x and t in y and x[t].shape == y[t].shape: worst = max(worst, float(np.abs(x[t] - y[t]).max()))
        same_outcome += A[k]["replay_success"] == B[k]["replay_success"]
    print(f"{a:12s} vs {b:12s} keys={len(keys)} same_outcome={same_outcome} episodes_with_any_diff={ndiff} max_abs_diff={worst:.3e}")
