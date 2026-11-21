#!/bin/bash
#SBATCH -J b2c_a80
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-80:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/a80_%j.log
# 顺手验证（协调方 2026-11-13 第 2 条）：A100 80GB（MIG 关闭）上 CogACT 的动作是否与 A100-40 逐位相同。
# B1、F1 各 2 个校准键的 C0 -> out_calib_a80，再与 A100-40 的 out_calib 同键比 env_actions / raw_actions / pred_chunks。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
for b in B1 F1; do
  python eval_views.py --block $b --cells C0 --keys keysets/calib2_$b.json --out out_calib_a80 --skip-done 2>&1 \
    | grep -v "$NOISE" > logs/a80_${SLURM_JOB_ID}_${b}.log
  grep -h "model loaded\|ERROR" logs/a80_${SLURM_JOB_ID}_${b}.log
done
python - <<'PY'
import glob, os, numpy as np
for p in sorted(glob.glob("out_calib_a80/*/C0/*.npz")):
    q = p.replace("out_calib_a80", "out_calib")
    a, b = np.load(p), np.load(q)
    print(os.path.basename(p), a["gpu_name"], "vs", b["gpu_name"],
          {f: bool(np.array_equal(a[f], b[f])) for f in ("env_actions", "raw_actions", "pred_chunks", "frame0_sha256", "seeds")})
PY
