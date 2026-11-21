#!/bin/bash
#SBATCH -J b2c_h47
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-47:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/h47_%j.log
# H100 NVL（h100-47）上一个进程：校准 C0 第二遍（A/A，第一遍在 h100-96 上，同一型号），四块各 10 个校准键 -> out_calib_h100_aa
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
nvidia-smi --query-gpu=memory.total,memory.used --format=csv
gpu_sampler logs/h47_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in B1 B2 F1 F2; do
  echo "== $b $(date +%T)"
  python eval_views.py --block $b --cells C0 --keys keysets/calib10_$b.json --out out_calib_h100_aa --skip-done 2>&1 \
    | grep -v "$NOISE" > logs/h47_${SLURM_JOB_ID}_${b}.log
  grep -h "model loaded\|ERROR\|finished" logs/h47_${SLURM_JOB_ID}_${b}.log
done
kill $GS
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/h47_${SLURM_JOB_ID}_gpu.csv | tail -1)"
