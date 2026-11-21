#!/bin/bash
#SBATCH -J b2c_calib
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/calib_%j.log
# 第二阶段校准：各块 10 个校准键的 C0（不盲，只有校准键原图格可看成败）。用法：sbatch jobs/calib.sbatch <输出目录> [块...]
# 同一脚本跑两遍（out_calib 与 out_calib_aa）就是 A/A：同键 C0 两遍，动作须逐位相同。一块一个进程，块与块顺序跑。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
OUT=${1:-out_calib}; shift || true
BL=${@:-B1 B2 F1 F2}
gpu_sampler logs/calib_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in $BL; do
  echo "== $b $(date +%T)"
  python eval_views.py --block $b --cells C0 --keys keysets/calib10_$b.json --out $OUT --skip-done 2>&1 \
    | grep -v "$NOISE" > logs/calib_${SLURM_JOB_ID}_${OUT}_${b}.log
  grep -h "model loaded\|env \|ERROR\|finished" logs/calib_${SLURM_JOB_ID}_${OUT}_${b}.log
done
kill $GS
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/calib_${SLURM_JOB_ID}_gpu.csv | tail -1)"
