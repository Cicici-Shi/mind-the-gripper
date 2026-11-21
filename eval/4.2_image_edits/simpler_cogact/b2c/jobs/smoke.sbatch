#!/bin/bash
#SBATCH -J b2c_smoke
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/smoke_%j.log
# 第二阶段冒烟：四块各 2 个校准键 x 五格，五格一律盲成败。一块一个进程（策略按身体载入一次），块与块顺序跑（一张卡一个进程）。
# 用法：sbatch jobs/smoke.sbatch <输出目录> [块...]；L 取环境变量 LAG_<块>（默认 keys.BLOCKS 的值），定稿冒烟按校准后的 L_final 传；存第 0、40 步的输入图到 <输出目录>/frames（拼图后删掉）。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
OUT=${1:-out_smoke}; shift || true
BL=${@:-B1 B2 F1 F2}
gpu_sampler logs/smoke_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in $BL; do
  echo "== $b $(date +%T)"
  LV=LAG_$b; LARG=""; [ -n "${!LV:-}" ] && LARG="--lag ${!LV}"
  python eval_views.py --block $b --cells C0,C1,C3,C4,C5 $LARG --keys keysets/smoke_$b.json --out $OUT --save-frames 0,40 \
    --frames-dir $OUT/frames --skip-done --blind-success 2>&1 | grep -v "$NOISE" > logs/smoke_${SLURM_JOB_ID}_${b}.log
  grep -h "model loaded\|env \|ERROR\|finished" logs/smoke_${SLURM_JOB_ID}_${b}.log
done
kill $GS
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/smoke_${SLURM_JOB_ID}_gpu.csv | tail -1)"
