#!/bin/bash
#SBATCH -J b2c_h47f
#SBATCH -p gpu
#SBATCH --gres=gpu:h100-47:1
# (site-specific #SBATCH --nodelist removed)
#SBATCH --cpus-per-task=6
#SBATCH --mem=64G
#SBATCH --time=01:30:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/b2c/logs/h47f_%j.log
# 协调方 2026-11-13 傍晚：H100 只用 node32（h100-47）。取代 h100_96_final.sbatch（<job> 已取消，未开始）。
# 一卡一进程，依次：定稿冒烟 B1、B2、F1、F2（F 块 --lag 8）-> out_smoke_h100_v2；补跑校准 C0 第一遍 B1 -> out_calib_h100
# （作业 <job> 里这个进程被主机内存 OOM 杀掉；第二遍在 out_calib_h100_aa，作业 <job>，也在 node32）。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
render_selfcheck
gpu_sampler logs/h47f_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in B1 B2 F1 F2; do
  LAG=""; [ "${b:0:1}" = "F" ] && LAG="--lag 8"
  echo "== smoke $b $(date +%T)"
  python eval_views.py --block $b --cells C0,C1,C3,C4,C5 $LAG --keys keysets/smoke_$b.json --out out_smoke_h100_v2 \
    --save-frames 0,40 --frames-dir out_smoke_h100_v2/frames --skip-done --blind-success 2>&1 | grep -v "$NOISE" > logs/h47f_${SLURM_JOB_ID}_smoke_$b.log
  grep -h "model loaded\|ERROR\|finished" logs/h47f_${SLURM_JOB_ID}_smoke_$b.log
done
echo "== calib B1 $(date +%T)"
python eval_views.py --block B1 --cells C0 --keys keysets/calib10_B1.json --out out_calib_h100 --skip-done 2>&1 | grep -v "$NOISE" > logs/h47f_${SLURM_JOB_ID}_calib_B1.log
grep -h "model loaded\|ERROR\|finished" logs/h47f_${SLURM_JOB_ID}_calib_B1.log
kill $GS
echo "gpu mem/util peak: $(sort -t, -k1 -n logs/h47f_${SLURM_JOB_ID}_gpu.csv | tail -1)"
