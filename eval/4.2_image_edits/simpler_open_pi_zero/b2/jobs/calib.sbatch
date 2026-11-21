#!/bin/bash
#SBATCH -J b2_calib
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:55:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/calib_%j.log
# 校准（v1，只用校准键、只跑 C0）：四块各 10 个校准键 C0（含本体状态探针），四块并行；
# 再对 B1、F1 原样跑第二遍做 A/A。滞后 IoU、涂黑强度都从这些 C0 回合算（calib.py）。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
mkdir -p keysets
for b in B1 B2 F1 F2; do python keys.py --block $b --set calib --n 10 --out keysets/calib10_$b.json; done
gpu_sampler logs/calib_${SLURM_JOB_ID}_gpu.csv & GS=$!
P=()
for b in B1 B2 F1 F2; do
  python eval_views.py --block $b --cell C0 --keys keysets/calib10_$b.json --out out_calib --proprio-probe \
    --save-frames 0 --skip-done > logs/calib_${SLURM_JOB_ID}_${b}_C0.log 2>&1 &
  P+=($!); sleep 15
done; waitall
for b in B1 B2 F1 F2; do echo "-- $b"; grep -v "$NOISE" logs/calib_${SLURM_JOB_ID}_${b}_C0.log | tail -4; done
echo "== A/A second pass"
P=()
for b in B1 F1; do
  python eval_views.py --block $b --cell C0 --keys keysets/calib10_$b.json --out out_calib_aa2 --proprio-probe \
    --skip-done > logs/calib_${SLURM_JOB_ID}_${b}_C0_aa2.log 2>&1 &
  P+=($!); sleep 15
done; waitall
for b in B1 F1; do echo "-- $b aa2"; grep -v "$NOISE" logs/calib_${SLURM_JOB_ID}_${b}_C0_aa2.log | tail -3; done
kill $GS
grep -il "llvmpipe\|lavapipe" logs/calib_${SLURM_JOB_ID}_* && echo "SOFTWARE RENDERER SEEN" || echo "no software renderer in logs"
ls out_calib/*/C0/_errors out_calib_aa2/*/C0/_errors
