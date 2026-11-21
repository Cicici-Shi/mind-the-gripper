#!/bin/bash
#SBATCH -J b2_smoke3
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:40:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/smoke3_%j.log
# 冒烟补做（协议 v1.1，接手方）：B2、F2（新键）各 2 个校准键 x 五格，五格同卡并行。五格一律 --blind-success（不写成败）。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
mkdir -p keysets
for b in B2 F2; do python keys.py --block $b --set calib --n 2 --out keysets/smoke_$b.json; done
run_block() { b=$1; mid=$2; P=()
  for c in C0 C1 C3 C4 C5; do
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke --save-frames 0,$mid \
      --save-video --skip-done --blind-success > logs/smoke3_${SLURM_JOB_ID}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 15
  done; waitall
  for c in C0 C1 C3 C4 C5; do echo "-- $b $c"; grep -v "$NOISE" logs/smoke3_${SLURM_JOB_ID}_${b}_${c}.log | tail -6; done
}
gpu_sampler logs/smoke3_${SLURM_JOB_ID}_gpu.csv & GS=$!
echo "== smoke B2"; run_block B2 7
echo "== smoke F2"; run_block F2 20
kill $GS
grep -il "llvmpipe\|lavapipe" logs/smoke3_${SLURM_JOB_ID}_* && echo "SOFTWARE RENDERER SEEN" || echo "no software renderer in logs"
find out_smoke/B2 out_smoke/F2 -name "*.npz" | wc -l; ls out_smoke/B2/*/_errors out_smoke/F2/*/_errors
