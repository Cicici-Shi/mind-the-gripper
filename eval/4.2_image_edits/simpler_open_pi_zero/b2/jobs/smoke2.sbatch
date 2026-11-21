#!/bin/bash
#SBATCH -J b2_smoke2
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:50:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/smoke2_%j.log
# 冒烟（v1）：环境核查 probe.py（四块）+ 每个身体一个任务五格各 2 个校准键（五格并行）。
# 校准键上 C0 以外的格用 --blind-success：npz 里不写成败。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
# probe.py 已在 <job> 跑完（四块全过），这里不重跑
mkdir -p keysets
for b in B1 F1; do [ -f keysets/smoke_$b.json ] || python keys.py --block $b --set calib --n 2 --out keysets/smoke_$b.json; done
run_block() { b=$1; mid=$2; P=()
  for c in C0 C1 C3 C4 C5; do
    extra=""; [ $c != C0 ] && extra="--blind-success"
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke --save-frames 0,$mid \
      --save-video --skip-done $extra > logs/smoke2_${SLURM_JOB_ID}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 15
  done; waitall
  for c in C0 C1 C3 C4 C5; do echo "-- $b $c"; grep -v "$NOISE" logs/smoke2_${SLURM_JOB_ID}_${b}_${c}.log | tail -6; done
}
gpu_sampler logs/smoke2_${SLURM_JOB_ID}_gpu.csv & GS=$!
echo "== smoke B1"; run_block B1 15
echo "== smoke F1"; run_block F1 20
kill $GS
grep -il "llvmpipe\|lavapipe" logs/smoke2_${SLURM_JOB_ID}_* && echo "SOFTWARE RENDERER SEEN" || echo "no software renderer in logs"
find out_smoke -name "*.npz" | wc -l; ls out_smoke/*/*/_errors
