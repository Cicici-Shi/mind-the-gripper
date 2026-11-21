#!/bin/bash
#SBATCH -J b2_smokepost
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/smokepost_%j.log
# 采集结束后的环境不变检查：用同一组校准键重跑定稿冒烟（四块 x 2 键 x 五格，盲成败），输出到 out_smoke_post；
# 再用 cmp_smoke.py 与定稿冒烟 out_smoke_v11（作业 <job>）逐条比 raw_actions、env_actions、frame0_sha256。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
gpu_sampler logs/smokepost_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in B1 B2 F1 F2; do
  echo "== $b"; P=()
  for c in C0 C1 C3 C4 C5; do
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke_post --save-frames 0 \
      --skip-done --blind-success > logs/smokepost_${SLURM_JOB_ID}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 10
  done; waitall
  grep -h "ERROR" logs/smokepost_${SLURM_JOB_ID}_${b}_*.log || echo "no ERROR in $b"
done
kill $GS
python cmp_smoke.py --new out_smoke_post --ref out_smoke_v11
for b in B1 B2 F1 F2; do python gate.py --root out_smoke_post --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2 --out out_smoke_post/gate_$b.json; echo; done
