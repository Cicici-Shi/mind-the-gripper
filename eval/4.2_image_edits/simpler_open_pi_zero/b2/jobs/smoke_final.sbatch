#!/bin/bash
#SBATCH -J b2_smokefin
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b2/logs/smokefin_%j.log
# 定稿代码的端到端冒烟（接手方）：四块各 2 个校准键 x 五格，五格一律 --blind-success；产物给加固后的 gate.py
# 对候选冻结清单做一遍（代码哈希、L、线宽、G、保护名单、ICD、第 0 帧五格一致、实际 episode_id / urdf）。
source $PROJ_ROOT/bench2_simpler/b2/jobs/common.sh
gpu_sampler logs/smokefin_${SLURM_JOB_ID}_gpu.csv & GS=$!
for b in B1 B2 F1 F2; do
  echo "== $b"; P=()
  for c in C0 C1 C3 C4 C5; do
    python eval_views.py --block $b --cell $c --keys keysets/smoke_$b.json --out out_smoke_v11 --save-frames 0 \
      --skip-done --blind-success > logs/smokefin_${SLURM_JOB_ID}_${b}_${c}.log 2>&1 &
    P+=($!); sleep 10
  done; waitall
  grep -h "ERROR" logs/smokefin_${SLURM_JOB_ID}_${b}_*.log || echo "no ERROR in $b"
done
kill $GS
for b in B1 B2 F1 F2; do python gate.py --root out_smoke_v11 --block $b --keys keysets/smoke_$b.json \
  --manifest FROZEN_bench2.candidate --out out_smoke_v11/gate_$b.json; echo; done
