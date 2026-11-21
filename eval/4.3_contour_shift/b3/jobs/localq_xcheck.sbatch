#!/bin/bash
#SBATCH -J b3_lqx
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:20:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/lqx_%j.log
# 零平移复现的交叉核对：状态来源换成 b2 冻结代码在 A100-40 上跑出的 C4 冒烟记录（b2/out_smoke_v11，校准键，只读，盲记录）。
# 这批记录没有成败与成败哈希，核对一不可用；只看核对二。闭合后只查零平移（--queries preclose）。
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
nvidia-smi -L | grep -q "A100" || { echo "not A100, abort"; exit 1; }
P=()
for b in B1 B2; do
  python localq.py --block $b --keys keysets/smoke_$b.json --src ../b2/out_smoke_v11 --out out_localq_xcheck --queries preclose \
    > logs/lqx_${SLURM_JOB_ID}_$b.log 2>&1 & P+=($!); sleep 8
done; waitall
grep -h "\[lq\]" logs/lqx_${SLURM_JOB_ID}_*.log; grep -l Traceback logs/lqx_${SLURM_JOB_ID}_*.log || echo "no Traceback"
