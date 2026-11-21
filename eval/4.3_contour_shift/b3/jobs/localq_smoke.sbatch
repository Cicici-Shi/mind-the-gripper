#!/bin/bash
#SBATCH -J b3_lqsmk
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:40:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/lqsmk_%j.log
# 局部响应工具冒烟（A100-40，零平移逐位复现要求同型号）：B1、B2 各 2 个校准键。
# 第一段：用 b3 eval_views 在校准键上跑 C4（盲成败，记录带 success_steps_sha256）作状态来源；
# 第二段：localq.py 对它回放并做九个画面的查询。不碰正式键。
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
nvidia-smi -L | grep -q "A100" || { echo "not A100, abort"; exit 1; }
python - <<'P'
import json, keys as K
json.dump(K.make_keys("B1", "calib")[:2], open("keysets/smoke_B1.json", "w"), indent=1)
P
P=()
for b in B1 B2; do
  python eval_views.py --block $b --cell C4 --keys keysets/smoke_$b.json --out out_c4src --blind-success --skip-done \
    > logs/lqsmk_${SLURM_JOB_ID}_c4_$b.log 2>&1 & P+=($!); sleep 8
done; waitall
grep -h "ERROR" logs/lqsmk_${SLURM_JOB_ID}_c4_*.log || echo "no ERROR in C4 source"
P=()
for b in B1 B2; do
  python localq.py --block $b --keys keysets/smoke_$b.json --src out_c4src --out out_localq_smoke --sheet-queries 2,5 \
    > logs/lqsmk_${SLURM_JOB_ID}_lq_$b.log 2>&1 & P+=($!); sleep 8
done; waitall
grep -h "\[lq\]" logs/lqsmk_${SLURM_JOB_ID}_lq_*.log
grep -l "Traceback" logs/lqsmk_${SLURM_JOB_ID}_*.log || echo "no Traceback"
