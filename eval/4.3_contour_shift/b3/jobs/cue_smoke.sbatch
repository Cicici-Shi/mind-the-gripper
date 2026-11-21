#!/bin/bash
#SBATCH -J b3_cuesmk
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/cuesmk_%j.log
# protocol_cue_widowx 冒烟（只用校准键，盲：日志不印成败）：
#   A §10 参考帧统计（cue_ref10，b2/out_calib 校准 C0 回放，不跑策略）+ 闭环 C0 校准 10 键（比 b2/out_calib）
#   B 闭环其余八格在 2 个冒烟键上（C3、C4 比 b2/out_smoke_v11）
# （查询工具 v1 的冒烟阶段 C 已按协调方 11-13 下午的通知删去：查询工具按 v2 重写）
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
nvidia-smi -L | grep -q "A100-PCIE-40GB\|A100-SXM4-40GB" || { echo "not A100-40, abort"; exit 1; }
O=out_cue_smoke; J=${SLURM_JOB_ID}
gpu_sampler logs/cuesmk_${J}_gpu.csv & GS=$!
S=${STAGE:-AB}
if [[ $S == *A* ]]; then
P=()
for b in B1 B2; do
  python cue_ref10.py replay --block $b --keys ../b2/keysets/calib10_$b.json --ref ../b2/out_calib --out out_ref10 > logs/cuesmk_${J}_ref10_$b.log 2>&1 & P+=($!)
  python cue_closed.py --block $b --cell C0 --keys ../b2/keysets/calib10_$b.json --out $O --skip-done > logs/cuesmk_${J}_C0_$b.log 2>&1 & P+=($!); sleep 10
done; waitall
python cue_ref10.py summarize --root out_ref10 --out cue_ref10.json > logs/cuesmk_${J}_ref10_sum.log 2>&1
date
fi
if [[ $S == *B* ]]; then
for b in B1 B2; do
  P=(); n=0
  for c in C3 C4 S-b S-a S+a S+b SC-b SC+b; do
    python cue_closed.py --block $b --cell $c --keys keysets/smoke_$b.json --out $O --skip-done > logs/cuesmk_${J}_${c}_$b.log 2>&1 & P+=($!); n=$((n+1)); sleep 10
    if [ $n -ge 4 ]; then waitall; P=(); n=0; fi
  done; waitall
done
date
fi
grep -l Traceback logs/cuesmk_${J}_*.log || echo "no Traceback"
grep -h "\[cue\] done\|\[cq\] done\|\[ref10\]\|ERROR" logs/cuesmk_${J}_*.log | sort | uniq -c | sort -rn | head -60
kill $GS
