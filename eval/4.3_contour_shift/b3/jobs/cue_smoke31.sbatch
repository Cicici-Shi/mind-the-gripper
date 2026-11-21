#!/bin/bash
#SBATCH -J b3_cuesmk31
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=01:30:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/b3/logs/cuesmk31_%j.log
# protocol_cue_widowx_v3_1_final 冒烟（只用校准键，盲：日志不印成败）：
#   M §17 第 9 条补量：cue_obcheck 回放 calib10 原图格参考轨迹（不跑策略）
#   C 闭环十三格：C0 在 calib10（比 b2/out_calib），其余十二格在 2 个冒烟键（C3、C4 比 b2/out_smoke_v11）；输出 out_cue_smoke31
source $PROJ_ROOT/bench2_simpler/b3/jobs/common.sh
nvidia-smi -L | grep -q "A100-PCIE-40GB\|A100-SXM4-40GB" || { echo "not A100-40, abort"; exit 1; }
O=out_cue_smoke31; J=${SLURM_JOB_ID}
P=()
for b in B1 B2; do
  python cue_obcheck.py replay --block $b --keys ../b2/keysets/calib10_$b.json --ref ../b2/out_calib --out out_obcheck31 > logs/cuesmk31_${J}_ob_$b.log 2>&1 & P+=($!)
  python cue_closed.py --block $b --cell C0 --keys ../b2/keysets/calib10_$b.json --out $O --skip-done > logs/cuesmk31_${J}_C0_$b.log 2>&1 & P+=($!); sleep 10
done; waitall
python cue_obcheck.py summarize --root out_obcheck31 > logs/cuesmk31_${J}_ob_sum.log 2>&1
date
for b in B1 B2; do
  P=(); n=0
  for c in C3 C4 S-b S-a S+a S+b SC-b SC+b OB-b OB+b S-h S+h; do
    python cue_closed.py --block $b --cell $c --keys keysets/smoke_$b.json --out $O --skip-done > logs/cuesmk31_${J}_${c}_$b.log 2>&1 & P+=($!); n=$((n+1)); sleep 10
    if [ $n -ge 4 ]; then waitall; P=(); n=0; fi
  done; waitall
done
date
grep -l Traceback logs/cuesmk31_${J}_*.log || echo "no Traceback"
grep -h "ERROR" logs/cuesmk31_${J}_*.log | head
