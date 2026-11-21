#!/bin/bash
#SBATCH -J c_s0rp_val
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:25:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/step0_replay/logs/val_%j.log
# 校验：每块每格 2 键，rgbd 与 state_dict 两种模式各回放一遍。不加载策略。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
D=$PROJ_ROOT/bench2_cogact/step0_replay; REC=$PROJ_ROOT/bench2_cogact/b2c/out_formal
cd $D
P=()
for b in B1 B2 F1 F2; do
  for m in rgbd state_dict; do
    python replay_cogact.py --rec-root $REC --out $D/val_$m --obs-mode $m --list $D/val_$b.txt 2>&1 | grep -v "$NOISE" > $D/logs/val_${SLURM_JOB_ID}_${b}_${m}.log &
    P+=($!)
  done
done
for p in "${P[@]}"; do wait $p; echo "pid $p exit $?"; done
grep -h "^\[rp\]" $D/logs/val_${SLURM_JOB_ID}_*.log | sort
python cmp_modes.py $D/val_rgbd $D/val_state_dict
