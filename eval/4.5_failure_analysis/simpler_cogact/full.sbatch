#!/bin/bash
#SBATCH -J c_s0rp_full
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:45:00
#SBATCH -o $PROJ_ROOT/bench2_cogact/step0_replay/logs/full_%j.log
# 全量回放：2400 条，按 MODE 指定的观测模式（默认 state_dict）。不加载策略。
source $PROJ_ROOT/bench2_cogact/b2c/jobs/common.sh
export OMP_NUM_THREADS=1
MODE=${MODE:-state_dict}
D=$PROJ_ROOT/bench2_cogact/step0_replay; REC=$PROJ_ROOT/bench2_cogact/b2c/out_formal
cd $D
P=()
for f in $D/full_*.txt; do
  n=$(basename $f .txt)
  python replay_cogact.py --rec-root $REC --out $D/out --obs-mode $MODE --list $f --skip-done 2>&1 | grep -v "$NOISE" > $D/logs/full_${SLURM_JOB_ID}_$n.log &
  P+=($!)
done
for p in "${P[@]}"; do wait $p; echo "pid $p exit $?"; done
echo "done files: $(find $D/out -name '*.npz' | wc -l)"
echo "errors: $(grep -h 'ERROR' $D/logs/full_${SLURM_JOB_ID}_*.log | wc -l)"
echo "match=False: $(grep -h 'match=False' $D/logs/full_${SLURM_JOB_ID}_*.log | wc -l)"
echo "init=False: $(grep -h 'init=False\|qpos0=False\|ep=False\|freq=False' $D/logs/full_${SLURM_JOB_ID}_*.log | wc -l)"
python cogact_feat.py $D/out $REC $D/cogact_replay_feats.json
