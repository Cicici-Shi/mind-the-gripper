#!/bin/bash
#SBATCH -J s0rp_full
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=00:32:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/step0_replay/logs/full_%j.log
# 全量回放：state_dict 模式（校验作业 <job> 上 40 对回合与 rgbd 模式逐位相同，且 success_steps 40/40 与记录一致）
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
hostname; nvidia-smi -L
D=$PROJ_ROOT/bench2_simpler/step0_replay; REC=$PROJ_ROOT/bench2_simpler/b2/out_formal
cd $D
P=()
for f in $D/full_*.txt; do
  n=$(basename $f .txt)
  python replay.py --rec-root $REC --out $D/out --obs-mode state_dict --list $f --skip-done > $D/logs/full_${SLURM_JOB_ID}_$n.log 2>&1 &
  P+=($!)
done
for p in "${P[@]}"; do wait $p; echo "pid $p exit $?"; done
echo "done files: $(find $D/out -name '*.npz' | wc -l)"
grep -h "ERROR" $D/logs/full_${SLURM_JOB_ID}_*.log | head
grep -h "match=False" $D/logs/full_${SLURM_JOB_ID}_*.log | wc -l
