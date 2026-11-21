#!/bin/bash
#SBATCH -J s0rp_val
#SBATCH -p gpu
#SBATCH --gres=gpu:a100-40:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=00:15:00
#SBATCH -o $PROJ_ROOT/bench2_simpler/step0_replay/logs/val_%j.log
set -u
export PYTHONDONTWRITEBYTECODE=1 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
unset VK_DRIVER_FILES VK_ADD_DRIVER_FILES
export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
source ~/miniforge3/etc/profile.d/conda.sh; conda activate bench2_simpler
hostname; nvidia-smi -L
D=$PROJ_ROOT/bench2_simpler/step0_replay; REC=$PROJ_ROOT/bench2_simpler/b2/out_formal
cd $D
P=()
for b in B1 B2 F1 F2; do
  for m in rgbd state_dict; do
    python replay.py --rec-root $REC --out $D/val_$m --obs-mode $m --list $D/val_$b.txt > $D/logs/val_${SLURM_JOB_ID}_${b}_${m}.log 2>&1 &
    P+=($!)
  done
done
for p in "${P[@]}"; do wait $p; echo "pid $p exit $?"; done
grep -h "^\[rp\]" $D/logs/val_${SLURM_JOB_ID}_*.log | sort
