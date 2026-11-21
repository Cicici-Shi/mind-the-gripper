#!/bin/bash
#SBATCH --job-name=toy_arm
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nv:1
# (site-specific #SBATCH --exclude list removed)
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.log
export TMPDIR=$HOME/tmp PYTHONFAULTHANDLER=1 MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 ORIGINAL_ONLY=1 PYTHONDONTWRITEBYTECODE=1; unset DISPLAY
nvidia-smi -L >/dev/null 2>&1 || { echo "no GPU visible on $(hostname)"; exit 3; }
source $HOME/miniforge3/etc/profile.d/conda.sh; conda activate oft
T=$PROJ_ROOT/embody_toy; cd $T; O=$T/out/t0check; mkdir -p $O
echo "node=$(hostname) gpu=$(nvidia-smi --query-gpu=name --format=csv,noheader) start=$(date)"
python -X faulthandler tools/t0check_arm.py --offsets ${OFFS:-40,45} --out $O/arm_${SLURM_JOB_ID}.jsonl --save-png 0
echo "exit=$? end=$(date)"
