#!/bin/bash
#SBATCH --job-name=toy_t0
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
python -X faulthandler tools/t0check_toy.py --offsets ${OFFS:-40,45} --tasks ${TASKS:-0,1,2,3,4,5,6,7,8,9} --out $O/t0_${SLURM_JOB_ID}.jsonl --save-png 5 --png-dir $O/png_${SLURM_JOB_ID}
echo "exit=$? end=$(date)"
